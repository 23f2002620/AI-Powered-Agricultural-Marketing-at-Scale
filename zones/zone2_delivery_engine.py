"""
zones/zone2_delivery_engine.py
════════════════════════════════════════════════════════════════════════════════
ZONE 2 — DELIVERY ENGINE  (runs continuously, flushes queue at send_at times)
════════════════════════════════════════════════════════════════════════════════

Responsibilities
----------------
1. Queue reader    — polls results/message_queue_<date>.jsonl for records
                     whose send_at ≤ now and status == "pending"
2. Channel router  — reads device_type from the pre-resolved channel field
                     routes to the correct adapter (no model calls here):
                       feature phone   → IVR voice call  (pre-gen Sarvam audio, 0 data cost)
                       smartphone 2G   → WhatsApp text   (pre-rendered ~1 KB)
                       smartphone 4G   → WhatsApp rich   (text + image, audio optional)
                       no phone / rep  → field rep brief (rep app push)
                       SMS fallback    → 160-char text   (0 data cost)
3. Delivery stub   — simulates / calls real gateway adapters
                     (WhatsApp Business API · Sarvam TTS IVR · SMS gateway · rep push)
4. Receipt handler — receives webhook callbacks (delivered/read/clicked, IVR keypress,
                     POS purchase) and writes them back to the queue record so
                     Zone 3 can read them in one place.
5. Dispatch log    — appends to results/dispatch_log_<date>.jsonl for audit trail

Key principle
-------------
NO model is called here.  Every payload (text, audio URL, image URL) was
pre-rendered by Zone 1.  The delivery engine is a pure I/O router.

Entry point (runs as a daemon, checks every 60 s):
    python -m zones.zone2_delivery_engine
    python -m zones.zone2_delivery_engine --queue results/message_queue_20260520.jsonl
    python -m zones.zone2_delivery_engine --once   # flush once and exit (for testing)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── project root ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR   = ROOT / "results"
DISPATCH_LOG  = "dispatch_log_{date}.jsonl"
POLL_INTERVAL = 60   # seconds between queue scans

# ── reward values (mirrored in Zone 3 — single source of truth here) ─────────
REWARD_DELIVERED  = 0.1
REWARD_OPENED     = 0.3
REWARD_CLICKED    = 0.7
REWARD_PURCHASED  = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Gateway adapters  (stubs — replace with real SDK calls in production)
# ─────────────────────────────────────────────────────────────────────────────

class WhatsAppAdapter:
    """
    Sends pre-rendered WhatsApp messages via WhatsApp Business API.
    No model calls.  Payload was fully assembled by Zone 1.

    In production:
        POST https://graph.facebook.com/v18.0/{phone_number_id}/messages
        with the pre-built template / free-form message body.
    """

    def send_text(self, grower_id: str, phone: str, text: str,
                  language: str) -> dict:
        # Production: call Meta Graph API
        print(f"    [WA TEXT] {grower_id} → {phone[:6]}… ({language}, {len(text)} chars)")
        return {"status": "sent", "message_id": f"wa_text_{grower_id}_{int(time.time())}"}

    def send_rich(self, grower_id: str, phone: str, text: str,
                  image_url: Optional[str], audio_url: Optional[str],
                  language: str) -> dict:
        # Production: send image + caption; optional audio note
        img = "✓ image" if image_url else "✗ no image"
        aud = "✓ audio" if audio_url else "✗ no audio"
        print(f"    [WA RICH] {grower_id} → {phone[:6]}… ({language}, {img}, {aud})")
        return {"status": "sent", "message_id": f"wa_rich_{grower_id}_{int(time.time())}"}


class IVRAdapter:
    """
    Triggers a pre-generated IVR voice call via Sarvam AI or Exotel.
    Audio blob was synthesised by Zone 1 (Bhashini → Sarvam → pyttsx3 fallback).
    0 data cost for the farmer — just a GSM voice call.
    """

    def call(self, grower_id: str, phone: str, audio_url: Optional[str],
             text_fallback: str, language: str) -> dict:
        # Production: POST to Exotel/Kaleyra with pre-gen audio URL
        src = "audio blob" if audio_url else "TTS-at-call-time"
        print(f"    [IVR    ] {grower_id} → {phone[:6]}… ({language}, src={src})")
        return {"status": "initiated", "call_id": f"ivr_{grower_id}_{int(time.time())}"}


class SMSAdapter:
    """
    Sends 160-char pre-rendered SMS.  0 data cost.
    Template was filled by Zone 1 content engine.
    """

    def send(self, grower_id: str, phone: str, text: str,
             language: str) -> dict:
        # Production: POST to Twilio / Kaleyra SMS API
        snippet = text[:40].replace("\n", " ")
        print(f"    [SMS    ] {grower_id} → {phone[:6]}… ({language}) '{snippet}…'")
        return {"status": "sent", "sms_id": f"sms_{grower_id}_{int(time.time())}"}


class FieldRepAdapter:
    """
    Pushes a field-rep talking-points brief to the rep's mobile app.
    Triggered when receptivity model predicted low digital conversion
    AND the grower has an assigned rep in reps_territory.csv.
    """

    def push_brief(self, grower_id: str, rep_id: str, brief_text: str,
                   language: str) -> dict:
        # Production: POST to rep mobile app notification API
        print(f"    [REP    ] {grower_id} → rep {rep_id} ({language}, {len(brief_text)} chars)")
        return {"status": "queued", "brief_id": f"rep_{grower_id}_{int(time.time())}"}


# ─────────────────────────────────────────────────────────────────────────────
# Queue manager  (reads / writes the JSONL queue file atomically)
# ─────────────────────────────────────────────────────────────────────────────

class QueueManager:
    """
    Reads the pre-rendered JSONL queue, yields pending records whose
    send_at ≤ now, and updates their status in-place.

    File format: one JSON object per line (QueueRecord from Zone 1).
    Extra fields written by Zone 2: dispatch_status, gateway_id, dispatched_at.
    Extra fields written by Zone 3: delivered, opened, clicked, purchased, reward.
    """

    def __init__(self, queue_path: Path):
        self.path = queue_path
        self._records: list[dict] = []
        self._load()

    def _load(self):
        if not self.path.exists():
            print(f"  Queue file not found: {self.path}")
            return
        with self.path.open(encoding="utf-8") as fh:
            self._records = [json.loads(line) for line in fh if line.strip()]
        print(f"  Loaded {len(self._records):,} records from {self.path.name}")

    def _save(self):
        with self.path.open("w", encoding="utf-8") as fh:
            for rec in self._records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def pending_now(self, now: datetime) -> list[dict]:
        """Return records that are due and not yet dispatched."""
        return [
            r for r in self._records
            if r.get("dispatch_status", "pending") == "pending"
            and datetime.fromisoformat(r["send_at"]) <= now
        ]

    def mark_dispatched(self, grower_id: str, gateway_id: str):
        for r in self._records:
            if r["grower_id"] == grower_id:
                r["dispatch_status"] = "dispatched"
                r["gateway_id"]      = gateway_id
                r["dispatched_at"]   = datetime.utcnow().isoformat()
                break
        self._save()

    def mark_failed(self, grower_id: str, error: str):
        for r in self._records:
            if r["grower_id"] == grower_id:
                r["dispatch_status"] = "failed"
                r["error"]           = error
                break
        self._save()

    def record_receipt(
        self, grower_id: str, *,
        delivered: Optional[bool] = None,
        opened:    Optional[bool] = None,
        clicked:   Optional[bool] = None,
        purchased: Optional[bool] = None,
    ) -> float:
        """
        Called by the webhook handler when a delivery receipt / callback arrives.
        Updates the queue record and computes the reward value for Zone 3.
        Returns the reward float (Zone 3 will inject it into the bandit).
        """
        reward = 0.0
        for r in self._records:
            if r["grower_id"] == grower_id:
                if delivered is not None:
                    r["delivered"] = delivered
                    reward += REWARD_DELIVERED if delivered else 0.0
                if opened is not None:
                    r["opened"] = opened
                    reward += REWARD_OPENED if opened else 0.0
                if clicked is not None:
                    r["clicked"] = clicked
                    reward += REWARD_CLICKED if clicked else 0.0
                if purchased is not None:
                    r["purchased"] = purchased
                    reward += REWARD_PURCHASED if purchased else 0.0
                r["reward"] = r.get("reward", 0.0) + reward  # accumulate
                break
        self._save()
        return reward

    def all_records(self) -> list[dict]:
        return self._records


# ─────────────────────────────────────────────────────────────────────────────
# Channel router
# ─────────────────────────────────────────────────────────────────────────────

class ChannelRouter:
    """
    Reads device_type (already resolved to a channel by Zone 1) from each
    queue record and dispatches to the correct gateway adapter.

    Never re-derives the channel — Zone 1 already made the optimal decision.
    The router just executes it.
    """

    def __init__(self):
        self.wa    = WhatsAppAdapter()
        self.ivr   = IVRAdapter()
        self.sms   = SMSAdapter()
        self.rep   = FieldRepAdapter()

    def _lookup_phone(self, grower_id: str) -> str:
        """
        In production: query growers.csv or a CRM for the farmer's phone number.
        Stub returns a synthetic number for testing.
        """
        return f"+91-{int(grower_id.replace('GRW', '').replace('_', '').strip() or 9999990000):010d}"

    def _lookup_rep(self, grower_id: str) -> str:
        """Stub rep_id lookup.  Production reads reps_territory.csv."""
        return f"REP_{grower_id[-4:]}"

    def dispatch(self, record: dict) -> dict:
        """
        Route one pre-rendered queue record to the appropriate gateway.
        Returns a result dict with status and gateway_id.

        CRITICAL: no model call happens here.  text / audio_url / image_url
        were all set by Zone 1.
        """
        channel  = record.get("channel", "whatsapp_text")
        gid      = record["grower_id"]
        language = record.get("language", "Hindi")
        text     = record.get("text", "")
        audio    = record.get("audio_url")
        image    = record.get("image_url")
        phone    = self._lookup_phone(gid)

        if channel == "whatsapp_rich":
            # Smartphone 4G: text + image, audio optional
            result = self.wa.send_rich(gid, phone, text, image, audio, language)

        elif channel == "whatsapp_text":
            # Smartphone 2G: text only, lightweight (~1 KB)
            result = self.wa.send_text(gid, phone, text, language)

        elif channel == "ivr_voice":
            # Feature phone: pre-gen Sarvam audio, 0 data cost for farmer
            result = self.ivr.call(gid, phone, audio, text, language)

        elif channel == "sms":
            # SMS fallback: 160-char template-filled, 0 data cost
            result = self.sms.send(gid, phone, text[:160], language)

        elif channel == "field_rep_brief":
            # No phone / low receptivity: push brief to rep app
            rep_id = self._lookup_rep(gid)
            result = self.rep.push_brief(gid, rep_id, text, language)

        else:
            # Unknown channel: fall back to SMS
            result = self.sms.send(gid, phone, text[:160], language)

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Delivery Engine
# ─────────────────────────────────────────────────────────────────────────────

class DeliveryEngine:
    """
    Zone 2 — the runner that polls the queue and flushes due records.

    In production this is a long-running daemon (cron or systemd service).
    The --once flag lets you run it as a one-shot (useful in tests / CI).
    """

    def __init__(self, queue_path: Optional[Path] = None):
        self.router       = ChannelRouter()
        self.queue_path   = queue_path
        self._dispatch_log: list[dict] = []

    # ── Queue discovery ───────────────────────────────────────────────────────

    def _resolve_queue(self) -> Optional[Path]:
        """Find the most recent message queue file."""
        if self.queue_path and self.queue_path.exists():
            return self.queue_path
        candidates = sorted(RESULTS_DIR.glob("message_queue_*.jsonl"), reverse=True)
        return candidates[0] if candidates else None

    # ── Single flush pass ─────────────────────────────────────────────────────

    def flush_once(self) -> int:
        """
        One delivery pass: find all pending-and-due records, route them,
        log results.  Returns the number of messages dispatched.
        """
        queue_file = self._resolve_queue()
        if not queue_file:
            print("  No queue file found. Zone 1 has not run yet.")
            return 0

        now = datetime.utcnow()
        mgr = QueueManager(queue_file)
        due = mgr.pending_now(now)

        if not due:
            print(f"  No messages due at {now.strftime('%H:%M:%S UTC')}.")
            return 0

        print(f"\n{'─'*60}")
        print(f"Zone 2 — flushing {len(due)} due message(s) at {now.strftime('%H:%M:%S UTC')}")
        print(f"{'─'*60}")

        dispatched = 0
        for record in due:
            gid     = record["grower_id"]
            channel = record.get("channel", "?")
            is_emerg = record.get("is_emergency", False)
            prefix  = "⚠  EMERGENCY" if is_emerg else "  "
            print(f"{prefix} [{channel:>18}] grower={gid}")
            try:
                result = self.router.dispatch(record)
                mgr.mark_dispatched(gid, result.get("message_id") or
                                         result.get("call_id") or
                                         result.get("sms_id") or
                                         result.get("brief_id") or "?")
                self._log(record, result, status="dispatched")
                dispatched += 1
            except Exception as exc:
                err = str(exc)
                print(f"  ✗ FAILED {gid}: {err}")
                mgr.mark_failed(gid, err)
                self._log(record, {}, status="failed", error=err)

        # Write dispatch log
        log_path = RESULTS_DIR / DISPATCH_LOG.format(date=now.strftime("%Y%m%d"))
        with log_path.open("a", encoding="utf-8") as fh:
            for entry in self._dispatch_log:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._dispatch_log.clear()

        print(f"\n  Dispatched: {dispatched}/{len(due)} this pass. Log → {log_path.name}")
        return dispatched

    # ── Webhook receipt handler ───────────────────────────────────────────────

    def handle_receipt(
        self,
        grower_id: str,
        queue_file: Optional[Path] = None, *,
        delivered: Optional[bool] = None,
        opened:    Optional[bool] = None,
        clicked:   Optional[bool] = None,
        purchased: Optional[bool] = None,
    ) -> float:
        """
        Called when a delivery receipt / engagement callback arrives
        (WhatsApp webhook: delivered/read/clicked;
         IVR: call-answered / keypress-1;
         POS: purchase-within-14-days from nightly attribution run in Zone 3).

        Records the event in the queue file and returns the computed reward
        value so Zone 3 can feed it straight to the bandit.

        reward table:
            delivered  → 0.1
            opened     → 0.3
            clicked    → 0.7
            purchased  → 1.0
        """
        qf  = queue_file or self._resolve_queue()
        mgr = QueueManager(qf)
        reward = mgr.record_receipt(
            grower_id,
            delivered=delivered,
            opened=opened,
            clicked=clicked,
            purchased=purchased,
        )
        event = {k: v for k, v in dict(
            delivered=delivered, opened=opened,
            clicked=clicked, purchased=purchased,
        ).items() if v is not None}
        print(f"  Receipt [{grower_id}] {event} → reward={reward:.2f}")
        return reward

    # ── Daemon loop ───────────────────────────────────────────────────────────

    def run_daemon(self, poll_interval: int = POLL_INTERVAL):
        """
        Continuous delivery daemon.  Polls every `poll_interval` seconds.
        Designed to run as a systemd service or supervised process.
        """
        print(f"\n{'═'*60}")
        print(f"ZONE 2 — DELIVERY ENGINE (daemon, poll={poll_interval}s)")
        print(f"{'═'*60}")
        try:
            while True:
                self.flush_once()
                time.sleep(poll_interval)
        except KeyboardInterrupt:
            print("\nZone 2 daemon stopped.")

    # ── Internal logging ──────────────────────────────────────────────────────

    def _log(self, record: dict, result: dict, status: str, error: str = ""):
        self._dispatch_log.append({
            "grower_id":    record.get("grower_id"),
            "campaign_id":  record.get("campaign_id"),
            "channel":      record.get("channel"),
            "language":     record.get("language"),
            "is_emergency": record.get("is_emergency"),
            "send_at":      record.get("send_at"),
            "dispatched_at": datetime.utcnow().isoformat(),
            "status":       status,
            "gateway_id":   result.get("message_id") or result.get("call_id") or "",
            "error":        error,
        })


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _cli():
    p = argparse.ArgumentParser(description="Zone 2 — Delivery Engine")
    p.add_argument("--queue",    default=None,   help="Path to a specific queue JSONL file")
    p.add_argument("--once",     action="store_true", help="Flush once and exit")
    p.add_argument("--poll",     type=int, default=POLL_INTERVAL,
                                 help=f"Daemon poll interval in seconds (default {POLL_INTERVAL})")
    # Receipt simulation (for testing)
    p.add_argument("--receipt",  default=None,   help="Simulate receipt for grower_id")
    p.add_argument("--delivered", action="store_true")
    p.add_argument("--opened",    action="store_true")
    p.add_argument("--clicked",   action="store_true")
    p.add_argument("--purchased", action="store_true")
    args = p.parse_args()

    queue = Path(args.queue) if args.queue else None
    engine = DeliveryEngine(queue_path=queue)

    if args.receipt:
        engine.handle_receipt(
            args.receipt,
            delivered=args.delivered or None,
            opened=args.opened or None,
            clicked=args.clicked or None,
            purchased=args.purchased or None,
        )
    elif args.once:
        engine.flush_once()
    else:
        engine.run_daemon(poll_interval=args.poll)


if __name__ == "__main__":
    _cli()