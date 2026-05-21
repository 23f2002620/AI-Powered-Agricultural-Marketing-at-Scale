"""
zones/zone3_feedback_loop.py
════════════════════════════════════════════════════════════════════════════════
ZONE 3 — FEEDBACK LOOP  (runs nightly before Zone 1, closes the loop)
════════════════════════════════════════════════════════════════════════════════

Responsibilities
----------------
1. Receipt aggregation — reads all dispatch_log_*.jsonl files and collects
                         WhatsApp delivered/read/clicked callbacks,
                         IVR call-answered / keypress-1 events,
                         from the queue files written by Zone 2.
2. POS attribution     — joins retailer_pos.csv against campaign sends
                         within a 14-day attribution window to determine
                         which sends resulted in a purchase.
3. Reward computation  — converts each engagement / purchase signal into a
                         scalar reward:
                           delivered  → 0.1
                           opened     → 0.3
                           clicked    → 0.7
                           purchased  → 1.0
4. Bandit reward injection — calls record_reward() on the LinUCB bandit for
                         every dispatched message, updates arm weights, and
                         saves models/linucb_bandit.pkl so Zone 1's next run
                         has a smarter bandit.
5. Reporting           — writes results/feedback_report_<date>.json with
                         conversion stats and arm-level performance.

Schedule
--------
Run nightly BEFORE Zone 1 (so the bandit is refreshed before the new batch):
    # 10:30 PM — feedback loop (uses yesterday's queue + POS data)
    30 22 * * * cd /opt/agri_marketing && python -m zones.zone3_feedback_loop
    # 11:00 PM — Zone 1 nightly batch (bandit now up to date)
    0  23 * * * cd /opt/agri_marketing && python -m zones.zone1_nightly_batch

Entry point:
    python -m zones.zone3_feedback_loop
    python -m zones.zone3_feedback_loop --window 14 --queue results/message_queue_20260520.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

# ── project root ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from engines.targeting_optimizer import (
    CONTEXT_DIM,
    N_ARMS,
    GrowerContext,
    LinUCBAgent,
    record_reward,
)
from utils.external_signals import enrich_grower_context

DATA_DIR    = ROOT / "data"
MODELS_DIR  = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# ── reward table (single source of truth — Zone 2 mirrors these values) ──────
REWARD_DELIVERED  = 0.1
REWARD_OPENED     = 0.3
REWARD_CLICKED    = 0.7
REWARD_PURCHASED  = 1.0
ATTRIBUTION_WINDOW_DAYS = 14


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_bandit() -> LinUCBAgent:
    path = MODELS_DIR / "linucb_bandit.pkl"
    if path.exists():
        agent = LinUCBAgent.load(path)
        print(f"  Bandit loaded: {agent.total_rounds:,} rounds in history")
        return agent
    print("  No saved bandit found — initialising fresh agent")
    return LinUCBAgent(n_arms=N_ARMS, context_dim=CONTEXT_DIM, alpha=0.5)


def _save_bandit(agent: LinUCBAgent):
    path = MODELS_DIR / "linucb_bandit.pkl"
    agent.save(path)
    print(f"  Bandit saved → {path}  (total rounds: {agent.total_rounds:,})")


def _load_queue_records(queue_paths: list[Path]) -> list[dict]:
    """Load all queue records from one or more JSONL files."""
    records = []
    for qp in queue_paths:
        if not qp.exists():
            continue
        with qp.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    records.append(json.loads(line))
    print(f"  Loaded {len(records):,} queue records from {len(queue_paths)} file(s)")
    return records


def _build_grower_meta() -> dict[str, dict]:
    """Build grower_id → metadata dict from growers.csv + WA history."""
    import json as _json
    growers = pd.read_csv(DATA_DIR / "growers.csv")

    def _safe(s, key, default):
        try:
            return _json.loads(s).get(key, default) if pd.notna(s) else default
        except Exception:
            return default

    growers["crop"] = growers["grower_crop_calendar"].apply(
        lambda s: _safe(s, "crop", "wheat")
    )

    wa = pd.read_csv(DATA_DIR / "whatsapp_campaign.csv")
    wa_agg = wa.groupby("grower_id").agg(
        wa_open_rate=("opened_status", "mean"),
        wa_click_rate=("clicked_status", "mean"),
        wa_messages=("id", "count"),
    ).reset_index()
    growers = growers.merge(wa_agg, on="grower_id", how="left")
    growers[["wa_open_rate", "wa_click_rate", "wa_messages"]] = (
        growers[["wa_open_rate", "wa_click_rate", "wa_messages"]].fillna(0)
    )
    return growers.set_index("grower_id").to_dict("index")


# ─────────────────────────────────────────────────────────────────────────────
# Signal aggregators
# ─────────────────────────────────────────────────────────────────────────────

class EngagementSignals:
    """
    Source 1 — WhatsApp webhooks + IVR callbacks.
    These arrive asynchronously via Zone 2's handle_receipt(), which writes
    them back into the queue record's delivered / opened / clicked fields.
    This class simply reads those fields from the queue JSONL.
    """

    def extract(self, records: list[dict]) -> dict[str, dict]:
        """
        Returns grower_id → {delivered, opened, clicked, reward_engagement}
        for all records that have at least one engagement signal.
        """
        signals: dict[str, dict] = {}
        for r in records:
            gid = r.get("grower_id")
            if not gid:
                continue
            delivered = r.get("delivered")
            opened    = r.get("opened")
            clicked   = r.get("clicked")
            if any(v is not None for v in [delivered, opened, clicked]):
                reward = (
                    (REWARD_DELIVERED if delivered else 0.0) +
                    (REWARD_OPENED    if opened    else 0.0) +
                    (REWARD_CLICKED   if clicked   else 0.0)
                )
                signals[gid] = {
                    "delivered":          delivered,
                    "opened":             opened,
                    "clicked":            clicked,
                    "reward_engagement":  reward,
                }
        print(f"  Engagement signals : {len(signals):,} growers with WA/IVR callbacks")
        return signals


class POSAttributionEngine:
    """
    Source 2 — Nightly POS attribution.
    Joins retailer_pos.csv with campaign sends within a 14-day window.
    A grower is attributed a purchase if ANY POS transaction for the
    campaign_product (crop-mapped SKU) occurs at a retailer in the
    same tehsil within `window_days` of send_at.
    """

    def __init__(self, window_days: int = ATTRIBUTION_WINDOW_DAYS):
        self.window_days = window_days
        self._pos_lookup = self._build_pos_lookup()

    def _build_pos_lookup(self) -> dict:
        """Build (tehsil, sku_name, date) → qty lookup for fast window queries."""
        pos_path  = DATA_DIR / "retailer_pos.csv"
        ret_path  = DATA_DIR / "retailers.csv"
        if not pos_path.exists() or not ret_path.exists():
            print("  POS / retailers data not found — skipping POS attribution")
            return {}

        pos  = pd.read_csv(pos_path)
        rets = pd.read_csv(ret_path)[["retailer_id", "tehsil"]].drop_duplicates()
        pos["transaction_date"] = pd.to_datetime(pos["transaction_date"])
        pos = pos.merge(rets, on="retailer_id", how="left")

        lookup: dict[tuple, float] = {}
        for _, r in pos.iterrows():
            key = (r["tehsil"], r["sku_name"], r["transaction_date"].date())
            lookup[key] = lookup.get(key, 0) + r.get("sku_qty", 1)
        return lookup

    # crop → typical promoted SKU (mirrors StockChecker.CROP_TO_PRODUCT)
    _CROP_SKU = {
        "wheat":     "Tilt 250 EC",
        "mustard":   "Score 250 EC",
        "chickpea":  "Amistar 250 SC",
        "potato":    "Kavach 75 WP",
        "barley":    "Tilt 250 EC",
        "lentil":    "Amistar 250 SC",
        "safflower": "Score 250 EC",
        "cumin":     "Tilt 250 EC",
        "maize":     "Amistar 250 SC",
    }

    def attribute(
        self,
        records: list[dict],
        grower_meta: dict[str, dict],
    ) -> dict[str, bool]:
        """
        Returns grower_id → True/False (purchased within window).
        """
        if not self._pos_lookup:
            return {}

        results: dict[str, bool] = {}
        for r in records:
            gid      = r.get("grower_id")
            tehsil   = grower_meta.get(gid, {}).get("tehsil", "")
            crop     = grower_meta.get(gid, {}).get("crop", "wheat")
            sku      = self._CROP_SKU.get(crop, "Tilt 250 EC")
            send_str = r.get("send_at")
            if not (gid and tehsil and send_str):
                continue
            send_dt  = datetime.fromisoformat(send_str)
            found    = False
            cur      = send_dt.date()
            end      = (send_dt + timedelta(days=self.window_days)).date()
            while cur <= end:
                if self._pos_lookup.get((tehsil, sku, cur), 0) > 0:
                    found = True
                    break
                cur += timedelta(days=1)
            results[gid] = found

        purchased = sum(1 for v in results.values() if v)
        total     = len(results)
        rate      = purchased / max(total, 1)
        print(f"  POS attribution    : {purchased:,}/{total:,} growers purchased "
              f"within {self.window_days}d ({rate:.2%})")
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Bandit reward injector
# ─────────────────────────────────────────────────────────────────────────────

class BanditRewardInjector:
    """
    Converts engagement + POS signals into LinUCB arm updates.

    Reward = sum of all applicable signals for a grower.
    LinUCB updates the arm weight vector A_inv and b for the arm
    that was selected during Zone 1's batch run.
    """

    def inject(
        self,
        records: list[dict],
        engagement: dict[str, dict],
        pos_purchases: dict[str, bool],
        grower_meta: dict[str, dict],
        agent: LinUCBAgent,
    ) -> dict:
        """
        Iterates over all dispatched queue records, combines engagement
        and POS signals, and calls record_reward() once per grower.

        Returns a summary dict for the feedback report.
        """
        updated   = 0
        total_rew = 0.0
        arm_rewards: dict[str, list] = {}

        for r in records:
            gid = r.get("grower_id", "")
            arm = r.get("bandit_arm", "")
            if not arm or "|" not in arm:
                continue  # no arm recorded → can't update

            meta      = grower_meta.get(gid, {})
            eng       = engagement.get(gid, {})
            purchased = pos_purchases.get(gid, False)

            reward = eng.get("reward_engagement", 0.0)
            if purchased:
                reward += REWARD_PURCHASED

            if reward == 0.0:
                # Even a non-responding grower is a signal (reward=0 = arm underperformed)
                # Still update so the bandit learns from non-events
                pass

            # Build a minimal GrowerContext for the bandit update
            ext = enrich_grower_context(
                tehsil=str(meta.get("tehsil", "")),
                state=str(meta.get("state", "")),
                crop=str(meta.get("crop", "wheat")),
            )
            ctx = GrowerContext(
                grower_id=gid,
                device_type=str(meta.get("device_type", "unknown")),
                language=str(meta.get("language", "Hindi")),
                crop=str(meta.get("crop", "wheat")),
                growth_stage="vegetative",
                season_progress=0.5,
                days_to_next_stage=14,
                hist_open_rate=float(meta.get("wa_open_rate", 0)),
                hist_click_rate=float(meta.get("wa_click_rate", 0)),
                tehsil_stock_rate=float(r.get("stock_rate", 0.7)),
                days_since_rep_visit=30,
                digital_literacy_score=0.7,
                grower_age=int(meta.get("grower_age", 45)),
                farm_size_acres=float(meta.get("grower_farm_size", 2.0)),
                offline_campaign_attended=bool(meta.get("offline_campaign_attended", False)),
                product_scan_done=bool(meta.get("product_scan", False)),
                weather_risk_score=ext.weather.weather_risk_score,
                pest_pressure_index=ext.pest.pest_pressure_index,
            )

            record_reward(agent, ctx, arm, reward)
            updated   += 1
            total_rew += reward

            # Track per-arm stats
            arm_rewards.setdefault(arm, []).append(reward)

        avg_reward = total_rew / max(updated, 1)
        arm_perf   = {arm: sum(v) / len(v) for arm, v in arm_rewards.items()}
        top_arms   = sorted(arm_perf.items(), key=lambda x: -x[1])[:5]

        print(f"  Bandit updates     : {updated:,} records")
        print(f"  Avg reward         : {avg_reward:.4f}")
        print(f"  Top 5 arms         : {top_arms}")

        return {
            "updated":     updated,
            "avg_reward":  avg_reward,
            "top_arms":    top_arms,
            "arm_rewards": arm_perf,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Zone 3 runner
# ─────────────────────────────────────────────────────────────────────────────

class FeedbackLoop:
    """
    Zone 3 — closes the loop between delivery outcomes and the bandit.

    Run this nightly before Zone 1 so the bandit is updated before the
    next batch is planned.
    """

    def __init__(self, window_days: int = ATTRIBUTION_WINDOW_DAYS):
        self.window_days = window_days

    def run(
        self,
        queue_paths: Optional[list[Path]] = None,
        dry_run: bool = False,
    ) -> dict:
        """
        Full Zone 3 pipeline:
          1. Load queue records (Zone 2 output)
          2. Extract engagement signals (WA / IVR callbacks in queue fields)
          3. Run POS attribution (14-day window join)
          4. Inject rewards into LinUCB bandit
          5. Save updated bandit
          6. Write feedback report

        Returns summary stats dict.
        """
        now = datetime.utcnow()
        print(f"\n{'═'*60}")
        print(f"ZONE 3 — FEEDBACK LOOP  {now.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"{'═'*60}\n")

        # ── 1. Discover queue files ───────────────────────────────────────────
        print("─" * 60)
        print("Step 1 │ Loading queue records")
        if not queue_paths:
            queue_paths = sorted(RESULTS_DIR.glob("message_queue_*.jsonl"), reverse=True)[:7]
        records = _load_queue_records(queue_paths)
        if not records:
            print("  No records found. Exiting Zone 3.")
            return {"updated": 0}

        dispatched = [r for r in records if r.get("dispatch_status") == "dispatched"]
        print(f"  Dispatched records : {len(dispatched):,} / {len(records):,} total")

        # ── 2. Engagement signals ─────────────────────────────────────────────
        print("\n─" * 60)
        print("Step 2 │ Extracting engagement signals (WA webhooks + IVR callbacks)")
        engagement = EngagementSignals().extract(dispatched)

        # ── 3. POS attribution ────────────────────────────────────────────────
        print("\n─" * 60)
        print(f"Step 3 │ POS attribution ({self.window_days}-day window)")
        grower_meta   = _build_grower_meta()
        pos_purchases = POSAttributionEngine(self.window_days).attribute(dispatched, grower_meta)

        # ── 4. Bandit reward injection ────────────────────────────────────────
        print("\n─" * 60)
        print("Step 4 │ Bandit reward injection — updating LinUCB arm weights")
        agent   = _load_bandit()
        summary = BanditRewardInjector().inject(
            dispatched, engagement, pos_purchases, grower_meta, agent
        )

        # ── 5. Save bandit ────────────────────────────────────────────────────
        if not dry_run:
            _save_bandit(agent)
        else:
            print("  [DRY RUN] Bandit NOT saved.")

        # ── 6. Feedback report ────────────────────────────────────────────────
        print("\n─" * 60)
        print("Step 5 │ Writing feedback report")

        channel_conv: dict[str, dict] = {}
        for r in dispatched:
            ch = r.get("channel", "unknown")
            if ch not in channel_conv:
                channel_conv[ch] = {"sent": 0, "purchased": 0}
            channel_conv[ch]["sent"] += 1
            if pos_purchases.get(r.get("grower_id", ""), False):
                channel_conv[ch]["purchased"] += 1

        report = {
            "run_at":               now.isoformat(),
            "queue_files":          [str(p) for p in queue_paths],
            "total_records":        len(records),
            "dispatched":           len(dispatched),
            "with_engagement":      len(engagement),
            "pos_attributed":       sum(1 for v in pos_purchases.values() if v),
            "conversion_rate":      sum(1 for v in pos_purchases.values() if v) / max(len(dispatched), 1),
            "bandit_updates":       summary["updated"],
            "avg_reward":           summary["avg_reward"],
            "top_5_arms":           summary["top_arms"],
            "channel_conversion":   channel_conv,
            "bandit_total_rounds":  agent.total_rounds,
        }

        report_path = RESULTS_DIR / f"feedback_report_{now.strftime('%Y%m%d')}.json"
        if not dry_run:
            with report_path.open("w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2, ensure_ascii=False)
            print(f"  Report → {report_path.name}")

        # Print summary
        print(f"\n{'─'*60}")
        print("Zone 3 — Feedback Loop complete")
        print(f"  Dispatched records   : {report['dispatched']:,}")
        print(f"  With engagement data : {report['with_engagement']:,}")
        print(f"  POS conversions      : {report['pos_attributed']:,}  "
              f"({report['conversion_rate']:.2%})")
        print(f"  Bandit rounds now    : {agent.total_rounds:,}")
        print(f"  Channel breakdown    :")
        for ch, stats in channel_conv.items():
            rate = stats["purchased"] / max(stats["sent"], 1)
            print(f"    {ch:>20}  sent={stats['sent']:>5}  conv={rate:.2%}")
        print(f"{'─'*60}\n")

        return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli():
    p = argparse.ArgumentParser(description="Zone 3 — Feedback Loop")
    p.add_argument("--window",  type=int, default=ATTRIBUTION_WINDOW_DAYS,
                                help="POS attribution window in days")
    p.add_argument("--queue",   nargs="*", default=None,
                                help="Specific queue JSONL file(s) to process")
    p.add_argument("--dry-run", action="store_true",
                                help="Run without saving bandit or report")
    args = p.parse_args()

    queue_paths = [Path(q) for q in args.queue] if args.queue else None
    FeedbackLoop(window_days=args.window).run(
        queue_paths=queue_paths,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    _cli()