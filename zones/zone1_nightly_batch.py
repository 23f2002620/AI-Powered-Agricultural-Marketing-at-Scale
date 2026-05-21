"""
zones/zone1_nightly_batch.py
════════════════════════════════════════════════════════════════════════════════
ZONE 1 — NIGHTLY BATCH  (runs on server at ~11 PM, internet available)
════════════════════════════════════════════════════════════════════════════════

Responsibilities
----------------
1. External signal fetch   — IMD weather · ICAR-NCIPM pest · Agmarknet mandi
                             → writes data/signal_cache.json
2. Grower-360 feature store — joins all 8 CSVs + signal cache → 1 row / grower
3. Four AI engines (parallel)
       E3  Receptivity   XGBoost scores every grower 0-1, filters below 0.03
       E4  Micro-segment HDBSCAN assigns segment + LLM template per grower
       E2  Targeting      LinUCB / Thompson picks channel × time × creative
       E1  Content gen    LLM renders text + Sarvam audio blob + infographic
4. Stock check gate        — suppresses growers where tehsil stock < 20 %
5. Queue write             — writes results/message_queue_<date>.jsonl
                             one record per grower, includes send_at timestamp
                             and pre-resolved channel → no model called at send time

Design constraints
------------------
* All heavy work (LLM, ML inference) happens HERE, never in Zone 2.
* Queue records are self-contained: channel, rendered text, audio URL,
  image URL, send_at. Zone 2 reads and flushes — it never calls a model.
* Emergency pest alerts bypass the normal schedule:
  send_at = now + EMERGENCY_CONTENT_WINDOW_HRS (default 2 h).

Entry point
-----------
    python -m zones.zone1_nightly_batch            # full run
    python -m zones.zone1_nightly_batch --dry-run  # skip queue write
    python -m zones.zone1_nightly_batch --crop wheat --max 200

Cron (server, runs nightly at 11 PM):
    0 23 * * * cd /opt/agri_marketing && python -m zones.zone1_nightly_batch
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

# ── project root on sys.path ────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Load API keys from .env at project root (must be before any os.getenv calls)
load_dotenv(ROOT / ".env")

from engines.content_generator import ContentFormat, ContentRequest, generate_content
from engines.micro_segmentation import MicroSegmentationEngine
from engines.receptivity_predictor import ReceptivityInput, ReceptivityPredictor
from engines.targeting_optimizer import (
    CONTEXT_DIM,
    N_ARMS,
    GrowerContext,
    LinUCBAgent,
    ThompsonSamplingAgent,
    decide,
)
from utils.crop_calendar import days_to_next_stage, get_growth_stage
from utils.external_signals import (
    ExternalContext,
    cache_signals_nightly,
    enrich_grower_context,
)
from utils.stock_checker import StockChecker

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR    = ROOT / "data"
MODELS_DIR  = ROOT / "models"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

QUEUE_FILE_TEMPLATE = "message_queue_{date}.jsonl"

# ── thresholds (mirror orchestrator constants) ────────────────────────────────
MIN_RECEPTIVITY_SCORE        = 0.03
MIN_STOCK_RATE               = 0.20
EMERGENCY_CONTENT_WINDOW_HRS = 2
PRE_STAGE_BOOST_FACTOR       = 1.35
PRE_STAGE_WINDOW             = (3, 7)   # days_to_next_stage inclusive

# ── channel-format map ────────────────────────────────────────────────────────
CHANNEL_TO_FORMAT: dict[str, ContentFormat] = {
    "whatsapp_rich":   ContentFormat.WHATSAPP_RICH,
    "whatsapp_text":   ContentFormat.WHATSAPP_TEXT,
    "ivr_voice":       ContentFormat.IVR_VOICE,
    "sms":             ContentFormat.SMS,
    "field_rep_brief": ContentFormat.FIELD_REP_BRIEF,
}


# ─────────────────────────────────────────────────────────────────────────────
# Queue record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QueueRecord:
    """One self-contained row written to the JSONL queue.

    Zone 2 reads this and flushes — it never needs to call a model or
    re-derive anything.  All decisions are final at write time.
    """
    grower_id:        str
    campaign_id:      str
    segment_id:       str

    # delivery co-ordinates
    channel:          str         # whatsapp_rich | whatsapp_text | ivr_voice | sms | field_rep_brief
    send_at:          str         # ISO-8601 — computed at batch time
    is_emergency:     bool

    # pre-rendered payload (Zone 2 sends this verbatim)
    text:             str         # rendered message text (or field-rep talking points)
    audio_url:        Optional[str]   # Sarvam / Bhashini TTS blob URL (IVR only)
    image_url:        Optional[str]   # lightweight infographic URL (WA rich only)
    language:         str

    # scoring metadata (for analytics / audit)
    receptivity_score: float
    ucb_score:         float
    stock_rate:        float
    weather_risk:      float
    pest_pressure:     float
    composite_urgency: float
    bandit_arm:        str
    creative_variant:  str
    time_slot:         str

    # feedback fields — written empty, filled by Zone 3
    delivered:         Optional[bool] = None
    opened:            Optional[bool] = None
    clicked:           Optional[bool] = None
    purchased:         Optional[bool] = None
    reward:            Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# Zone 1 runner
# ─────────────────────────────────────────────────────────────────────────────

class NightlyBatch:
    """
    Runs the full server-side intelligence pipeline and writes the message
    queue that Zone 2 will flush at scheduled send times.
    """

    def __init__(
        self,
        campaign_id: str = "CMP_RABI25",
        api_key: str = "",
        sarvam_api_key: str = "",
        bhashini_api_key: str = "",
        imd_key: str = "",
        ncipm_key: str = "",
        agmarknet_key: str = "",
    ):
        self.campaign_id      = campaign_id
        self.api_key          = api_key
        self.sarvam_api_key   = sarvam_api_key
        self.bhashini_api_key = bhashini_api_key
        self.imd_key          = imd_key
        self.ncipm_key        = ncipm_key
        self.agmarknet_key    = agmarknet_key

        print("Initialising Zone 1 — Nightly Batch")
        self._receptivity  = ReceptivityPredictor()
        self._segmentation = MicroSegmentationEngine()
        self._stock        = StockChecker()

        bandit_path = MODELS_DIR / "linucb_bandit.pkl"
        self._bandit = (
            LinUCBAgent.load(bandit_path)
            if bandit_path.exists()
            else LinUCBAgent(n_arms=N_ARMS, context_dim=CONTEXT_DIM, alpha=0.5)
        )
        self._thompson = ThompsonSamplingAgent(n_arms=N_ARMS)
        print(f"  Bandit: {self._bandit.total_rounds:,} rounds in history")
        print("  Zone 1 ready.\n")

    # ── Step 1: External signal fetch ────────────────────────────────────────

    def fetch_signals(self, growers: pd.DataFrame) -> dict[str, ExternalContext]:
        """
        Step 1  — External signal fetch (runs first at 11 PM nightly).
        Calls IMD / ICAR-NCIPM / Agmarknet for every unique (tehsil, crop, state)
        and writes data/signal_cache.json so daytime Zone 2 works offline.
        Returns an in-memory ext_cache keyed by "tehsil|crop".
        """
        print("─" * 60)
        print("Step 1 │ External signal fetch")

        combos = (
            growers[["tehsil", "crop", "state"]]
            .drop_duplicates()
            .values.tolist()
        )
        print(f"  Fetching signals for {len(combos)} (tehsil × crop) combinations …")

        # Write / refresh the cache file  →  Zone 2 reads this when offline
        cache_signals_nightly(
            [(r[0], r[1], r[2]) for r in combos],
            imd_key=self.imd_key,
            ncipm_key=self.ncipm_key,
            agmarknet_key=self.agmarknet_key,
        )
        print(f"  Cache written → {DATA_DIR / 'signal_cache.json'}")

        # Build in-memory lookup (same data, used immediately by later steps)
        ext_cache: dict[str, ExternalContext] = {}
        for tehsil, crop, state in combos:
            key = f"{tehsil}|{crop}"
            ext_cache[key] = enrich_grower_context(
                tehsil=tehsil, state=state, crop=crop,
                imd_key=self.imd_key,
                ncipm_key=self.ncipm_key,
                agmarknet_key=self.agmarknet_key,
            )

        n_emergency = sum(1 for c in ext_cache.values() if c.is_emergency)
        sources = {c.weather.source for c in ext_cache.values()}
        print(f"  Signal sources : {sources}")
        if n_emergency:
            print(f"  ⚠  {n_emergency} tehsil(s) with SEVERE pest pressure — emergency queue enabled")
        return ext_cache

    # ── Step 2: Grower-360 feature store ─────────────────────────────────────

    def build_feature_store(self) -> pd.DataFrame:
        """
        Step 2  — Grower-360 feature store.
        Joins all 8 source CSVs into a single wide dataframe (1 row / grower).
        """
        print("\n─" * 60)
        print("Step 2 │ Building Grower-360 feature store")

        growers = pd.read_csv(DATA_DIR / "growers.csv")

        # Parse crop calendar JSON
        def _safe(s, key, default):
            try:
                return json.loads(s).get(key, default) if pd.notna(s) else default
            except Exception:
                return default

        growers["crop"]          = growers["grower_crop_calendar"].apply(lambda s: _safe(s, "crop", "wheat"))
        growers["stages"]        = growers["grower_crop_calendar"].apply(lambda s: _safe(s, "stages", []))
        growers["sowing_start"]  = growers["grower_crop_calendar"].apply(lambda s: _safe(s, "sowing", {}).get("start"))
        growers["harvest_start"] = growers["grower_crop_calendar"].apply(lambda s: _safe(s, "harvest", {}).get("start"))

        # WhatsApp engagement history
        wa = pd.read_csv(DATA_DIR / "whatsapp_campaign.csv")
        wa_agg = wa.groupby("grower_id").agg(
            wa_open_rate=("opened_status", "mean"),
            wa_click_rate=("clicked_status", "mean"),
            wa_delivery_rate=("delivered_status", "mean"),
            wa_messages=("id", "count"),
        ).reset_index()
        growers = growers.merge(wa_agg, on="grower_id", how="left")
        growers[["wa_open_rate", "wa_click_rate", "wa_delivery_rate", "wa_messages"]] = (
            growers[["wa_open_rate", "wa_click_rate", "wa_delivery_rate", "wa_messages"]].fillna(0)
        )

        # Rep territory (for field-rep hybrid routing)
        # reps_territory.csv has: rep_id, territory_id, tehsil_list (JSON array)
        # There is no grower_id column — reps are mapped to growers via tehsil.
        reps_path = DATA_DIR / "reps_territory.csv"
        if reps_path.exists():
            reps_raw = pd.read_csv(reps_path)
            rep_rows = []
            for _, r in reps_raw.iterrows():
                try:
                    tehsils = json.loads(r["tehsil_list"])
                except Exception:
                    tehsils = [r.get("tehsil_list", "")]
                for t in tehsils:
                    rep_rows.append({"tehsil": t, "rep_id": r["rep_id"]})
            if rep_rows:
                rep_tehsil = (
                    pd.DataFrame(rep_rows)
                    .drop_duplicates("tehsil")
                )
                growers = growers.merge(rep_tehsil, on="tehsil", how="left")

        print(f"  Feature store: {len(growers):,} growers, {len(growers.columns)} columns")
        return growers

    # ── Step 3: Stock gate ───────────────────────────────────────────────────

    def apply_stock_gate(
        self, growers: pd.DataFrame, min_stock_rate: float = MIN_STOCK_RATE
    ) -> pd.DataFrame:
        """
        Step 3  — Stock check gate.
        Hard-blocks growers where ALL nearby retailers are OOS (stockout_risk=1).
        Soft-filters growers where fewer than min_stock_rate of retailers have stock.
        Sits BETWEEN the engines and the queue — no suppressed grower ever reaches
        content generation, so no wasted LLM calls.
        """
        print("\n─" * 60)
        print("Step 3 │ Stock check gate")

        growers["tehsil_stock_rate"] = growers.apply(
            lambda r: self._stock.get_stock_rate(r.get("tehsil", ""), r.get("crop", "wheat")),
            axis=1,
        )
        before = len(growers)

        # Hard block: all retailers OOS
        if "stockout_risk" in growers.columns:
            n_hard = int((growers["stockout_risk"] == 1).sum())
            growers = growers[growers["stockout_risk"] != 1]
            if n_hard:
                print(f"  Hard-blocked  : {n_hard} growers (stockout_risk = 1, all retailers OOS)")

        # Soft filter: below tehsil stock threshold
        growers = growers[growers["tehsil_stock_rate"] >= min_stock_rate]
        print(f"  Total removed : {before - len(growers)} growers below stock threshold ({min_stock_rate:.0%})")
        print(f"  Eligible      : {len(growers):,} growers pass stock gate")
        return growers

    # ── Step 4: Four AI engines ──────────────────────────────────────────────

    def run_engines(
        self,
        growers: pd.DataFrame,
        ext_cache: dict[str, ExternalContext],
        today: datetime,
        max_growers: int = 500,
        min_receptivity: float = MIN_RECEPTIVITY_SCORE,
    ) -> pd.DataFrame:
        """
        Step 4  — Four AI engines running in dependency order.
        E3 (Receptivity) and E4 (Micro-segment) run first because they filter
        and annotate the grower list.  E2 (Targeting) picks the arm per grower.
        E1 (Content) generates the final payload.

        The engines are logically sequential here because E2 depends on E3 output
        (receptivity tier affects eligible arms) and E1 depends on E2 (channel
        determines ContentFormat).  Within each engine, row-level work is
        parallelised via ThreadPoolExecutor.
        """
        print("\n─" * 60)
        print("Step 4 │ Four AI engines")

        # ── E3: Receptivity scoring ──────────────────────────────────────────
        print("  E3 │ Receptivity — XGBoost scores every grower …")

        def _season_stats(row):
            sow = pd.to_datetime(row.get("sowing_start"))
            har = pd.to_datetime(row.get("harvest_start"))
            if pd.notna(sow) and pd.notna(har):
                total = (har - sow).days or 180
                prog  = min(max((today - sow).days / total, 0), 1)
                dth   = max((har - today).days, 0)
            else:
                prog, dth = 0.5, 60
            return prog, dth

        rec_inputs = []
        for _, row in growers.iterrows():
            sp, dth = _season_stats(row)
            rec_inputs.append(ReceptivityInput(
                grower_id=str(row["grower_id"]),
                grower_age=int(row.get("grower_age", 45)),
                grower_farm_size=float(row.get("grower_farm_size", 2.0)),
                language=str(row.get("language", "Hindi")),
                device_type=str(row.get("device_type", "unknown")),
                crop=str(row.get("crop", "wheat")),
                state=str(row.get("state", "Uttar Pradesh")),
                gender=str(row.get("gender", "male")),
                offline_campaign_attended=bool(row.get("offline_campaign_attended", False)),
                product_scan=bool(row.get("product_scan", False)),
                hist_open_rate=float(row.get("wa_open_rate", 0)),
                hist_click_rate=float(row.get("wa_click_rate", 0)),
                hist_delivery_rate=float(row.get("wa_delivery_rate", 1)),
                hist_pos_rate=float(row.get("pos_converted_rate", row.get("hist_pos_rate", 0.0))),
                cum_messages=int(row.get("wa_messages", 0)),
                days_to_harvest=dth,
                season_progress=sp,
                tehsil_stock_rate=float(row.get("tehsil_stock_rate", 0.7)),
                days_since_rep_visit=30,
                send_date=today,
            ))

        rec_scores = self._receptivity.score_batch(rec_inputs)
        growers = growers.copy()
        growers["receptivity_score"] = [s.score for s in rec_scores]
        growers["receptivity_tier"]  = [s.tier  for s in rec_scores]

        # Crop-calendar pre-stage boost (3–7 days before next stage → ×1.35)
        if "days_to_next_stage" in growers.columns:
            mask = growers["days_to_next_stage"].between(*PRE_STAGE_WINDOW)
            growers.loc[mask, "receptivity_score"] = (
                growers.loc[mask, "receptivity_score"] * PRE_STAGE_BOOST_FACTOR
            ).clip(upper=1.0)
            n_boost = int(mask.sum())
            if n_boost:
                print(f"     Pre-stage boost (×{PRE_STAGE_BOOST_FACTOR}) applied to {n_boost} growers")

        # Field-rep hybrid: low-receptivity + rep assigned → route to field rep
        if "rep_id" in growers.columns:
            low_rep = (growers["receptivity_tier"] == "low") & growers["rep_id"].notna()
            growers.loc[low_rep, "receptivity_tier"] = "low_route_to_rep"
            growers.loc[low_rep, "receptivity_score"] = growers.loc[
                low_rep, "receptivity_score"
            ].clip(lower=min_receptivity)
            print(f"     {int(low_rep.sum())} growers flagged → field_rep_brief channel")

        # Filter below threshold and select top-K
        before  = len(growers)
        growers = growers[growers["receptivity_score"] >= min_receptivity].copy()
        growers = growers.sort_values("receptivity_score", ascending=False).head(max_growers)
        print(f"     {before - len(growers)} below threshold. Selected {len(growers):,} for pipeline.")

        # ── E4: Micro-segmentation ────────────────────────────────────────────
        print("  E4 │ Micro-segment — HDBSCAN assigns segment + template …")

        def _assign_segment(row) -> str:
            result = self._segmentation.assign(
                grower_id=str(row["grower_id"]),
                crop=str(row.get("crop", "wheat")),
                device_type=str(row.get("device_type", "unknown")),
                language=str(row.get("language", "Hindi")),
                state=str(row.get("state", "Uttar Pradesh")),
                grower_age=int(row.get("grower_age", 45)),
                farm_size=float(row.get("grower_farm_size", 2.0)),
                open_rate=float(row.get("wa_open_rate", 0)),
                click_rate=float(row.get("wa_click_rate", 0)),
                offline_attended=bool(row.get("offline_campaign_attended", False)),
                product_scan=bool(row.get("product_scan", False)),
            )
            return result.segment_id

        growers["segment_id"] = growers.apply(_assign_segment, axis=1)
        n_segs = growers["segment_id"].nunique()
        print(f"     {n_segs} distinct micro-segments across {len(growers):,} growers")

        # ── E2: Targeting — LinUCB / Thompson arm selection ───────────────────
        print("  E2 │ Targeting — LinUCB picks channel × time slot × creative …")

        def _bandit_decide(row):
            key        = f"{row.get('tehsil', '')}|{row.get('crop', 'wheat')}"
            ext        = ext_cache.get(key) or enrich_grower_context(
                row.get("tehsil", ""), row.get("state", ""), row.get("crop", "wheat")
            )
            stage      = get_growth_stage(row.get("stages", []), today)
            stage_days = days_to_next_stage(row.get("stages", []), today)
            device     = str(row.get("device_type", "unknown"))
            dl         = {"smartphone": 0.85, "keypad": 0.4, "unknown": 0.2}.get(device, 0.2)
            sp, _      = (
                (lambda sow, har: (
                    min(max((today - sow).days / ((har - sow).days or 180), 0), 1),
                    max((har - today).days, 0),
                ))(pd.to_datetime(row.get("sowing_start")), pd.to_datetime(row.get("harvest_start")))
                if pd.notna(row.get("sowing_start")) and pd.notna(row.get("harvest_start"))
                else (0.5, 60)
            )
            ctx = GrowerContext(
                grower_id=str(row["grower_id"]),
                device_type=device,
                language=str(row.get("language", "Hindi")),
                crop=str(row.get("crop", "wheat")),
                growth_stage=stage,
                season_progress=sp,
                days_to_next_stage=stage_days,
                hist_open_rate=float(row.get("wa_open_rate", 0)),
                hist_click_rate=float(row.get("wa_click_rate", 0)),
                tehsil_stock_rate=float(row.get("tehsil_stock_rate", 0.7)),
                days_since_rep_visit=30,
                digital_literacy_score=dl,
                grower_age=int(row.get("grower_age", 45)),
                farm_size_acres=float(row.get("grower_farm_size", 2.0)),
                offline_campaign_attended=bool(row.get("offline_campaign_attended", False)),
                product_scan_done=bool(row.get("product_scan", False)),
                weather_risk_score=ext.weather.weather_risk_score,
                pest_pressure_index=ext.pest.pest_pressure_index,
            )
            dec = decide(self._bandit, ctx, thompson_agent=self._thompson)
            return {
                "grower_id":           row["grower_id"],
                "bandit_channel":      dec.channel,
                "bandit_time_slot":    dec.time_slot,
                "bandit_creative":     dec.creative_variant,
                "bandit_ucb_score":    dec.ucb_score,
                "bandit_arm":          dec.selected_arm,
                "growth_stage":        stage,
                "days_to_next_stage":  stage_days,
                "weather_risk_score":  ext.weather.weather_risk_score,
                "pest_pressure_index": ext.pest.pest_pressure_index,
                "composite_urgency":   ext.composite_urgency_score,
                "weather_alert":       ext.weather.alert_message,
                "pest_advisory":       ext.pest.advisory,
                "is_emergency":        ext.is_emergency,
                "_ext_key":            key,
            }

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_bandit_decide, row): idx
                       for idx, (_, row) in enumerate(growers.iterrows())}
            dec_rows = [None] * len(growers)
            for fut in as_completed(futures):
                dec_rows[futures[fut]] = fut.result()

        dec_df  = pd.DataFrame(dec_rows)
        growers = growers.merge(dec_df, on="grower_id", how="left")

        cold_start = int((growers["wa_messages"] == 0).sum())
        print(f"     {cold_start} cold-start growers → Thompson Sampling; rest → LinUCB")

        # ── E1: Content generation ────────────────────────────────────────────
        print("  E1 │ Content gen — LLM renders text + Sarvam audio + infographic …")

        def _resolve_channel(row) -> str:
            if str(row.get("receptivity_tier", "")) == "low_route_to_rep":
                return "field_rep_brief"
            return str(row.get("bandit_channel", "whatsapp_text"))

        def _generate_one(row):
            channel = _resolve_channel(row)
            fmt     = CHANNEL_TO_FORMAT.get(channel, ContentFormat.WHATSAPP_TEXT)
            req     = ContentRequest(
                grower_id=str(row["grower_id"]),
                crop=str(row.get("crop", "wheat")),
                growth_stage=str(row.get("growth_stage", "vegetative")),
                language=str(row.get("language", "Hindi")),
                state=str(row.get("state", "")),
                tehsil=str(row.get("tehsil", "")),
                content_format=fmt,
                device_type=str(row.get("device_type", "unknown")),
                farm_size_acres=float(row.get("grower_farm_size", 2.0)),
                days_to_next_stage=int(row.get("days_to_next_stage", 14)),
                weather_alert=str(row.get("weather_alert", "")),
                pest_pressure_index=float(row.get("pest_pressure_index", 0)),
                weather_risk_score=float(row.get("weather_risk_score", 0)),
            )
            content = generate_content(req, self.api_key, self.sarvam_api_key, self.bhashini_api_key)
            return row["grower_id"], channel, content

        # Parallel content generation (I/O bound — LLM calls)
        content_map: dict = {}
        channel_map: dict = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = {pool.submit(_generate_one, row): row["grower_id"]
                    for _, row in growers.iterrows()}
            for fut in as_completed(futs):
                gid, ch, content = fut.result()
                content_map[gid] = content
                channel_map[gid] = ch

        growers["_resolved_channel"] = growers["grower_id"].map(channel_map)
        growers["_content"]          = growers["grower_id"].map(content_map)

        ch_counts = growers["_resolved_channel"].value_counts().to_dict()
        print(f"     Channel mix: {ch_counts}")
        return growers

    # ── Step 5: Write message queue ──────────────────────────────────────────

    def write_queue(
        self,
        growers: pd.DataFrame,
        today: datetime,
        dry_run: bool = False,
    ) -> Path:
        """
        Step 5  — Pre-rendered message queue write.
        Serialises one QueueRecord per grower to results/message_queue_<date>.jsonl.
        Each record includes send_at (pre-computed), channel (pre-resolved),
        rendered text, audio URL, and image URL.
        Zone 2 reads this file and flushes — zero model calls at send time.
        """
        print("\n─" * 60)
        print("Step 5 │ Writing pre-rendered message queue")

        slot_hours = {"morning_7_10": 8, "midday_12_14": 13, "evening_18_21": 19}

        records: list[QueueRecord] = []
        for _, row in growers.iterrows():
            content   = row["_content"]
            channel   = row["_resolved_channel"]
            is_emerg  = bool(row.get("is_emergency", False))

            # send_at: emergency → +2 h from now; normal → scheduled slot next day
            if is_emerg:
                send_at = today + timedelta(hours=EMERGENCY_CONTENT_WINDOW_HRS)
            else:
                hour    = slot_hours.get(str(row.get("bandit_time_slot", "morning_7_10")), 8)
                # Schedule for tomorrow's slot (batch runs at night)
                send_at = (today + timedelta(days=1)).replace(
                    hour=hour, minute=0, second=0, microsecond=0
                )

            records.append(QueueRecord(
                grower_id=str(row["grower_id"]),
                campaign_id=self.campaign_id,
                segment_id=str(row.get("segment_id", "UNKNOWN")),
                channel=channel,
                send_at=send_at.isoformat(),
                is_emergency=is_emerg,
                text=content.text,
                audio_url=content.tts_audio_url,
                image_url=getattr(content, "image_url", None),
                language=content.language,
                receptivity_score=float(row.get("receptivity_score", 0)),
                ucb_score=float(row.get("bandit_ucb_score", 0)),
                stock_rate=float(row.get("tehsil_stock_rate", 0)),
                weather_risk=float(row.get("weather_risk_score", 0)),
                pest_pressure=float(row.get("pest_pressure_index", 0)),
                composite_urgency=float(row.get("composite_urgency", 0)),
                bandit_arm=str(row.get("bandit_arm", "")),
                creative_variant=str(row.get("bandit_creative", "")),
                time_slot=str(row.get("bandit_time_slot", "morning_7_10")),
            ))

        queue_path = RESULTS_DIR / QUEUE_FILE_TEMPLATE.format(date=today.strftime("%Y%m%d"))

        if dry_run:
            print(f"  [DRY RUN] Would write {len(records):,} records to {queue_path}")
        else:
            with queue_path.open("w", encoding="utf-8") as fh:
                for rec in records:
                    fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
            print(f"  Queue written → {queue_path}  ({len(records):,} records)")

        # Summary
        channels  = {}
        languages = {}
        emergency = 0
        for r in records:
            channels[r.channel]   = channels.get(r.channel, 0) + 1
            languages[r.language] = languages.get(r.language, 0) + 1
            emergency             += int(r.is_emergency)

        print(f"\n{'─'*60}")
        print("Zone 1 — Nightly Batch complete")
        print(f"  Growers queued   : {len(records):,}")
        print(f"  Emergency sends  : {emergency}")
        print(f"  Channel mix      : {channels}")
        print(f"  Language mix     : {languages}")
        avg_rec = sum(r.receptivity_score for r in records) / max(len(records), 1)
        print(f"  Avg receptivity  : {avg_rec:.4f}")
        print(f"{'─'*60}\n")
        return queue_path

    # ── Public entry point ────────────────────────────────────────────────────

    def run(
        self,
        target_crop: Optional[str] = None,
        max_growers: int = 500,
        min_receptivity: float = MIN_RECEPTIVITY_SCORE,
        min_stock_rate: float = MIN_STOCK_RATE,
        dry_run: bool = False,
    ) -> Path:
        """
        Full Zone 1 pipeline:
          1. External signal fetch   → data/signal_cache.json
          2. Grower-360 feature store
          3. Stock gate
          4. Four AI engines (E3 → E4 → E2 → E1)
          5. Queue write             → results/message_queue_<date>.jsonl

        Returns the path to the written queue file (or None on dry-run).
        """
        t0    = time.time()
        today = datetime.utcnow()

        print(f"\n{'═'*60}")
        print(f"ZONE 1 — NIGHTLY BATCH  {today.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"Campaign : {self.campaign_id}")
        print(f"{'═'*60}\n")

        # 2. Feature store (must come before signal fetch so we know the tehsils)
        growers = self.build_feature_store()

        if target_crop:
            growers = growers[growers["crop"] == target_crop].copy()
            print(f"  Filtered to {len(growers):,} growers growing {target_crop}")

        # 1. External signal fetch (needs tehsils from feature store)
        ext_cache = self.fetch_signals(growers)

        # 3. Stock gate
        growers = self.apply_stock_gate(growers, min_stock_rate=min_stock_rate)

        # 4. Four AI engines
        growers = self.run_engines(
            growers, ext_cache, today,
            max_growers=max_growers,
            min_receptivity=min_receptivity,
        )

        # 5. Write queue
        queue_path = self.write_queue(growers, today, dry_run=dry_run)

        elapsed = time.time() - t0
        print(f"Zone 1 finished in {elapsed:.1f}s\n")
        return queue_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _cli():
    p = argparse.ArgumentParser(description="Zone 1 — Nightly Batch runner")
    p.add_argument("--campaign",   default="CMP_RABI25",  help="Campaign ID")
    p.add_argument("--crop",       default=None,           help="Filter to a single crop")
    p.add_argument("--max",        type=int, default=500,  help="Max growers to queue")
    p.add_argument("--dry-run",    action="store_true",    help="Skip queue write")
    # CLI flags can still override .env values when supplied explicitly
    p.add_argument("--api-key",       default=None, help="Override GEMINI_API_KEY from .env")
    p.add_argument("--sarvam-key",    default=None, help="Override SARVAM_API_KEY from .env")
    p.add_argument("--bhashini-key",  default=None, help="Override BHASHINI_API_KEY from .env")
    p.add_argument("--imd-key",       default=None, help="Override IMD_API_KEY from .env")
    p.add_argument("--ncipm-key",     default=None, help="Override NCIPM_API_KEY from .env")
    p.add_argument("--agmarknet-key", default=None, help="Override AGMARKNET_KEY from .env")
    args = p.parse_args()

    # Resolve: CLI flag > .env > empty string
    batch = NightlyBatch(
        campaign_id=args.campaign,
        api_key="AIzaSyCf2wk6tDzH8DSilXO5Hl2QkRH_C6GXAFU",
        sarvam_api_key="sk_3rgcx9gi_5h2NHUj1DuB2PYY6HbwlN5a4",
        bhashini_api_key=args.bhashini_key  or os.getenv("BHASHINI_API_KEY", ""),
        imd_key=args.imd_key         or os.getenv("IMD_API_KEY",      ""),
        ncipm_key=args.ncipm_key       or os.getenv("NCIPM_API_KEY",    ""),
        agmarknet_key=args.agmarknet_key  or os.getenv("AGMARKNET_KEY",    ""),
    )
    batch.run(
        target_crop=args.crop,
        max_growers=args.max,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    _cli()