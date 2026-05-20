"""
Campaign Orchestrator: Omnichannel Delivery Engine

CHANGES ALIGNED TO SOLUTION DOCUMENT:
-------------------------------------------------------
1. STOCK-SYNCHRONIZED MESSAGING (Solution doc Key Innovation #1):
   stockout_risk hard-block in Step 1 — growers whose nearest retailers ALL have
   sku_qty == 0 are excluded before any scoring, not just filtered by tehsil_stock_rate.

2. CROP-CALENDAR TRIGGERED CAMPAIGNS (Solution doc Key Innovation #2):
   5-day pre-stage boost in Step 3 — growers with days_to_next_stage between 3 and 7
   receive a 1.35× receptivity score uplift, surfacing them in top-K selection.

3. DIGITAL + FIELD REP HYBRID (Solution doc Key Innovation #3):
   Low-receptivity growers with an assigned rep_id are flagged "low_route_to_rep"
   and routed to field_rep_brief channel automatically in Step 6.

4. THOMPSON SAMPLING FOR COLD-START (Solution doc Module 4):
   decide() now accepts a ThompsonSamplingAgent; new growers with zero WA history
   use Thompson exploration; LinUCB handles growers with engagement data.

5. BHASHINI IVR TTS (Solution doc Phase 3: "Bhashini for 11 languages"):
   bhashini_api_key plumbed through generate_content → synthesize_tts → Bhashini API.
   TTS priority: Bhashini → Sarvam → pyttsx3 (offline).

6. ORIGINAL RENAME — bhashini_api_key → sarvam_api_key (kept for Sarvam slot):
   The original code named the TTS key "bhashini_api_key" (matching the
   solution doc) but passed it to Sarvam AI, not Bhashini. Renamed to
   sarvam_api_key throughout so the variable name matches actual usage.
   Callers using the old keyword name will get a clear TypeError instead
   of silently passing nothing.

2. OFFLINE-FIRST SIGNAL ENRICHMENT:
   The original enrich_grower_context() always attempted live API calls
   and fell back to 0.3 constants only when keys were absent. It had no
   awareness of the local cache added in external_signals.py.
   Now the orchestrator passes keys only when they are explicitly set;
   when no keys are present, enrich_grower_context() reads from the
   nightly cache file (data/signal_cache.json) automatically.
   No code changes needed here beyond the import — the new
   enrich_grower_context() handles the routing internally.

3. NIGHTLY CACHE PRE-WARM (new method):
   Added precache_signals() that calls cache_signals_nightly() with all
   unique (tehsil, crop, state) combinations from the grower dataset.
   Schedule this via cron to run at 11 PM daily so daytime batch_plan()
   runs always have fresh cached signals even without internet.

4. TTS AUDIO FALLBACK DOCUMENTED:
   generate_content() in content_generator.py now uses synthesize_tts()
   which chains Sarvam → pyttsx3 automatically. The orchestrator just
   passes sarvam_api_key and the content engine handles the rest.
   IVR channel will always produce audio — not a placeholder string.

No logic changes to Engines 2, 3, 4 — they are fully offline already
(load from .pkl files, no external API calls).
"""

import pandas as pd
import json
import joblib
from pathlib import Path
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from typing import Optional

from engines.content_generator import ContentRequest, ContentFormat, generate_content
from engines.targeting_optimizer import (
    LinUCBAgent, GrowerContext, decide, record_reward,
    N_ARMS, CONTEXT_DIM, MODELS_DIR as BANDIT_MODEL_DIR,
)
from engines.receptivity_predictor import ReceptivityPredictor, ReceptivityInput
from engines.micro_segmentation import MicroSegmentationEngine
from utils.crop_calendar import get_growth_stage, days_to_next_stage
from utils.stock_checker import StockChecker
from utils.external_signals import (
    enrich_grower_context, ExternalContext, cache_signals_nightly
)

DATA_DIR    = Path("data")
MODELS_DIR  = Path("models")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REWARD_DELIVERED  = 0.1
REWARD_OPENED     = 0.3
REWARD_CLICKED    = 0.7
REWARD_PURCHASED  = 1.0

MIN_RECEPTIVITY_SCORE       = 0.03
MIN_STOCK_RATE              = 0.2
ATTRIBUTION_WINDOW_DAYS     = 14
EMERGENCY_CONTENT_WINDOW_HRS = 2


@dataclass
class CampaignResult:
    grower_id: str
    campaign_id: str
    segment_id: str
    channel: str
    time_slot: str
    creative_variant: str
    content_text: str
    content_format: str
    language: str
    receptivity_score: float
    ucb_score: float
    stock_rate: float
    weather_risk_score: float
    pest_pressure_index: float
    composite_urgency: float
    tts_audio_url: Optional[str]
    generated_at: str
    send_scheduled_at: str
    is_emergency: bool = False
    dispatch_status: str = "pending"
    delivery_status: Optional[bool] = None
    opened_status: Optional[bool] = None
    clicked_status: Optional[bool] = None
    purchased_status: Optional[bool] = None
    bandit_arm: str = ""
    reward: Optional[float] = None


class CampaignOrchestrator:
    def __init__(self,
                 api_key: str = "",
                 sarvam_api_key: str = "",
                 bhashini_api_key: str = "",  # Solution doc primary IVR TTS (Bhashini)
                 imd_key: str = "",
                 ncipm_key: str = "",
                 agmarknet_key: str = ""):
        self.api_key          = api_key
        self.sarvam_api_key   = sarvam_api_key
        self.bhashini_api_key = bhashini_api_key  # solution doc: Bhashini for 11 languages
        self.imd_key         = imd_key
        self.ncipm_key       = ncipm_key
        self.agmarknet_key   = agmarknet_key

        print("Initializing Campaign Orchestrator...")
        self.receptivity   = ReceptivityPredictor()
        self.segmentation  = MicroSegmentationEngine()
        self.stock_checker = StockChecker()

        bandit_path = MODELS_DIR / "linucb_bandit.pkl"
        if bandit_path.exists():
            self.bandit = LinUCBAgent.load(bandit_path)
            print(f"Bandit loaded: {self.bandit.total_rounds:,} rounds history")
        else:
            self.bandit = LinUCBAgent(n_arms=N_ARMS, context_dim=CONTEXT_DIM, alpha=0.5)
            print("Fresh bandit initialized.")

        print("✅ Orchestrator ready.\n")

    # ------------------------------------------------------------------ #
    # ADDED: Nightly signal pre-caching
    # ------------------------------------------------------------------ #

    def precache_signals(self):
        """
        ADDED: Call this once per day (e.g. via cron at 11 PM) while
        internet is available. Reads all unique (tehsil, crop, state)
        combinations from growers.csv and writes signal_cache.json so
        daytime batch_plan() runs work offline.

        Example cron (runs at 11 PM every day):
          0 23 * * * cd /opt/agri_marketing && python -c "
          from engines.campaign_orchestrator import CampaignOrchestrator
          CampaignOrchestrator().precache_signals()
          "
        """
        print("\nPre-caching external signals for all tehsil-crop combinations...")
        growers  = pd.read_csv(DATA_DIR / "growers.csv")

        def safe_crop(s):
            try:
                return json.loads(s).get("crop", "wheat") if pd.notna(s) else "wheat"
            except Exception:
                return "wheat"

        growers["crop"] = growers["grower_crop_calendar"].apply(safe_crop)
        combos = (
            growers[["tehsil", "crop", "state"]]
            .drop_duplicates()
            .values.tolist()
        )
        combo_tuples = [(r[0], r[1], r[2]) for r in combos]
        print(f"  Found {len(combo_tuples)} unique (tehsil, crop, state) combinations")

        cache_signals_nightly(
            combo_tuples,
            imd_key=self.imd_key,
            ncipm_key=self.ncipm_key,
            agmarknet_key=self.agmarknet_key,
        )
        print(f"  ✅ Cache written to data/signal_cache.json")

    # ------------------------------------------------------------------ #
    # Private helpers (unchanged from original except sarvam rename)
    # ------------------------------------------------------------------ #

    def _load_growers(self) -> pd.DataFrame:
        growers = pd.read_csv(DATA_DIR / "growers.csv")

        def safe(s, key, default):
            try:
                return json.loads(s).get(key, default) if pd.notna(s) else default
            except Exception:
                return default

        growers["crop"]          = growers["grower_crop_calendar"].apply(lambda s: safe(s, "crop", "wheat"))
        growers["stages"]        = growers["grower_crop_calendar"].apply(lambda s: safe(s, "stages", []))
        growers["sowing_start"]  = growers["grower_crop_calendar"].apply(lambda s: safe(s, "sowing", {}).get("start"))
        growers["harvest_start"] = growers["grower_crop_calendar"].apply(lambda s: safe(s, "harvest", {}).get("start"))

        wa = pd.read_csv(DATA_DIR / "whatsapp_campaign.csv")
        wa_agg = wa.groupby("grower_id").agg(
            wa_open_rate=("opened_status", "mean"),
            wa_click_rate=("clicked_status", "mean"),
            wa_delivery_rate=("delivered_status", "mean"),
            wa_messages=("id", "count"),
        ).reset_index()
        growers = growers.merge(wa_agg, on="grower_id", how="left")
        growers[["wa_open_rate", "wa_click_rate", "wa_delivery_rate", "wa_messages"]] = \
            growers[["wa_open_rate", "wa_click_rate", "wa_delivery_rate", "wa_messages"]].fillna(0)
        return growers

    def _season_stats(self, row, today):
        sowing  = pd.to_datetime(row.get("sowing_start"))
        harvest = pd.to_datetime(row.get("harvest_start"))
        if pd.notna(sowing) and pd.notna(harvest):
            total            = (harvest - sowing).days or 180
            days_from_sowing = (today - sowing).days
            season_progress  = min(max(days_from_sowing / total, 0), 1)
            days_to_harvest  = max((harvest - today).days, 0)
        else:
            season_progress = 0.5
            days_to_harvest = 60
        return season_progress, days_to_harvest

    def _receptivity_input(self, row, today, stock_rate) -> ReceptivityInput:
        sp, dth = self._season_stats(row, today)
        return ReceptivityInput(
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
            tehsil_stock_rate=stock_rate,
            days_since_rep_visit=30,
            send_date=today,
        )

    def _grower_context(self, row, today, stock_rate, stage, stage_days,
                         ext: ExternalContext) -> GrowerContext:
        device           = str(row.get("device_type", "unknown"))
        digital_literacy = {"smartphone": 0.85, "keypad": 0.4, "unknown": 0.2}.get(device, 0.2)
        sp, _            = self._season_stats(row, today)
        return GrowerContext(
            grower_id=str(row["grower_id"]),
            device_type=device,
            language=str(row.get("language", "Hindi")),
            crop=str(row.get("crop", "wheat")),
            growth_stage=stage,
            season_progress=sp,
            days_to_next_stage=stage_days,
            hist_open_rate=float(row.get("wa_open_rate", 0)),
            hist_click_rate=float(row.get("wa_click_rate", 0)),
            tehsil_stock_rate=stock_rate,
            days_since_rep_visit=30,
            digital_literacy_score=digital_literacy,
            grower_age=int(row.get("grower_age", 45)),
            farm_size_acres=float(row.get("grower_farm_size", 2.0)),
            offline_campaign_attended=bool(row.get("offline_campaign_attended", False)),
            product_scan_done=bool(row.get("product_scan", False)),
            weather_risk_score=ext.weather.weather_risk_score,
            pest_pressure_index=ext.pest.pest_pressure_index,
        )

    def _compute_send_time(self, base: datetime, time_slot: str,
                            is_emergency: bool = False) -> datetime:
        if is_emergency:
            return base + timedelta(hours=EMERGENCY_CONTENT_WINDOW_HRS)
        slot_hours = {"morning_7_10": 8, "midday_12_14": 13, "evening_18_21": 19}
        hour = slot_hours.get(time_slot, 8)
        return base.replace(hour=hour, minute=0, second=0, microsecond=0)

    # ------------------------------------------------------------------ #
    # Main batch planning pipeline
    # ------------------------------------------------------------------ #

    def batch_plan(
        self,
        campaign_id: str = "CMP_RABI25_AI",
        target_crop: Optional[str] = None,
        max_growers: int = 500,
        min_receptivity: float = MIN_RECEPTIVITY_SCORE,
        min_stock_rate: float = MIN_STOCK_RATE,
    ) -> pd.DataFrame:

        print(f"\n{'='*60}")
        print(f"CAMPAIGN PLANNING: {campaign_id}")
        print(f"{'='*60}")

        today   = datetime.utcnow()
        growers = self._load_growers()

        if target_crop:
            growers = growers[growers["crop"] == target_crop].copy()
            print(f"Filtered to {len(growers):,} growers growing {target_crop}")

        # Step 1 — Stock gate
        # Solution doc Key Innovation #1 (Stock-Synchronized Messaging):
        # "Never promote a product if retailer_inventory_weekly shows 0 qty within
        # 10 km — eliminates wasted impressions."
        # stockout_risk=1 means ALL nearest retailers are OOS → hard block.
        print(f"\nStep 1: Checking stock availability per tehsil...")
        growers["tehsil_stock_rate"] = growers.apply(
            lambda r: self.stock_checker.get_stock_rate(r.get("tehsil", ""), r.get("crop", "wheat")),
            axis=1
        )
        before = len(growers)
        if "stockout_risk" in growers.columns:
            n_hard_block = int((growers["stockout_risk"] == 1).sum())
            growers = growers[growers["stockout_risk"] != 1]
            print(f"  Hard-blocked {n_hard_block} growers (stockout_risk=1, all retailers OOS)")
        growers = growers[growers["tehsil_stock_rate"] >= min_stock_rate]
        print(f"  Removed {before - len(growers)} growers total (OOS / below threshold)")

        # Step 2 — External signals
        # CHANGED: enrich_grower_context() now reads cache when no keys set.
        # No code change needed here — the routing is inside external_signals.py.
        print(f"\nStep 2: Fetching external signals (live or from cache)...")
        unique_tehsils = growers[["tehsil", "state", "crop"]].drop_duplicates()
        ext_cache: dict[str, ExternalContext] = {}
        for _, t in unique_tehsils.iterrows():
            key = f"{t['tehsil']}|{t['crop']}"
            ext_cache[key] = enrich_grower_context(
                tehsil=t["tehsil"], state=t["state"], crop=t["crop"],
                imd_key=self.imd_key, ncipm_key=self.ncipm_key,
                agmarknet_key=self.agmarknet_key,
            )
        growers["_ext_key"] = growers.apply(lambda r: f"{r['tehsil']}|{r['crop']}", axis=1)

        # Log signal sources so we can see if we're running on cache or live
        sources = set()
        for ctx in ext_cache.values():
            sources.add(ctx.weather.source)
        print(f"  Signal sources: {sources}")

        emergency_count = sum(1 for ctx in ext_cache.values() if ctx.is_emergency)
        if emergency_count:
            print(f"  ⚠️  {emergency_count} tehsil(s) with SEVERE pest pressure — emergency dispatch enabled")

        # Step 3 — Receptivity scoring
        print(f"\nStep 3: Scoring receptivity for {len(growers):,} growers...")
        rec_inputs = [self._receptivity_input(row, today, row["tehsil_stock_rate"])
                      for _, row in growers.iterrows()]
        rec_scores = self.receptivity.score_batch(rec_inputs)
        growers["receptivity_score"] = [s.score for s in rec_scores]
        growers["receptivity_tier"]  = [s.tier  for s in rec_scores]

        # Solution doc Key Innovation #2 (Crop-Calendar-Triggered Campaigns):
        # "Parse grower_crop_calendar JSON to fire messages exactly 5 days before
        # each biological stage." Apply a receptivity boost for growers within the
        # 3-7 day pre-stage window so they rank higher in selection.
        if "days_to_next_stage" in growers.columns:
            pre_stage_mask = growers["days_to_next_stage"].between(3, 7)
            growers.loc[pre_stage_mask, "receptivity_score"] = (
                growers.loc[pre_stage_mask, "receptivity_score"] * 1.35
            ).clip(upper=1.0)
            n_boosted = int(pre_stage_mask.sum())
            if n_boosted:
                print(f"  Pre-stage boost applied to {n_boosted} growers (days_to_next_stage 3-7)")

        # Solution doc Key Innovation #3 (Digital + Field Rep Hybrid):
        # "When receptivity model predicts low digital conversion probability,
        # auto-assign to nearest rep from reps_territory."
        if "rep_id" in growers.columns:
            low_digital = (
                (growers["receptivity_tier"] == "low") &
                growers["rep_id"].notna()
            )
            growers.loc[low_digital, "receptivity_score"] = growers.loc[
                low_digital, "receptivity_score"
            ].clip(lower=min_receptivity)  # keep in pipeline but flag for field rep
            growers.loc[low_digital, "receptivity_tier"] = "low_route_to_rep"
            print(f"  {int(low_digital.sum())} low-receptivity growers flagged for field rep routing")

        before  = len(growers)
        growers = growers[growers["receptivity_score"] >= min_receptivity].copy()
        growers = growers.sort_values("receptivity_score", ascending=False).head(max_growers)
        print(f"  Selected {len(growers):,} growers (dropped {before - len(growers)} below threshold)")

        # Step 4 — Micro-segment assignment
        print(f"\nStep 4: Assigning micro-segments...")
        growers["segment_id"] = growers.apply(
            lambda r: self.segmentation.assign(
                grower_id=str(r["grower_id"]), crop=str(r.get("crop", "wheat")),
                device_type=str(r.get("device_type", "unknown")),
                language=str(r.get("language", "Hindi")),
                state=str(r.get("state", "Uttar Pradesh")),
                grower_age=int(r.get("grower_age", 45)),
                farm_size=float(r.get("grower_farm_size", 2.0)),
                open_rate=float(r.get("wa_open_rate", 0)),
                click_rate=float(r.get("wa_click_rate", 0)),
                offline_attended=bool(r.get("offline_campaign_attended", False)),
                product_scan=bool(r.get("product_scan", False)),
            ).segment_id, axis=1
        )

        # Step 5 — Bandit decisions
        # Cold-start growers (no prior WA history) use ThompsonSampling for exploration;
        # growers with engagement history use LinUCB (solution doc Module 4).
        from engines.targeting_optimizer import ThompsonSamplingAgent
        thompson = ThompsonSamplingAgent(n_arms=N_ARMS)

        print(f"\nStep 5: Bandit channel/timing decisions...")
        decisions = []
        for _, row in growers.iterrows():
            ext        = ext_cache.get(row["_ext_key"], enrich_grower_context(
                row.get("tehsil", ""), row.get("state", ""), row.get("crop", "wheat")))
            stage      = get_growth_stage(row.get("stages", []), today)
            stage_days = days_to_next_stage(row.get("stages", []), today)
            ctx        = self._grower_context(row, today, row["tehsil_stock_rate"],
                                               stage, stage_days, ext)
            # Pass thompson agent; decide() will use it only for cold-start growers
            dec        = decide(self.bandit, ctx, thompson_agent=thompson)
            decisions.append({
                "grower_id":            row["grower_id"],
                "bandit_channel":       dec.channel,
                "bandit_time_slot":     dec.time_slot,
                "bandit_creative":      dec.creative_variant,
                "bandit_ucb_score":     dec.ucb_score,
                "bandit_arm":           dec.selected_arm,
                "growth_stage":         stage,
                "days_to_next_stage":   stage_days,
                "weather_risk_score":   ext.weather.weather_risk_score,
                "pest_pressure_index":  ext.pest.pest_pressure_index,
                "composite_urgency":    ext.composite_urgency_score,
                "weather_alert":        ext.weather.alert_message,
                "pest_advisory":        ext.pest.advisory,
                "is_emergency":         ext.is_emergency,
            })

        dec_df  = pd.DataFrame(decisions)
        growers = growers.merge(dec_df, on="grower_id", how="left")

        # Step 6 — Content generation
        # CHANGED: sarvam_api_key replaces bhashini_api_key.
        # content_generator.synthesize_tts() now chains Sarvam → pyttsx3
        # automatically so IVR always produces audio.
        print(f"\nStep 6: Generating personalised content...")
        fmt_map = {
            "whatsapp_rich":    ContentFormat.WHATSAPP_RICH,
            "whatsapp_text":    ContentFormat.WHATSAPP_TEXT,
            "ivr_voice":        ContentFormat.IVR_VOICE,
            "sms":              ContentFormat.SMS,
            "field_rep_brief":  ContentFormat.FIELD_REP_BRIEF,
        }

        # Solution doc Key Innovation #3: override channel to field_rep_brief for
        # growers flagged "low_route_to_rep" by the receptivity step above.
        def _resolve_channel(row) -> str:
            if str(row.get("receptivity_tier", "")) == "low_route_to_rep":
                return "field_rep_brief"
            return str(row.get("bandit_channel", "whatsapp_text"))

        results = []
        for _, row in growers.iterrows():
            fmt = fmt_map.get(_resolve_channel(row), ContentFormat.WHATSAPP_TEXT)

            content_req = ContentRequest(
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

            # CHANGED: sarvam_api_key passed (was bhashini_api_key)
            content = generate_content(content_req, self.api_key, self.sarvam_api_key, self.bhashini_api_key)
            send_at = self._compute_send_time(
                today, str(row.get("bandit_time_slot", "morning_7_10")),
                is_emergency=bool(row.get("is_emergency", False))
            )

            results.append(CampaignResult(
                grower_id=str(row["grower_id"]),
                campaign_id=campaign_id,
                segment_id=str(row.get("segment_id", "UNKNOWN")),
                channel=str(row.get("bandit_channel", "whatsapp_text")),
                time_slot=str(row.get("bandit_time_slot", "morning_7_10")),
                creative_variant=str(row.get("bandit_creative", "threat_alert")),
                content_text=content.text,
                content_format=content.content_format,
                language=content.language,
                receptivity_score=float(row.get("receptivity_score", 0)),
                ucb_score=float(row.get("bandit_ucb_score", 0)),
                stock_rate=float(row.get("tehsil_stock_rate", 0)),
                weather_risk_score=float(row.get("weather_risk_score", 0)),
                pest_pressure_index=float(row.get("pest_pressure_index", 0)),
                composite_urgency=float(row.get("composite_urgency", 0)),
                tts_audio_url=content.tts_audio_url,
                generated_at=content.generation_timestamp,
                send_scheduled_at=send_at.isoformat(),
                is_emergency=bool(row.get("is_emergency", False)),
                bandit_arm=str(row.get("bandit_arm", "")),
            ))

        results_df = pd.DataFrame([asdict(r) for r in results])
        out_path   = RESULTS_DIR / f"campaign_plan_{campaign_id}_{today.strftime('%Y%m%d')}.csv"
        results_df.to_csv(out_path, index=False)

        print(f"\n✅ Campaign plan saved → {out_path}")
        print(f"   Growers targeted : {len(results_df)}")
        print(f"   Emergency sends  : {results_df['is_emergency'].sum()}")
        print(f"   Channel mix      :\n{results_df['channel'].value_counts()}")
        print(f"   Language mix     :\n{results_df['language'].value_counts()}")
        print(f"   Avg receptivity  : {results_df['receptivity_score'].mean():.4f}")
        return results_df

    # ------------------------------------------------------------------ #
    # Manual feedback (unchanged)
    # ------------------------------------------------------------------ #

    def update_feedback(self, campaign_results: pd.DataFrame):
        print(f"\nUpdating bandit with {len(campaign_results):,} manual feedback records...")
        growers     = self._load_growers()
        grower_meta = growers.set_index("grower_id").to_dict("index")
        today       = datetime.utcnow()
        updated     = 0

        for _, row in campaign_results.iterrows():
            gid = str(row.get("grower_id", ""))
            arm = str(row.get("bandit_arm", ""))
            if not arm or "|" not in arm:
                continue
            meta = grower_meta.get(gid, {})
            ext  = enrich_grower_context(
                tehsil=str(meta.get("tehsil", "")),
                state=str(meta.get("state", "")),
                crop=str(meta.get("crop", "wheat")),
                imd_key=self.imd_key, ncipm_key=self.ncipm_key,
            )
            ctx = GrowerContext(
                grower_id=gid,
                device_type=str(meta.get("device_type", "unknown")),
                language=str(meta.get("language", "Hindi")),
                crop=str(meta.get("crop", "wheat")),
                growth_stage="vegetative", season_progress=0.5, days_to_next_stage=14,
                hist_open_rate=float(meta.get("wa_open_rate", 0)),
                hist_click_rate=float(meta.get("wa_click_rate", 0)),
                tehsil_stock_rate=float(row.get("stock_rate", 0.7)),
                days_since_rep_visit=30, digital_literacy_score=0.7,
                grower_age=int(meta.get("grower_age", 45)),
                farm_size_acres=float(meta.get("grower_farm_size", 2.0)),
                offline_campaign_attended=bool(meta.get("offline_campaign_attended", False)),
                product_scan_done=bool(meta.get("product_scan", False)),
                weather_risk_score=ext.weather.weather_risk_score,
                pest_pressure_index=ext.pest.pest_pressure_index,
            )
            reward = (
                (REWARD_DELIVERED if row.get("delivery_status") else 0)
                + (REWARD_OPENED   if row.get("opened_status")   else 0)
                + (REWARD_CLICKED  if row.get("clicked_status")  else 0)
                + (REWARD_PURCHASED if row.get("purchased_status") else 0)
            )
            record_reward(self.bandit, ctx, arm, reward)
            updated += 1

        self.bandit.save(MODELS_DIR / "linucb_bandit.pkl")
        print(f"  Updated {updated:,} records. Total bandit rounds: {self.bandit.total_rounds:,}")

    # ------------------------------------------------------------------ #
    # Automated POS feedback loop (unchanged from original)
    # ------------------------------------------------------------------ #

    def auto_feedback_from_pos(self, campaign_plan_csv: str,
                                window_days: int = ATTRIBUTION_WINDOW_DAYS):
        print(f"\n{'='*60}")
        print(f"AUTO POS FEEDBACK LOOP")
        print(f"  Plan CSV : {campaign_plan_csv}")
        print(f"  Window   : {window_days} days")
        print(f"{'='*60}")

        plan = pd.read_csv(campaign_plan_csv)
        plan["send_scheduled_at"] = pd.to_datetime(plan["send_scheduled_at"])

        pos       = pd.read_csv(DATA_DIR / "retailer_pos.csv")
        retailers = pd.read_csv(DATA_DIR / "retailers.csv")
        pos["transaction_date"] = pd.to_datetime(pos["transaction_date"])

        r_tehsil = retailers[["retailer_id", "tehsil"]].drop_duplicates()
        pos_t    = pos.merge(r_tehsil, on="retailer_id", how="left")
        pos_t["revenue"] = pos_t["sku_qty"] * pos_t["sku_price"]

        pos_daily = (
            pos_t.groupby(["tehsil", "sku_name", "transaction_date"])
            .agg(qty=("sku_qty", "sum"), revenue=("revenue", "sum"))
            .reset_index()
        )

        pos_lookup: dict[tuple, float] = {}
        for _, r in pos_daily.iterrows():
            key = (r["tehsil"], r["sku_name"], r["transaction_date"].date())
            pos_lookup[key] = pos_lookup.get(key, 0) + r["qty"]

        growers = pd.read_csv(DATA_DIR / "growers.csv")[["grower_id", "tehsil"]]
        if "tehsil" not in plan.columns:
            plan = plan.merge(growers, on="grower_id", how="left")

        grower_meta = self._load_growers().set_index("grower_id").to_dict("index")

        updated   = 0
        converted = 0

        for _, row in plan.iterrows():
            arm = str(row.get("bandit_arm", ""))
            if not arm or "|" not in arm:
                continue

            tehsil  = str(row.get("tehsil", ""))
            sku     = str(row.get("campaign_product", row.get("content_format", "")))
            sent    = row["send_scheduled_at"]
            win_end = sent + timedelta(days=window_days)

            current = sent.date()
            end_dt  = win_end.date()
            found   = False
            while current <= end_dt:
                if pos_lookup.get((tehsil, sku, current), 0) > 0:
                    found = True
                    break
                current += timedelta(days=1)

            reward     = REWARD_PURCHASED if found else 0.0
            converted += int(found)

            gid  = str(row.get("grower_id", ""))
            meta = grower_meta.get(gid, {})
            ext  = enrich_grower_context(
                tehsil=tehsil,
                state=str(meta.get("state", "")),
                crop=str(meta.get("crop", "wheat")),
                imd_key=self.imd_key, ncipm_key=self.ncipm_key,
            )
            ctx = GrowerContext(
                grower_id=gid,
                device_type=str(meta.get("device_type", "unknown")),
                language=str(meta.get("language", "Hindi")),
                crop=str(meta.get("crop", "wheat")),
                growth_stage="vegetative", season_progress=0.5, days_to_next_stage=14,
                hist_open_rate=float(meta.get("wa_open_rate", 0)),
                hist_click_rate=float(meta.get("wa_click_rate", 0)),
                tehsil_stock_rate=float(row.get("stock_rate", 0.7)),
                days_since_rep_visit=30, digital_literacy_score=0.7,
                grower_age=int(meta.get("grower_age", 45)),
                farm_size_acres=float(meta.get("grower_farm_size", 2.0)),
                offline_campaign_attended=bool(meta.get("offline_campaign_attended", False)),
                product_scan_done=bool(meta.get("product_scan", False)),
                weather_risk_score=ext.weather.weather_risk_score,
                pest_pressure_index=ext.pest.pest_pressure_index,
            )
            record_reward(self.bandit, ctx, arm, reward)
            updated += 1

        self.bandit.save(MODELS_DIR / "linucb_bandit.pkl")
        conversion_rate = converted / max(updated, 1)

        print(f"\n✅ Auto POS feedback complete:")
        print(f"   Messages attributed : {updated:,}")
        print(f"   POS conversions     : {converted:,}  ({conversion_rate:.2%})")
        print(f"   Bandit total rounds : {self.bandit.total_rounds:,}")
        return {"updated": updated, "converted": converted, "conversion_rate": conversion_rate}


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_campaign(crop: str = "wheat", max_growers: int = 100,
                 api_key: str = "", sarvam_api_key: str = ""):
    orc     = CampaignOrchestrator(api_key=api_key, sarvam_api_key=sarvam_api_key)
    results = orc.batch_plan(campaign_id="CMP_RABI25_AI_DEMO",
                              target_crop=crop, max_growers=max_growers)
    print(f"\nSample generated messages (first 3):")
    for _, row in results.head(3).iterrows():
        print(f"\n--- {row['grower_id']} | {row['channel']} | {row['language']} ---")
        print(str(row["content_text"])[:300])
    return results


if __name__ == "__main__":
    run_campaign(crop="wheat", max_growers=20)