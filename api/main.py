"""
api/main.py — FastAPI Application
All four AI engines + orchestrator + automated POS feedback loop.

Run:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.schemas import (
    HealthResponse,
    ContentGenerateRequest, ContentGenerateResponse,
    TargetingDecisionRequest, TargetingDecisionResponse,
    BanditFeedbackRequest, BanditFeedbackResponse, BanditStatsResponse,
    ReceptivityRequest, ReceptivityResponse,
    BatchReceptivityRequest, BatchReceptivityResponse,
    SegmentRequest, SegmentResponse,
    CampaignPlanRequest, CampaignPlanSummary,
    FeedbackUpdateRequest, FeedbackUpdateResponse,
    AutoPOSFeedbackRequest, AutoPOSFeedbackResponse,
    ExternalSignalsRequest, ExternalSignalsResponse,
    StockCheckRequest, StockCheckResponse,
    AttributionRequest, AttributionResponse,
    CampaignAnalyticsResponse, InventoryAlertResponse,
    GrowerProfileResponse, FunnelMetrics, ErrorResponse,
)
from engines.content_generator import ContentRequest, ContentFormat, generate_content
from engines.targeting_optimizer import (
    LinUCBAgent, GrowerContext, decide, record_reward, N_ARMS, CONTEXT_DIM,
)
from engines.receptivity_predictor import ReceptivityPredictor, ReceptivityInput
from engines.micro_segmentation import MicroSegmentationEngine
from utils.stock_checker import StockChecker
from utils.attribution import AttributionEngine
from utils.crop_calendar import get_growth_stage, days_to_next_stage
from utils.external_signals import enrich_grower_context

DATA_DIR    = Path("data")
MODELS_DIR  = Path("models")
RESULTS_DIR = Path("results")

# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(
    title="Syngenta Agri-Marketing AI API",
    description=(
        "Four AI engines: Gemini GenAI Content · LinUCB Bandit · "
        "XGBoost Receptivity · HDBSCAN Micro-Segmentation.\n\n"
        "11 Indian languages · Bhashini TTS · Live IMD + ICAR signals · "
        "Automated POS feedback loop."
    ),
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ─────────────────────────────────────────────
# Lazy-loaded singletons
# ─────────────────────────────────────────────
_receptivity: Optional[ReceptivityPredictor]   = None
_segmentation: Optional[MicroSegmentationEngine] = None
_bandit: Optional[LinUCBAgent]                 = None
_stock_checker: Optional[StockChecker]         = None
_attribution: Optional[AttributionEngine]      = None
_growers_df: Optional[pd.DataFrame]            = None


def _get_receptivity():
    global _receptivity
    if _receptivity is None:
        _receptivity = ReceptivityPredictor()
    return _receptivity


def _get_segmentation():
    global _segmentation
    if _segmentation is None:
        _segmentation = MicroSegmentationEngine()
    return _segmentation


def _get_bandit():
    global _bandit
    if _bandit is None:
        p = MODELS_DIR / "linucb_bandit.pkl"
        _bandit = LinUCBAgent.load(p) if p.exists() else \
                  LinUCBAgent(n_arms=N_ARMS, context_dim=CONTEXT_DIM, alpha=0.5)
    return _bandit


def _get_stock():
    global _stock_checker
    if _stock_checker is None:
        _stock_checker = StockChecker()
    return _stock_checker


def _get_attribution():
    global _attribution
    if _attribution is None:
        _attribution = AttributionEngine()
    return _attribution


def _get_growers():
    global _growers_df
    if _growers_df is None and (DATA_DIR / "growers.csv").exists():
        _growers_df = pd.read_csv(DATA_DIR / "growers.csv")
        _growers_df["crop"] = _growers_df["grower_crop_calendar"].apply(
            lambda s: json.loads(s).get("crop", "wheat") if pd.notna(s) else "wheat"
        )
    return _growers_df if _growers_df is not None else pd.DataFrame()


def _ts():
    return datetime.utcnow().isoformat()


def _grower_ctx(req: TargetingDecisionRequest) -> GrowerContext:
    return GrowerContext(
        grower_id=req.grower_id, device_type=req.device_type,
        language=req.language, crop=req.crop, growth_stage=req.growth_stage,
        season_progress=req.season_progress, days_to_next_stage=req.days_to_next_stage,
        hist_open_rate=req.hist_open_rate, hist_click_rate=req.hist_click_rate,
        tehsil_stock_rate=req.tehsil_stock_rate,
        days_since_rep_visit=req.days_since_rep_visit,
        digital_literacy_score=req.digital_literacy_score,
        grower_age=req.grower_age, farm_size_acres=req.farm_size_acres,
        offline_campaign_attended=req.offline_campaign_attended,
        product_scan_done=req.product_scan_done,
        weather_risk_score=req.weather_risk_score,
        pest_pressure_index=req.pest_pressure_index,
    )


# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"message": "Syngenta Agri-Marketing AI API v2.0", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return HealthResponse(
        status="ok", version="2.0.0", timestamp=_ts(),
        models_loaded={
            "receptivity_model":  (MODELS_DIR / "receptivity_model.pkl").exists(),
            "linucb_bandit":      (MODELS_DIR / "linucb_bandit.pkl").exists(),
            "hdbscan_clusterer":  (MODELS_DIR / "hdbscan_clusterer.pkl").exists(),
            "kmeans_clusterer":   (MODELS_DIR / "kmeans_clusterer.pkl").exists(),
            "embedding_pipeline": (MODELS_DIR / "embedding_pipeline.pkl").exists(),
            "segment_profiles":   (RESULTS_DIR / "segment_profiles.parquet").exists(),
            "grower_data":        (DATA_DIR / "growers.csv").exists(),
        },
    )


# ─────────────────────────────────────────────
# Engine 1 — Content Generation
# ─────────────────────────────────────────────
@app.post("/api/v1/content/generate", response_model=ContentGenerateResponse,
          tags=["Engine 1 – Content"],
          summary="Generate personalised content (Gemini + Bhashini TTS)")
def content_generate(req: ContentGenerateRequest,
                      api_key: str = Query("", description="Gemini API key override"),
                      sarvam_api_key: str = Query("", description="Sarvam TTS key")):
    try:
        cr = ContentRequest(
            grower_id=req.grower_id, crop=req.crop, growth_stage=req.growth_stage,
            language=req.language, state=req.state, tehsil=req.tehsil,
            content_format=ContentFormat(req.content_format),
            device_type=req.device_type, grower_name=req.grower_name,
            farm_size_acres=req.farm_size_acres, nearest_retailer=req.nearest_retailer,
            active_threats=req.active_threats, days_to_next_stage=req.days_to_next_stage,
            weather_alert=req.weather_alert, rep_name=req.rep_name,
            pest_pressure_index=req.pest_pressure_index,
            weather_risk_score=req.weather_risk_score,
        )
        result = generate_content(cr, api_key=api_key, sarvam_api_key=sarvam_api_key)
        return ContentGenerateResponse(**result.__dict__)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/content/batch", tags=["Engine 1 – Content"],
          summary="Batch content generation (up to 100 growers)")
def content_batch(requests: list[ContentGenerateRequest],
                   api_key: str = Query(""),
                   sarvam_api_key: str = Query("")):
    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Max 100 per batch.")
    results = []
    for req in requests:
        try:
            cr = ContentRequest(
                grower_id=req.grower_id, crop=req.crop, growth_stage=req.growth_stage,
                language=req.language, state=req.state, tehsil=req.tehsil,
                content_format=ContentFormat(req.content_format),
                device_type=req.device_type, grower_name=req.grower_name,
                farm_size_acres=req.farm_size_acres, active_threats=req.active_threats,
                days_to_next_stage=req.days_to_next_stage,
                pest_pressure_index=req.pest_pressure_index,
                weather_risk_score=req.weather_risk_score,
            )
            r = generate_content(cr, api_key=api_key, sarvam_api_key=sarvam_api_key)
            results.append({"status": "ok", **r.__dict__})
        except Exception as e:
            results.append({"status": "error", "grower_id": req.grower_id, "error": str(e)})
    return {"total": len(results), "results": results}


# ─────────────────────────────────────────────
# Engine 2 — Targeting (LinUCB Bandit)
# ─────────────────────────────────────────────
@app.post("/api/v1/targeting/decide", response_model=TargetingDecisionResponse,
          tags=["Engine 2 – Targeting"],
          summary="LinUCB bandit channel + timing decision")
def targeting_decide(req: TargetingDecisionRequest):
    try:
        dec = decide(_get_bandit(), _grower_ctx(req))
        return TargetingDecisionResponse(
            grower_id=dec.grower_id, selected_arm=dec.selected_arm,
            channel=dec.channel, time_slot=dec.time_slot,
            creative_variant=dec.creative_variant, ucb_score=dec.ucb_score,
            eligible_arms_count=dec.eligible_arms_count,
            decision_timestamp=dec.decision_timestamp,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/targeting/feedback", response_model=BanditFeedbackResponse,
          tags=["Engine 2 – Targeting"],
          summary="Feed delivery/click/purchase reward back to bandit")
def targeting_feedback(req: BanditFeedbackRequest, background_tasks: BackgroundTasks):
    try:
        bandit = _get_bandit()
        ctx    = _grower_ctx(TargetingDecisionRequest(
            grower_id=req.grower_id, device_type=req.device_type,
            language=req.language, crop=req.crop, growth_stage=req.growth_stage,
            season_progress=req.season_progress, days_to_next_stage=req.days_to_next_stage,
            hist_open_rate=req.hist_open_rate, hist_click_rate=req.hist_click_rate,
            tehsil_stock_rate=req.tehsil_stock_rate,
            days_since_rep_visit=req.days_since_rep_visit,
            digital_literacy_score=req.digital_literacy_score,
            grower_age=req.grower_age, farm_size_acres=req.farm_size_acres,
            offline_campaign_attended=req.offline_campaign_attended,
            product_scan_done=req.product_scan_done,
            weather_risk_score=req.weather_risk_score,
            pest_pressure_index=req.pest_pressure_index,
        ))
        reward = (
            (0.1  if req.delivered  else 0)
            + (0.3 if req.opened    else 0)
            + (0.7 if req.clicked   else 0)
            + (1.0 if req.purchased else 0)
        )
        record_reward(bandit, ctx, req.arm, reward)
        background_tasks.add_task(bandit.save, MODELS_DIR / "linucb_bandit.pkl")
        return BanditFeedbackResponse(
            grower_id=req.grower_id, arm=req.arm,
            reward=round(reward, 4), total_rounds=bandit.total_rounds,
            message=f"Bandit updated. Total rounds: {bandit.total_rounds:,}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/targeting/stats", response_model=BanditStatsResponse,
         tags=["Engine 2 – Targeting"],
         summary="Arm empirical reward rates")
def targeting_stats(top_n: int = Query(10, ge=1, le=60)):
    b = _get_bandit()
    return BanditStatsResponse(total_rounds=b.total_rounds, n_arms=b.n_arms,
                                top_arms=b.arm_stats()[:top_n])


# ─────────────────────────────────────────────
# Engine 3 — Receptivity
# ─────────────────────────────────────────────
def _to_rec_input(g: ReceptivityRequest) -> ReceptivityInput:
    return ReceptivityInput(
        grower_id=g.grower_id, grower_age=g.grower_age,
        grower_farm_size=g.grower_farm_size, language=g.language,
        device_type=g.device_type, crop=g.crop, state=g.state, gender=g.gender,
        offline_campaign_attended=g.offline_campaign_attended, product_scan=g.product_scan,
        hist_open_rate=g.hist_open_rate, hist_click_rate=g.hist_click_rate,
        hist_delivery_rate=g.hist_delivery_rate,
        hist_pos_rate=g.hist_pos_rate,          # feature [5] — POS conversion rate
        cum_messages=g.cum_messages,
        days_to_harvest=g.days_to_harvest, season_progress=g.season_progress,
        tehsil_stock_rate=g.tehsil_stock_rate, days_since_rep_visit=g.days_since_rep_visit,
    )


@app.post("/api/v1/receptivity/score", response_model=ReceptivityResponse,
          tags=["Engine 3 – Receptivity"],
          summary="Predict receptivity score (0-1) for a single grower")
def receptivity_score(req: ReceptivityRequest):
    try:
        s = _get_receptivity().score(_to_rec_input(req))
        return ReceptivityResponse(**s.__dict__)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/receptivity/batch", response_model=BatchReceptivityResponse,
          tags=["Engine 3 – Receptivity"],
          summary="Batch score up to 1,000 growers")
def receptivity_batch(req: BatchReceptivityRequest):
    try:
        scores  = _get_receptivity().score_batch([_to_rec_input(g) for g in req.growers])
        results = [ReceptivityResponse(**s.__dict__) for s in scores]
        return BatchReceptivityResponse(
            results=results, total=len(results),
            high_tier_count=sum(1 for r in results if r.tier == "high"),
            medium_tier_count=sum(1 for r in results if r.tier == "medium"),
            low_tier_count=sum(1 for r in results if r.tier == "low"),
            avg_score=round(sum(r.score for r in results) / len(results), 4),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# Engine 4 — Micro-Segmentation
# ─────────────────────────────────────────────
@app.post("/api/v1/segment/assign", response_model=SegmentResponse,
          tags=["Engine 4 – Segmentation"],
          summary="Assign grower to micro-segment")
def segment_assign(req: SegmentRequest):
    try:
        r = _get_segmentation().assign(
            grower_id=req.grower_id, crop=req.crop, device_type=req.device_type,
            language=req.language, state=req.state, grower_age=req.grower_age,
            farm_size=req.farm_size, open_rate=req.open_rate,
            click_rate=req.click_rate, offline_attended=req.offline_attended,
            product_scan=req.product_scan,
        )
        return SegmentResponse(**r.__dict__)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/segment/profiles", tags=["Engine 4 – Segmentation"],
         summary="List micro-segment profiles")
def segment_profiles(limit: int = Query(20, ge=1, le=500)):
    p = RESULTS_DIR / "segment_profiles.parquet"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Run script 04 first.")
    df   = pd.read_parquet(p)
    cols = ["segment_id", "size", "dominant_crop", "dominant_language",
            "dominant_device", "dominant_state", "avg_click_rate", "recommended_channel"]
    avail = [c for c in cols if c in df.columns]
    return {"total_segments": len(df),
            "segments": df[avail].head(limit).to_dict("records")}


# ─────────────────────────────────────────────
# Campaign Orchestrator
# ─────────────────────────────────────────────
@app.post("/api/v1/campaign/plan", response_model=CampaignPlanSummary,
          tags=["Orchestrator"],
          summary="Full 4-engine campaign plan with live external signals")
def campaign_plan(req: CampaignPlanRequest, background_tasks: BackgroundTasks):
    try:
        from engines.campaign_orchestrator import CampaignOrchestrator
        orc = CampaignOrchestrator(
            api_key=req.api_key, sarvam_api_key=req.sarvam_api_key,
            imd_key=req.imd_key, ncipm_key=req.ncipm_key,
            agmarknet_key=req.agmarknet_key,
        )
        df = orc.batch_plan(
            campaign_id=req.campaign_id, target_crop=req.target_crop,
            max_growers=req.max_growers, min_receptivity=req.min_receptivity,
            min_stock_rate=req.min_stock_rate,
        )
        today    = datetime.utcnow()
        out_file = str(RESULTS_DIR / f"campaign_plan_{req.campaign_id}_{today.strftime('%Y%m%d')}.csv")

        growers_df = _get_growers()
        crop_mix   = {}
        if not growers_df.empty and "crop" in growers_df.columns:
            filtered = growers_df[growers_df["grower_id"].isin(df["grower_id"])]
            crop_mix = filtered["crop"].value_counts().to_dict()

        return CampaignPlanSummary(
            campaign_id=req.campaign_id,
            total_growers_targeted=len(df),
            emergency_sends=int(df.get("is_emergency", pd.Series([False])).sum()),
            channel_mix=df["channel"].value_counts().to_dict() if "channel" in df.columns else {},
            language_mix=df["language"].value_counts().to_dict() if "language" in df.columns else {},
            crop_mix=crop_mix,
            avg_receptivity_score=round(df["receptivity_score"].mean(), 4) if "receptivity_score" in df.columns else 0.0,
            avg_stock_rate=round(df["stock_rate"].mean(), 4) if "stock_rate" in df.columns else 0.0,
            avg_weather_risk=round(df["weather_risk_score"].mean(), 4) if "weather_risk_score" in df.columns else 0.0,
            avg_pest_pressure=round(df["pest_pressure_index"].mean(), 4) if "pest_pressure_index" in df.columns else 0.0,
            oos_growers_excluded=0, low_receptivity_excluded=0,
            output_file=out_file, planned_at=today.isoformat(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


@app.post("/api/v1/campaign/feedback", response_model=FeedbackUpdateResponse,
          tags=["Orchestrator"],
          summary="Manual feedback: update bandit from delivery outcomes")
def campaign_feedback(req: FeedbackUpdateRequest, background_tasks: BackgroundTasks):
    try:
        from engines.campaign_orchestrator import CampaignOrchestrator
        orc = CampaignOrchestrator()
        orc.update_feedback(pd.DataFrame(req.results))
        background_tasks.add_task(orc.bandit.save, MODELS_DIR / "linucb_bandit.pkl")
        return FeedbackUpdateResponse(
            campaign_id=req.campaign_id, records_processed=len(req.results),
            bandit_total_rounds=orc.bandit.total_rounds,
            message="Bandit updated from manual delivery outcomes.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/campaign/auto-feedback-pos", response_model=AutoPOSFeedbackResponse,
          tags=["Orchestrator"],
          summary="Automated POS feedback loop — reads retailer_pos.csv, closes bandit reward loop")
def campaign_auto_pos_feedback(req: AutoPOSFeedbackRequest, background_tasks: BackgroundTasks):
    """
    Automatically attributes 14-day POS conversions from retailer_pos.csv
    to campaign sends in the specified plan CSV, then updates the LinUCB
    bandit with purchase rewards — no manual data entry required.

    Schedule this endpoint via cron after each campaign window closes.
    """
    csv_path = RESULTS_DIR / req.campaign_plan_csv
    if not csv_path.exists():
        raise HTTPException(status_code=404,
                            detail=f"{req.campaign_plan_csv} not found in results/")
    try:
        from engines.campaign_orchestrator import CampaignOrchestrator
        orc    = CampaignOrchestrator()
        result = orc.auto_feedback_from_pos(str(csv_path), req.window_days)
        background_tasks.add_task(orc.bandit.save, MODELS_DIR / "linucb_bandit.pkl")
        return AutoPOSFeedbackResponse(
            updated=result["updated"], converted=result["converted"],
            conversion_rate=round(result["conversion_rate"], 4),
            bandit_total_rounds=orc.bandit.total_rounds,
            message=f"Auto POS feedback complete. Conversion rate: {result['conversion_rate']:.2%}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/campaign/download/{filename}", tags=["Orchestrator"],
         summary="Download a campaign plan CSV")
def download_campaign(filename: str):
    path = RESULTS_DIR / filename
    if not path.exists() or not filename.endswith(".csv"):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(str(path), media_type="text/csv", filename=filename)


# ─────────────────────────────────────────────
# External Signals
# ─────────────────────────────────────────────
@app.post("/api/v1/signals/external", response_model=ExternalSignalsResponse,
          tags=["External Signals"],
          summary="Fetch live IMD weather + ICAR pest + Agmarknet price for a tehsil")
def external_signals(req: ExternalSignalsRequest):
    """
    Returns composite context signals for a tehsil × crop combination.
    Used by the orchestrator automatically; also available for direct querying.
    Falls back to safe defaults when API keys are not provided.
    """
    try:
        ctx = enrich_grower_context(
            tehsil=req.tehsil, state=req.state, crop=req.crop,
            imd_key=req.imd_key, ncipm_key=req.ncipm_key,
            agmarknet_key=req.agmarknet_key,
        )
        return ExternalSignalsResponse(
            tehsil=req.tehsil, crop=req.crop,
            weather_risk_score=ctx.weather.weather_risk_score,
            weather_alert=ctx.weather.alert_message,
            pest_pressure_index=ctx.pest.pest_pressure_index,
            pest_severity=ctx.pest.severity,
            pest_name=ctx.pest.pest_name,
            pest_advisory=ctx.pest.advisory,
            is_emergency=ctx.is_emergency,
            mandi_price_per_quintal=ctx.mandi.modal_price_per_quintal,
            price_trend=ctx.mandi.price_trend,
            composite_urgency_score=ctx.composite_urgency_score,
            sources={
                "weather": ctx.weather.source,
                "pest":    ctx.pest.source,
                "mandi":   ctx.mandi.source,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────
@app.post("/api/v1/stock/check", response_model=StockCheckResponse,
          tags=["Utilities"], summary="Check product stock availability for a tehsil")
def stock_check(req: StockCheckRequest):
    from utils.stock_checker import CROP_TO_PRODUCT, LOW_STOCK_THRESHOLD
    checker = _get_stock()
    product = CROP_TO_PRODUCT.get(req.crop, req.crop)
    rate    = checker.get_stock_rate(req.tehsil, req.crop)
    best    = checker.get_best_stocked_product(req.tehsil, req.crop)
    return StockCheckResponse(
        tehsil=req.tehsil, crop=req.crop, primary_product=product,
        stock_rate=rate, in_stock=rate >= LOW_STOCK_THRESHOLD,
        best_stocked_product=best["product"], best_stocked_rate=best["stock_rate"],
    )


@app.get("/api/v1/stock/oos-alerts", response_model=InventoryAlertResponse,
         tags=["Utilities"], summary="Out-of-stock alerts across all tehsils")
def stock_oos_alerts(crop: str = Query("wheat")):
    from utils.stock_checker import LOW_STOCK_THRESHOLD
    checker      = _get_stock()
    oos          = checker.oos_tehsils(crop, threshold=0.0)
    low          = checker.oos_tehsils(crop, threshold=LOW_STOCK_THRESHOLD)
    low_not_oos  = [t for t in low if t not in oos]
    return InventoryAlertResponse(
        oos_tehsils=[{"tehsil": t, "stock_rate": 0.0} for t in oos],
        low_stock_tehsils=[{"tehsil": t} for t in low_not_oos],
        total_oos=len(oos), total_low_stock=len(low_not_oos),
        generated_at=_ts(),
    )


@app.post("/api/v1/attribution/compute", response_model=AttributionResponse,
          tags=["Utilities"], summary="14-day POS attribution for a single message")
def attribution_compute(req: AttributionRequest):
    try:
        engine      = _get_attribution()
        engine.window_days = req.window_days
        result      = engine.attribute(
            grower_id=req.grower_id, message_id=req.message_id,
            campaign_product=req.campaign_product, tehsil=req.tehsil,
            sent_date=datetime.fromisoformat(req.sent_date),
        )
        return AttributionResponse(**result.__dict__)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# Grower Lookup
# ─────────────────────────────────────────────
@app.get("/api/v1/growers/{grower_id}", response_model=GrowerProfileResponse,
         tags=["Growers"], summary="Grower profile with AI-enriched signals")
def get_grower(grower_id: str):
    growers = _get_growers()
    if growers.empty:
        raise HTTPException(status_code=503, detail="Grower data not loaded.")
    row = growers[growers["grower_id"] == grower_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"{grower_id} not found.")
    row   = row.iloc[0]
    today = datetime.utcnow()
    cal   = json.loads(row.get("grower_crop_calendar", "{}")) if pd.notna(row.get("grower_crop_calendar")) else {}
    stages     = cal.get("stages", [])
    stage      = get_growth_stage(stages, today)
    stage_days = days_to_next_stage(stages, today)
    device     = str(row.get("device_type", "unknown"))
    channel    = {"smartphone": "whatsapp", "keypad": "ivr_sms", "unknown": "field_rep"}.get(device, "field_rep")

    wa_path = DATA_DIR / "whatsapp_campaign.csv"
    open_rate = click_rate = engage_score = 0.0
    if wa_path.exists():
        wa = pd.read_csv(wa_path)
        g_wa = wa[wa["grower_id"] == grower_id]
        if not g_wa.empty:
            open_rate    = float(g_wa["opened_status"].mean())
            click_rate   = float(g_wa["clicked_status"].mean())
            engage_score = 0.2 * float(g_wa["delivered_status"].mean()) + 0.4 * open_rate + 0.4 * click_rate

    stock = _get_stock().get_stock_rate(str(row.get("tehsil", "")), str(row.get("crop", "wheat")))
    return GrowerProfileResponse(
        grower_id=grower_id, state=str(row.get("state", "")),
        district=str(row.get("district", "")), tehsil=str(row.get("tehsil", "")),
        language=str(row.get("language", "")), device_type=device,
        grower_age=int(row.get("grower_age", 0)), gender=str(row.get("gender", "")),
        grower_farm_size=float(row.get("grower_farm_size", 0)),
        crop=str(row.get("crop", "")), growth_stage=stage,
        days_to_next_stage=stage_days, primary_channel=channel,
        wa_open_rate=round(open_rate, 4), wa_click_rate=round(click_rate, 4),
        engagement_score=round(engage_score, 4), tehsil_stock_rate=round(stock, 4),
    )


@app.get("/api/v1/growers", tags=["Growers"], summary="List growers with optional filters")
def list_growers(state: Optional[str] = None, crop: Optional[str] = None,
                  device_type: Optional[str] = None,
                  limit: int = Query(20, ge=1, le=200),
                  offset: int = Query(0, ge=0)):
    df = _get_growers()
    if df.empty:
        return {"total": 0, "growers": []}
    if state:
        df = df[df["state"].str.lower() == state.lower()]
    if crop:
        df = df[df["crop"].str.lower() == crop.lower()]
    if device_type:
        df = df[df["device_type"].str.lower() == device_type.lower()]
    cols  = ["grower_id", "state", "district", "tehsil", "language",
              "device_type", "grower_age", "gender", "grower_farm_size", "crop"]
    avail = [c for c in cols if c in df.columns]
    page = df[avail].iloc[offset:offset + limit]
    return {"total": len(df), "offset": offset, "limit": limit,
            "growers": page.astype(object).where(pd.notna(page), other=None).to_dict("records")}


# ─────────────────────────────────────────────
# Analytics
# ─────────────────────────────────────────────
@app.get("/api/v1/analytics/campaign", response_model=CampaignAnalyticsResponse,
         tags=["Analytics"], summary="WhatsApp campaign funnel analytics")
def campaign_analytics():
    wa_path = DATA_DIR / "whatsapp_campaign.csv"
    if not wa_path.exists():
        raise HTTPException(status_code=503, detail="WhatsApp data not found.")
    wa      = pd.read_csv(wa_path)
    growers = _get_growers()
    if not growers.empty:
        wa = wa.merge(growers[["grower_id", "language", "state", "device_type"]],
                      on="grower_id", how="left")

    def funnel(df):
        n = len(df)
        return {"sent": n,
                "delivered": int(df["delivered_status"].sum()),
                "opened":    int(df["opened_status"].sum()),
                "clicked":   int(df["clicked_status"].sum()),
                "delivery_rate": round(df["delivered_status"].mean(), 4),
                "open_rate":     round(df["opened_status"].mean(), 4),
                "click_rate":    round(df["clicked_status"].mean(), 4)}

    overall     = funnel(wa)
    by_campaign = []
    if "campaign_id" not in wa.columns:
        wa["campaign_id"] = (
            wa["campaign_product"].str.replace(" ", "_", regex=False)
            + "_"
            + wa["campaign_crop"].str.replace(" ", "_", regex=False)
        )
    for (cid, crop, prod), grp in wa.groupby(["campaign_id", "campaign_crop", "campaign_product"]):
        f = funnel(grp)
        by_campaign.append(FunnelMetrics(
            campaign_id=str(cid), campaign_crop=str(crop), campaign_product=str(prod),
            messages_sent=f["sent"], delivered=f["delivered"],
            opened=f["opened"], clicked=f["clicked"],
            delivery_rate=f["delivery_rate"], open_rate=f["open_rate"],
            click_rate=f["click_rate"],
            open_to_click_rate=round(f["clicked"] / f["opened"], 4) if f["opened"] else 0.0,
        ))

    def by_col(col):
        if col not in wa.columns:
            return []
        return sorted(
            [{"" + col: str(v), **funnel(g)}
             for v, g in wa.groupby(col)],
            key=lambda x: x["click_rate"], reverse=True,
        )

    top_tehsils = []
    if not growers.empty:
        wa_t = wa.merge(growers[["grower_id", "tehsil"]], on="grower_id", how="left")
        if "tehsil" in wa_t.columns:
            tg = (wa_t.groupby("tehsil")["clicked_status"]
                  .agg(messages="count", clicks="sum").reset_index())
            tg["click_rate"] = (tg["clicks"] / tg["messages"]).round(4)
            top_tehsils = tg.sort_values("click_rate", ascending=False).head(10).to_dict("records")

    return CampaignAnalyticsResponse(
        overall=overall, by_campaign=by_campaign,
        by_language=by_col("language"), by_state=by_col("state"),
        by_device=by_col("device_type"), top_tehsils=top_tehsils,
        generated_at=_ts(),
    )


@app.get("/api/v1/analytics/pos", tags=["Analytics"],
         summary="POS sales by product and monthly trend")
def pos_analytics(top_n: int = Query(10, ge=1, le=50)):
    p = DATA_DIR / "retailer_pos.csv"
    if not p.exists():
        raise HTTPException(status_code=503, detail="POS data not found.")
    pos = pd.read_csv(p)
    pos["revenue"]          = pos["sku_qty"] * pos["sku_price"]
    pos["transaction_date"] = pd.to_datetime(pos["transaction_date"])

    by_product = (
        pos.groupby("sku_name")
        .agg(total_qty=("sku_qty", "sum"), total_revenue=("revenue", "sum"),
             transactions=("transaction_id", "count"))
        .sort_values("total_revenue", ascending=False)
        .head(top_n).reset_index().to_dict("records")
    )
    monthly = (
        pos.groupby(pos["transaction_date"].dt.to_period("M").astype(str))
        .agg(qty=("sku_qty", "sum"), revenue=("revenue", "sum"))
        .reset_index().rename(columns={"transaction_date": "month"})
        .to_dict("records")
    )
    return {"by_product": by_product, "monthly_trend": monthly,
            "total_revenue": round(float(pos["revenue"].sum()), 2),
            "total_qty": int(pos["sku_qty"].sum()), "generated_at": _ts()}


# ─────────────────────────────────────────────
# Error handler
# ─────────────────────────────────────────────
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    return JSONResponse(status_code=500,
                        content=ErrorResponse(
                            error=type(exc).__name__,
                            detail=str(exc), timestamp=_ts(),
                        ).model_dump())