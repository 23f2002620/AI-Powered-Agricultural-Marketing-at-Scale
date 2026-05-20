"""
api/schemas.py — Pydantic v2 request & response models for all API endpoints.
"""

from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
VALID_CROPS = [
    "wheat", "mustard", "chickpea", "potato", "barley",
    "lentil", "safflower", "cumin", "maize",
]
# 11 Indian languages matching LANGUAGE_META in content_generator.py
VALID_LANGUAGES = [
    "Hindi", "Punjabi", "Marathi", "Gujarati", "Kannada",
    "Bengali", "Tamil", "Telugu", "Odia", "Assamese", "Malayalam",
]
VALID_CHANNELS = [
    "whatsapp_rich", "whatsapp_text", "ivr_voice",
    "sms", "field_rep_brief", "social_post",
]
VALID_DEVICES = ["smartphone", "keypad", "unknown"]
VALID_TIERS   = ["high", "medium", "low"]

# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    models_loaded: dict[str, bool]

# ─────────────────────────────────────────────
# Engine 1 – Content Generation
# ─────────────────────────────────────────────
class ContentGenerateRequest(BaseModel):
    grower_id: str = Field(..., description="Unique grower identifier")
    crop: str      = Field(..., description="Crop name")
    growth_stage: str = Field(..., description="Current growth stage")
    language: str  = Field(..., description="One of 11 supported Indian languages")
    state: str     = Field(..., description="Indian state")
    tehsil: str    = Field(..., description="Tehsil identifier")
    content_format: str = Field("whatsapp_rich", description="Output format")
    device_type: str    = Field("smartphone")
    grower_name: str    = Field("किसान भाई")
    farm_size_acres: float = Field(2.0, ge=0.1, le=500)
    nearest_retailer: str  = Field("")
    active_threats: list[str] = Field(default_factory=list)
    days_to_next_stage: int   = Field(14, ge=0, le=120)
    weather_alert: str        = Field("")
    rep_name: str             = Field("")
    pest_pressure_index: float = Field(0.0, ge=0.0, le=1.0,
        description="0-1 from ICAR-NCIPM; injected by orchestrator automatically")
    weather_risk_score: float  = Field(0.0, ge=0.0, le=1.0,
        description="0-1 from IMD; injected by orchestrator automatically")

    @field_validator("crop")
    @classmethod
    def validate_crop(cls, v):
        if v not in VALID_CROPS:
            raise ValueError(f"crop must be one of {VALID_CROPS}")
        return v

    @field_validator("language")
    @classmethod
    def validate_language(cls, v):
        if v not in VALID_LANGUAGES:
            raise ValueError(f"language must be one of {VALID_LANGUAGES}")
        return v

    @field_validator("content_format")
    @classmethod
    def validate_format(cls, v):
        if v not in VALID_CHANNELS:
            raise ValueError(f"content_format must be one of {VALID_CHANNELS}")
        return v

    @field_validator("device_type")
    @classmethod
    def validate_device(cls, v):
        if v not in VALID_DEVICES:
            raise ValueError(f"device_type must be one of {VALID_DEVICES}")
        return v


class ContentGenerateResponse(BaseModel):
    grower_id: str
    content_format: str
    language: str
    text: str
    char_count: int
    llm_model: str
    prompt_tokens: int
    completion_tokens: int
    generation_timestamp: str
    tts_audio_url: Optional[str] = None          # Sarvam TTS audio (IVR channel)
    video_thumbnail_url: Optional[str] = None    # Visual/video card (data-URI or URL)
    video_caption: Optional[str] = None          # Alt-text caption for visual card
    metadata: dict[str, Any] = Field(default_factory=dict)

# ─────────────────────────────────────────────
# Engine 2 – Targeting / Bandit
# ─────────────────────────────────────────────
class TargetingDecisionRequest(BaseModel):
    grower_id: str
    device_type: str  = "smartphone"
    language: str     = "Hindi"
    crop: str         = "wheat"
    growth_stage: str = "tillering"
    season_progress: float         = Field(0.5,  ge=0.0, le=1.0)
    days_to_next_stage: int        = Field(14,   ge=0,   le=120)
    hist_open_rate: float          = Field(0.0,  ge=0.0, le=1.0)
    hist_click_rate: float         = Field(0.0,  ge=0.0, le=1.0)
    tehsil_stock_rate: float       = Field(0.7,  ge=0.0, le=1.0)
    days_since_rep_visit: int      = Field(30,   ge=0)
    digital_literacy_score: float  = Field(0.7,  ge=0.0, le=1.0)
    grower_age: int                = Field(45,   ge=18,  le=100)
    farm_size_acres: float         = Field(2.0,  ge=0.1)
    offline_campaign_attended: bool = False
    product_scan_done: bool         = False
    weather_risk_score: float      = Field(0.3,  ge=0.0, le=1.0)
    pest_pressure_index: float     = Field(0.3,  ge=0.0, le=1.0)


class TargetingDecisionResponse(BaseModel):
    grower_id: str
    selected_arm: str
    channel: str
    time_slot: str
    creative_variant: str
    ucb_score: float
    eligible_arms_count: int
    decision_timestamp: str


class BanditFeedbackRequest(BaseModel):
    grower_id: str
    arm: str = Field(..., description="channel|time_slot|creative_variant")
    device_type: str  = "smartphone"
    language: str     = "Hindi"
    crop: str         = "wheat"
    growth_stage: str = "tillering"
    season_progress: float        = Field(0.5,  ge=0.0, le=1.0)
    days_to_next_stage: int       = 14
    hist_open_rate: float         = 0.0
    hist_click_rate: float        = 0.0
    tehsil_stock_rate: float      = 0.7
    days_since_rep_visit: int     = 30
    digital_literacy_score: float = 0.7
    grower_age: int               = 45
    farm_size_acres: float        = 2.0
    offline_campaign_attended: bool = False
    product_scan_done: bool         = False
    weather_risk_score: float     = 0.3
    pest_pressure_index: float    = 0.3
    delivered: bool  = False
    opened: bool     = False
    clicked: bool    = False
    purchased: bool  = False


class BanditFeedbackResponse(BaseModel):
    grower_id: str
    arm: str
    reward: float
    total_rounds: int
    message: str


class BanditStatsResponse(BaseModel):
    total_rounds: int
    n_arms: int
    top_arms: list[dict[str, Any]]

# ─────────────────────────────────────────────
# Engine 3 – Receptivity
# ─────────────────────────────────────────────
class ReceptivityRequest(BaseModel):
    grower_id: str
    grower_age: int           = Field(45,  ge=18, le=100)
    grower_farm_size: float   = Field(2.0, ge=0.1)
    language: str             = "Hindi"
    device_type: str          = "smartphone"
    crop: str                 = "wheat"
    state: str                = "Uttar Pradesh"
    gender: str               = "male"
    offline_campaign_attended: bool = False
    product_scan: bool              = False
    hist_open_rate: float     = Field(0.0, ge=0.0, le=1.0)
    hist_click_rate: float    = Field(0.0, ge=0.0, le=1.0)
    hist_delivery_rate: float = Field(1.0, ge=0.0, le=1.0)
    hist_pos_rate: float      = Field(0.0, ge=0.0, le=1.0)   # POS conversion rate
    cum_messages: int         = Field(0,   ge=0)
    days_to_harvest: int      = Field(60,  ge=0)
    season_progress: float    = Field(0.5, ge=0.0, le=1.0)
    tehsil_stock_rate: float  = Field(0.7, ge=0.0, le=1.0)
    days_since_rep_visit: int = Field(30,  ge=0)


class ReceptivityResponse(BaseModel):
    grower_id: str
    score: float
    tier: str
    recommended_action: str
    score_components: dict[str, Any]


class BatchReceptivityRequest(BaseModel):
    growers: list[ReceptivityRequest] = Field(..., min_length=1, max_length=1000)


class BatchReceptivityResponse(BaseModel):
    results: list[ReceptivityResponse]
    total: int
    high_tier_count: int
    medium_tier_count: int
    low_tier_count: int
    avg_score: float

# ─────────────────────────────────────────────
# Engine 4 – Micro-Segmentation
# ─────────────────────────────────────────────
class SegmentRequest(BaseModel):
    grower_id: str
    crop: str         = "wheat"
    device_type: str  = "smartphone"
    language: str     = "Hindi"
    state: str        = "Uttar Pradesh"
    grower_age: int   = Field(45, ge=18, le=100)
    farm_size: float  = Field(2.0, ge=0.1)
    open_rate: float  = Field(0.0, ge=0.0, le=1.0)
    click_rate: float = Field(0.0, ge=0.0, le=1.0)
    offline_attended: bool = False
    product_scan: bool     = False


class SegmentResponse(BaseModel):
    grower_id: str
    segment_id: str
    recommended_channel: str
    recommended_product: str
    content_format: str
    language: str
    tts_code: str
    llm_prompt_template: str
    segment_size: int
    avg_segment_click_rate: float

# ─────────────────────────────────────────────
# Campaign Orchestrator
# ─────────────────────────────────────────────
class CampaignPlanRequest(BaseModel):
    campaign_id: str    = Field("CMP_RABI25_AI")
    target_crop: Optional[str] = Field(None)
    max_growers: int    = Field(500, ge=1, le=6000)
    min_receptivity: float = Field(0.03, ge=0.0, le=1.0)
    min_stock_rate: float  = Field(0.2,  ge=0.0, le=1.0)
    api_key: str           = Field("", description="Google Gemini API key")
    sarvam_api_key: str  = Field("", description="Sarvam TTS API key for IVR audio")
    imd_key: str           = Field("", description="IMD weather API key")
    ncipm_key: str         = Field("", description="ICAR-NCIPM pest surveillance API key")
    agmarknet_key: str     = Field("", description="Agmarknet mandi price API key")


class CampaignPlanSummary(BaseModel):
    campaign_id: str
    total_growers_targeted: int
    emergency_sends: int
    channel_mix: dict[str, int]
    language_mix: dict[str, int]
    crop_mix: dict[str, int]
    avg_receptivity_score: float
    avg_stock_rate: float
    avg_weather_risk: float
    avg_pest_pressure: float
    oos_growers_excluded: int
    low_receptivity_excluded: int
    output_file: str
    planned_at: str


class FeedbackUpdateRequest(BaseModel):
    campaign_id: str
    results: list[dict[str, Any]]


class FeedbackUpdateResponse(BaseModel):
    campaign_id: str
    records_processed: int
    bandit_total_rounds: int
    message: str


class AutoPOSFeedbackRequest(BaseModel):
    campaign_plan_csv: str = Field(...,
        description="Filename (not path) of the campaign plan CSV in results/")
    window_days: int = Field(14, ge=1, le=90)


class AutoPOSFeedbackResponse(BaseModel):
    updated: int
    converted: int
    conversion_rate: float
    bandit_total_rounds: int
    message: str

# ─────────────────────────────────────────────
# External Signals
# ─────────────────────────────────────────────
class ExternalSignalsRequest(BaseModel):
    tehsil: str
    state: str
    crop: str
    imd_key: str      = ""
    ncipm_key: str    = ""
    agmarknet_key: str = ""


class ExternalSignalsResponse(BaseModel):
    tehsil: str
    crop: str
    weather_risk_score: float
    weather_alert: str
    pest_pressure_index: float
    pest_severity: str
    pest_name: str
    pest_advisory: str
    is_emergency: bool
    mandi_price_per_quintal: float
    price_trend: str
    composite_urgency_score: float
    sources: dict[str, str]

# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────
class StockCheckRequest(BaseModel):
    tehsil: str
    crop: str


class StockCheckResponse(BaseModel):
    tehsil: str
    crop: str
    primary_product: str
    stock_rate: float
    in_stock: bool
    best_stocked_product: str
    best_stocked_rate: float
    low_stock_threshold: float = 0.2


class AttributionRequest(BaseModel):
    grower_id: str
    message_id: str
    campaign_product: str
    tehsil: str
    sent_date: str   = Field(..., description="ISO datetime e.g. 2026-03-01T08:00:00")
    window_days: int = Field(14, ge=1, le=90)


class AttributionResponse(BaseModel):
    grower_id: str
    message_id: str
    campaign_product: str
    tehsil: str
    sent_date: str
    window_end: str
    attributed_sales_count: int
    attributed_qty: float
    attributed_revenue: float
    converted: bool
    days_to_first_sale: Optional[int]

# ─────────────────────────────────────────────
# Grower Lookup
# ─────────────────────────────────────────────
class GrowerProfileResponse(BaseModel):
    grower_id: str
    state: str
    district: str
    tehsil: str
    language: str
    device_type: str
    grower_age: int
    gender: str
    grower_farm_size: float
    crop: str
    growth_stage: str
    days_to_next_stage: int
    primary_channel: str
    wa_open_rate: float
    wa_click_rate: float
    engagement_score: float
    tehsil_stock_rate: float
    segment_id: Optional[str]     = None
    receptivity_score: Optional[float] = None

# ─────────────────────────────────────────────
# Analytics
# ─────────────────────────────────────────────
class FunnelMetrics(BaseModel):
    campaign_id: str
    campaign_crop: str
    campaign_product: str
    messages_sent: int
    delivered: int
    opened: int
    clicked: int
    delivery_rate: float
    open_rate: float
    click_rate: float
    open_to_click_rate: float


class CampaignAnalyticsResponse(BaseModel):
    overall: dict[str, Any]
    by_campaign: list[FunnelMetrics]
    by_language: list[dict[str, Any]]
    by_state: list[dict[str, Any]]
    by_device: list[dict[str, Any]]
    top_tehsils: list[dict[str, Any]]
    generated_at: str


class InventoryAlertResponse(BaseModel):
    oos_tehsils: list[dict[str, Any]]
    low_stock_tehsils: list[dict[str, Any]]
    total_oos: int
    total_low_stock: int
    generated_at: str


class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: str