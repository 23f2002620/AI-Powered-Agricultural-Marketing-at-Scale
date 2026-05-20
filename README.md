# AI-Powered Hyper-Personalized Agricultural Marketing System

A closed-loop, omnichannel campaign engine for Rabi 2025-26 that targets ~6,000 Indian
growers with the right message, channel, language, and timing — driven by four AI engines
and grounded in real POS conversion data.

---

## Table of Contents

1. [What This System Does](#what-this-system-does)
2. [Project Structure](#project-structure)
3. [Data Inputs](#data-inputs)
4. [The Four AI Engines](#the-four-ai-engines)
5. [Key Innovations](#key-innovations)
6. [Setup & Installation](#setup--installation)
7. [Running the Pipeline](#running-the-pipeline)
8. [API Server](#api-server)
9. [Configuration & API Keys](#configuration--api-keys)
10. [Expected Outcomes](#expected-outcomes)
11. [Architecture Overview](#architecture-overview)

---

## What This System Does

Instead of running 4 generic Rabi campaigns (Wheat / Mustard / Chickpea / Potato),
KisanAI generates **thousands of micro-targeted campaign variants** by combining:

- Grower context (crop stage, language, device type)
- Geo-temporal signals (IMD weather, ICAR-NCIPM pest alerts)
- Real-time stock availability from `retailer_inventory_weekly`
- Historical engagement from `whatsapp_campaign` logs

Every message is timed to arrive **3–7 days before the grower's next crop stage**,
delivered on the channel their device supports, and blocked if the product is out of
stock at all nearby retailers.

**Primary KPI:**
```
Campaign-to-Action Conversion Rate =
    (POS purchases of campaign_product within 14 days of message)
    / (Messages delivered)
```

---

## Project Structure

```
agri_marketing/
├── data/                          # All 8 source CSV files
│   ├── growers.csv                # 6,000 growers — identity & crop calendar
│   ├── whatsapp_campaign.csv      # 4,479 WA message logs — engagement labels
│   ├── retailer_pos.csv           # 235,042 POS transactions — ground-truth conversion
│   ├── retailer_inventory_weekly.csv  # 310,544 rows — stock availability
│   ├── retailer_visit_log.csv     # 30,000 rep visit records
│   ├── retailers.csv              # 4,000 retailer profiles
│   ├── reps_territory.csv         # 500 rep-territory mappings
│   └── digital_funnel_weekly.csv  # 104 weekly funnel snapshots
│
├── engines/                       # Core AI engines (inference)
│   ├── content_generator.py       # Engine 1 — GenAI content (Ollama/Gemini + Bhashini TTS)
│   ├── targeting_optimizer.py     # Engine 2 — LinUCB + Thompson Sampling bandit
│   ├── receptivity_predictor.py   # Engine 3 — XGBoost/LightGBM scoring
│   ├── micro_segmentation.py      # Engine 4 — HDBSCAN ~200 micro-segments
│   └── campaign_orchestrator.py   # Ties all 4 engines into a single pipeline
│
├── scripts/                       # One-time training & setup scripts
│   ├── 01_eda.py                  # Exploratory data analysis
│   ├── 02_feature_store.py        # Build Grower-360 feature store
│   ├── 03_train_receptivity.py    # Train XGBoost + LightGBM receptivity model
│   ├── 04_micro_segmentation.py   # Cluster growers into ~200 micro-segments
│   ├── 05_attribution_analysis.py # POS attribution analysis
│   └── 06_run_pipeline.py         # End-to-end pipeline runner
│
├── api/
│   ├── main.py                    # FastAPI server
│   └── schemas.py                 # Request/response schemas
│
├── frontend/
│   └── dashboard.html             # Campaign monitoring dashboard
│
└── requirements.txt
```

---

## Data Inputs

| File | Rows | Role |
|---|---|---|
| `growers.csv` | 6,000 | Primary identity, crop calendar JSON, device type, language |
| `whatsapp_campaign.csv` | 4,479 | Engagement ground truth — open/click labels for training |
| `retailer_pos.csv` | 235,042 | Ultimate success metric — POS conversion signal |
| `retailer_inventory_weekly.csv` | 310,544 | Stock-aware messaging — blocks OOS promotions |
| `retailer_visit_log.csv` | 30,000 | Rep visit recency & frequency per tehsil |
| `retailers.csv` | 4,000 | Nearest-retailer injection into personalized messages |
| `reps_territory.csv` | 500 | Field rep routing for low-digital-literacy growers |
| `digital_funnel_weekly.csv` | 104 | Funnel benchmarks (impression → lead → conversion) |

---

## The Four AI Engines

### Engine 1 — Context-Aware Content Generator (`content_generator.py`)

Generates channel-specific content in 11 Indian languages for every grower segment.

**Generation priority:**
1. **Ollama** (local Gemma4) — fully offline, no API key needed
2. **Google Gemini** — cloud upgrade if `GEMINI_API_KEY` is set
3. **Rule-based fallback** — always works, all 11 languages, zero dependencies

**TTS priority for IVR:**
1. **Bhashini** (solution doc primary — govt-backed Indic TTS, 11 languages)
2. **Sarvam AI** — high-quality cloud fallback
3. **pyttsx3** — local offline fallback, no API key needed

**Output formats per grower:**

| Format | Channel | Max Length |
|---|---|---|
| Rich WhatsApp (image + caption) | Smartphone | 1,024 chars |
| WhatsApp text | Smartphone | 512 chars |
| IVR voice script + TTS audio | Keypad users | 30 sec |
| SMS | Keypad + literate | 160 chars |
| Field rep visit brief | Low-receptivity / high-LTV | Full brief |
| Social post | WhatsApp Status / Facebook | 300 chars |

**ICAR RAG:** An inline knowledge base of ICAR-NCIPM advisories (wheat rust, mustard aphid,
potato late blight, chickpea pod borer, etc.) is injected into every LLM system prompt,
grounding the content in verified agronomic guidance.

---

### Engine 2 — Targeting & Timing Optimizer (`targeting_optimizer.py`)

Decides the optimal `channel × time_of_day × creative_variant` arm for each grower.

**Algorithms:**

- **LinUCB** (Contextual Multi-Armed Bandit) — for growers with existing WA engagement history.
  Each of the 60 arms (`5 channels × 3 time slots × 4 creative variants`) has its own
  ridge regression model that updates in real time from POS feedback.

- **Thompson Sampling** — for cold-start growers with zero WA history. Explores arms via
  Beta distribution sampling; hands off to LinUCB once ≥3 engagement events are recorded.

**Arm cost table (₹ per delivery):**

| Channel | Cost |
|---|---|
| WhatsApp text | ₹0.10 |
| WhatsApp rich (image/video) | ₹0.30 |
| SMS | ₹0.15 |
| IVR voice | ₹0.80 |
| Retailer push | ₹5.00 |
| Field rep visit | ₹50.00 |

**Device constraints:** Smartphones → WhatsApp channels only. Keypad → IVR + SMS only.
Unknown → Field rep or SMS.

---

### Engine 3 — Campaign Receptivity Predictor (`receptivity_predictor.py`)

Predicts `P(purchase within 14 days | grower, message, channel)` before any message is sent.

**Training:** Join `whatsapp_campaign` ⟕ `growers` ⟕ `retailer_pos` on tehsil + date
within a 14-day attribution window. Both XGBoost and LightGBM are trained; the model with
the higher cross-validated ROC-AUC is saved as `models/receptivity_model.pkl`.

**Feature set (23 features):**

| Category | Features |
|---|---|
| Demographic | `grower_age`, `grower_farm_size`, `gender`, `language`, `device_type`, `state` |
| Behavioral | `hist_open_rate`, `hist_click_rate`, `hist_delivery_rate`, `hist_pos_rate`, `cum_messages`, `offline_campaign_attended`, `product_scan` |
| Contextual | `tehsil_stock_rate`, `days_since_rep_visit`, `tehsil_pest_pressure_index`, `whatsapp_propensity` |
| Temporal | `days_to_harvest`, `season_progress`, `month`, `day_of_week`, `week_of_year` |
| Crop | `crop_enc` |

**Tier routing:**

| Score | Tier | Action |
|---|---|---|
| ≥ 0.15 | High | Send immediately via WhatsApp rich |
| 0.05–0.15 | Medium | Send WhatsApp text or IVR |
| < 0.05 (+ rep assigned) | Low → Rep | Route to field rep visit brief |

---

### Engine 4 — Personalization Scaler (`micro_segmentation.py`)

Clusters 6,000 growers into ~200 micro-segments for scalable personalization.

**Approach:**
1. Each grower is encoded into a 64-dim embedding (Grower Tower) via PCA over
   one-hot categoricals + scaled numerics + behavioral flags.
2. **HDBSCAN** (`min_cluster_size=30`) clusters growers into ~200 segments vs.
   the traditional 5–10, enabling far more precise targeting.
3. One LLM prompt template per cluster, then hyper-personalized with four variables:
   `{name_token}`, `{tehsil}`, `{crop_stage}`, `{nearest_retailer}`.
4. **Effort scaling:** 200 templates → 6,000 personalized variants with zero additional
   human writing.

---

## Key Innovations

### 1. Stock-Synchronized Messaging
Never promote a product when `retailer_inventory_weekly.sku_qty == 0` at all nearby
retailers. Two-layer gate:
- **Hard block** (`stockout_risk = 1`): all retailers in tehsil are OOS → grower excluded
  entirely from the campaign batch.
- **Soft filter** (`tehsil_stock_rate < 0.2`): fewer than 20% of retailers have stock →
  grower deprioritized.

### 2. Crop-Calendar-Triggered Campaigns
`grower_crop_calendar` JSON is parsed to compute `days_to_next_stage` for every grower.
Growers within a **3–7 day pre-stage window** receive a 1.35× receptivity score boost,
surfacing them at the top of the campaign selection list.

### 3. Digital + Field Rep Hybrid
When the receptivity model predicts low digital conversion probability AND the grower has
an assigned rep (`rep_id` from `reps_territory`), the orchestrator auto-routes the grower
to `field_rep_brief` channel — the rep gets a personalized talking-points brief for a
face-to-face visit.

### 4. Pest Surveillance Webhook
ICAR-NCIPM alert in a tehsil → emergency flag propagates to all affected growers →
content is generated and queued within 2 hours (controlled by `EMERGENCY_CONTENT_WINDOW_HRS = 2`).

---

## Setup & Installation

**Requirements:** Python 3.10+

```bash
# 1. Clone / unzip the project
cd agri_marketing

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Install LightGBM for faster receptivity model
pip install lightgbm

# 4. (Optional) Install Ollama for local LLM inference
#    https://ollama.com/download
ollama pull gemm4

```

---

## Running the Pipeline

Run scripts in order from the `agri_marketing/` directory:

```bash
# Step 1 — Exploratory data analysis (optional, produces reports/)
python scripts/01_eda.py

# Step 2 — Build Grower-360 feature store (required before training)
python scripts/02_feature_store.py
# Output: results/grower_feature_store.parquet

# Step 3 — Train receptivity model (XGBoost + LightGBM, best saved)
python scripts/03_train_receptivity.py
# Output: models/receptivity_model.pkl  (target AUC ≥ 0.78)

# Step 4 — Cluster growers into ~200 micro-segments
python scripts/04_micro_segmentation.py
# Output: models/embedding_pipeline.pkl, results/segment_profiles.parquet

# Step 5 — Attribution analysis (optional diagnostics)
python scripts/05_attribution_analysis.py

# Step 6 — Run the full campaign pipeline
python scripts/06_run_pipeline.py
# Output: results/campaign_plan_CMP_RABI25_AI_<date>.csv
```

**Quick demo (single engine):**
```bash
python engines/content_generator.py    # Engine 1 demo
python engines/targeting_optimizer.py  # Engine 2 demo
python engines/receptivity_predictor.py # Engine 3 demo
python engines/micro_segmentation.py   # Engine 4 demo
```

**Run a campaign batch directly:**
```python
from engines.campaign_orchestrator import run_campaign

results = run_campaign(
    crop="wheat",
    max_growers=500,
    api_key="",           # Gemini (optional)
    sarvam_api_key="",    # Sarvam TTS (optional)
    bhashini_api_key="",  # Bhashini TTS (optional, recommended for IVR)
)
```

---

## API Server

```bash
cd agri_marketing
uvicorn api.main:app --reload --port 8000
```

Interactive docs available at `http://localhost:8000/docs`.

---

## Configuration & API Keys

All keys are optional. The system degrades gracefully when none are set.

| Environment Variable | Purpose | Fallback |
|---|---|---|
| `GEMINI_API_KEY` | Richer LLM content via Google Gemini | Ollama local → rule-based |
| `OLLAMA_MODEL` | Local model name (default: `llama3`) | `gemma2` also works |
| `BHASHINI_API_KEY` | Bhashini IVR TTS (format: `userID\|ulcaApiKey`) | Sarvam → pyttsx3 |
| `SARVAM_API_KEY` | High-quality Indic TTS | pyttsx3 |
| `STABILITY_API_KEY` | Real AI-generated crop images | SVG placeholder |
| `IMD_API_KEY` | Live weather signals | Cached / default values |
| `NCIPM_API_KEY` | Live pest pressure index | Cached / 0.0 |
| `AGMARKNET_KEY` | Mandi price data | Not used in core pipeline |

Create a `.env` file in `agri_marketing/` and the system will pick them up automatically
via `python-dotenv`.

**Nightly signal pre-caching** (run via cron at 11 PM for offline daytime operation):
```bash
cd agri_marketing && python -c "
from engines.campaign_orchestrator import CampaignOrchestrator
CampaignOrchestrator().precache_signals()
"
```

---

## Expected Outcomes

| Metric | Baseline (mass campaign) | KisanAI (targeted) | Lift |
|---|---|---|---|
| WhatsApp open rate | ~22% | 45–55% | ~2× |
| Click-through rate | ~3% | 8–12% | ~3× |
| Campaign-to-POS conversion | ~0.8% | 3–5% | ~4× |
| Cost per conversion | ₹180 | ₹55 | −70% |
| Creative variants per campaign | 4 | 800–2,000+ | 500× |
| Human content writing effort | Linear with variants | Near-flat (LLM) | — |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   DATA INGESTION LAYER                  │
│  Internal CSVs (8 tables)  +  External APIs             │
│  growers · pos · inventory · wa_log · visit_log · reps  │
│  IMD weather · ICAR-NCIPM pest · Bhuvan NDVI            │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│               FEATURE ENGINEERING LAYER                 │
│  Grower-360 Profile  ·  Geo-Temporal Risk               │
│  stockout_risk · local_sku_velocity · whatsapp_propensity│
│  days_to_next_stage · rep_touch_frequency               │
└──┬──────────────┬───────────────┬───────────┬───────────┘
   │              │               │           │
┌──▼──┐      ┌───▼───┐      ┌────▼───┐  ┌───▼──────────┐
│ E1  │      │  E2   │      │   E3   │  │     E4        │
│Cont │      │LinUCB │      │XGB/LGB │  │ HDBSCAN       │
│Gen  │      │+Thomp │      │Receptiv│  │ ~200 segments │
│GenAI│      │Samplng│      │Predictr│  │ LLM prompts   │
└──┬──┘      └───┬───┘      └────┬───┘  └───┬───────────┘
   └─────────────┴───────────────┴───────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│          CAMPAIGN ORCHESTRATOR (6-step pipeline)        │
│  1. Stock gate (stockout_risk hard-block)               │
│  2. External signals (IMD / ICAR / cache)              │
│  3. Receptivity scoring + pre-stage boost + rep routing │
│  4. Micro-segment assignment                            │
│  5. Bandit arm selection (LinUCB / Thompson)            │
│  6. Content generation (LLM + Bhashini TTS)            │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│               OMNICHANNEL DELIVERY                      │
│  WhatsApp Rich  ·  WhatsApp Text  ·  IVR Voice          │
│  SMS  ·  Field Rep Brief  ·  Social Post                │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              CLOSED-LOOP FEEDBACK                       │
│  WA engagement → POS lift → Rep visit log               │
│  → Bandit reward update (real-time)                     │
│  → Receptivity model retrain (daily)                   │
│  → Grower re-segmentation (weekly)                     │
└─────────────────────────────────────────────────────────┘
```
