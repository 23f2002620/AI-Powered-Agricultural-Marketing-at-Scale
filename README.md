# AI-Powered Hyper-Personalized Agricultural Marketing System


---

## Table of Contents

1. [What This System Does](#1-what-this-system-does)
2. [Project Structure](#2-project-structure)
3. [Architecture — Three Zones](#3-architecture--three-zones)
4. [The Four AI Engines](#4-the-four-ai-engines)
5. [Data Inputs](#5-data-inputs)
6. [Key Innovations](#6-key-innovations)
7. [Setup & Installation](#7-setup--installation)
8. [Running the Pipeline](#8-running-the-pipeline)
9. [Three-Zone Runner (New)](#9-three-zone-runner-new)
10. [API Server](#10-api-server)
11. [Configuration & API Keys](#11-configuration--api-keys)
12. [Expected Outcomes](#12-expected-outcomes)

---

## 1. What This System Does

Instead of running 4 generic Rabi campaigns (Wheat / Mustard / Chickpea / Potato), KisanAI generates **thousands of micro-targeted campaign variants** by combining:

- Grower context — crop stage, language, device type, farm size
- Geo-temporal signals — IMD weather, ICAR-NCIPM pest alerts, Agmarknet mandi prices
- Real-time stock availability — from `retailer_inventory_weekly`
- Historical engagement — from `whatsapp_campaign` logs and POS transactions

Every message is timed to arrive **3–7 days before the grower's next crop stage**, delivered on the channel their device actually supports, and silently suppressed if the product is out of stock at all nearby retailers.

**Primary KPI:**
```
Campaign-to-Action Conversion Rate =
    POS purchases of campaign_product within 14 days of message send
    ─────────────────────────────────────────────────────────────────
                    Messages delivered
```

---

## 2. Project Structure

```
agri_marketing/
│
├── data/                               # All 8 source CSV files
│   ├── growers.csv                     # 6,000 growers — identity, crop calendar, device
│   ├── whatsapp_campaign.csv           # 4,479 WA logs — open/click engagement labels
│   ├── retailer_pos.csv                # 235,042 POS transactions — ground-truth conversion
│   ├── retailer_inventory_weekly.csv   # 310,544 rows — weekly stock per SKU per retailer
│   ├── retailer_visit_log.csv          # 30,000 rep visit records
│   ├── retailers.csv                   # 4,000 retailer profiles with tehsil mapping
│   ├── reps_territory.csv              # 500 rep–territory–grower assignments
│   ├── digital_funnel_weekly.csv       # 104 weekly funnel snapshots
│   └── signal_cache.json               # Nightly external signal cache (auto-generated)
│
├── engines/                            # Core AI engines (inference only)
│   ├── content_generator.py            # Engine 1 — GenAI content + Bhashini/Sarvam TTS
│   ├── targeting_optimizer.py          # Engine 2 — LinUCB + Thompson Sampling bandit
│   ├── receptivity_predictor.py        # Engine 3 — XGBoost/LightGBM scoring
│   ├── micro_segmentation.py           # Engine 4 — HDBSCAN ~200 micro-segments
│   └── campaign_orchestrator.py        # Original 6-step orchestrator
│
├── zones/                              # Three-zone architecture (new)
│   ├── zone1_nightly_batch.py          # Zone 1 — Nightly batch: signals + engines + queue
│   ├── zone2_delivery_engine.py        # Zone 2 — Delivery: queue flush + channel router
│   └── zone3_feedback_loop.py          # Zone 3 — Feedback: POS attribution + bandit update
│
├── utils/
│   ├── external_signals.py             # IMD / ICAR-NCIPM / Agmarknet + offline cache
│   ├── stock_checker.py                # Tehsil-level stock availability lookup
│   ├── crop_calendar.py                # days_to_next_stage, get_growth_stage
│   └── attribution.py                  # POS attribution engine
│
├── scripts/                            # One-time training & setup scripts
│   ├── 01_eda.py                       # Exploratory data analysis
│   ├── 02_feature_store.py             # Build Grower-360 feature store
│   ├── 03_train_receptivity.py         # Train XGBoost + LightGBM receptivity model
│   ├── 04_micro_segmentation.py        # Cluster growers into ~200 micro-segments
│   ├── 05_attribution_analysis.py      # POS attribution diagnostics
│   └── 06_run_pipeline.py              # End-to-end pipeline runner
│
├── api/
│   ├── main.py                         # FastAPI server (all engines exposed)
│   └── schemas.py                      # Request / response Pydantic schemas
│
├── models/                             # Trained model artefacts (auto-generated)
│   ├── receptivity_model.pkl
│   ├── embedding_pipeline.pkl
│   ├── hdbscan_clusterer.pkl
│   ├── linucb_bandit.pkl
│   └── ...
│
├── results/                            # Pipeline outputs (auto-generated)
│   ├── message_queue_<date>.jsonl      # Zone 1 output — pre-rendered queue
│   ├── dispatch_log_<date>.jsonl       # Zone 2 output — delivery audit trail
│   ├── feedback_report_<date>.json     # Zone 3 output — conversion + bandit stats
│   └── campaign_plan_<id>_<date>.csv   # Legacy orchestrator output
│
├── frontend/
│   └── dashboard.html                  # Campaign monitoring dashboard
│
├── requirements.txt
└── .env
```

---

## 3. Architecture — Three Zones

The system separates concerns into three zones that each own a distinct phase of the campaign lifecycle. **All AI inference happens in Zone 1. Zone 2 never calls a model.**

```
┌─────────────────────────────────────────────────────────────────────┐
│          ZONE 1 — SERVER SIDE  (always online, runs at 11 PM)       │
│                                                                     │
│  ┌─────────────────┐   ┌──────────────────┐   ┌─────────────────┐  │
│  │   Data store    │──▶│  Nightly batch   │──▶│ Delivery queue  │  │
│  │                 │   │                  │   │                 │  │
│  │ growers.csv     │   │ Signal fetch     │   │ text (rendered) │  │
│  │ POS / inventory │   │ (IMD, ICAR)      │   │ audio blob      │  │
│  │ signal_cache    │   │                  │   │ image URL       │  │
│  │ segment .pkl    │   │ 4 AI engines     │   │ send_at stamp   │  │
│  │ linucb .pkl     │   │ (all content     │   │ channel resolved│  │
│  │ message queue   │   │  generated here) │   │                 │  │
│  └─────────────────┘   └──────────────────┘   └────────┬────────┘  │
└───────────────────────────────────────────────────────┬─┴───────────┘
                                                        │
              ╔═════════════════════════════════════════╪════════════╗
              ║      NETWORK BOUNDARY (GSM / internet)  │            ║
              ║   WhatsApp Business API · Sarvam TTS    │            ║
              ║   IVR · SMS gateway · carrier GSM voice │            ║
              ║   messages leave as pre-rendered text,  │            ║
              ║   audio, or image — no model at send time│           ║
              ╚═════════════════════════════════════════╪════════════╝
                                                        │
┌───────────────────────────────────────────────────────▼────────────┐
│         ZONE 2 — DELIVERY ENGINE  (flush queue at send_at)         │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Channel router — reads device_type from pre-resolved channel│  │
│  │                                                              │  │
│  │  Feature phone  → IVR voice call  (pre-gen Sarvam audio)    │  │
│  │  Smartphone 2G  → WhatsApp text   (~1 KB, pre-rendered)     │  │
│  │  Smartphone 4G  → WhatsApp rich   (text + image, optional   │  │
│  │                                    audio)                   │  │
│  │  No phone       → Field rep brief (rep app push)            │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  delivery receipts / click callbacks ──────────────▶ bandit reward │
└───────────────────────────────────────────────────────┬────────────┘
                                                        │
┌───────────────────────────────────────────────────────▼────────────┐
│         ZONE 3 — FEEDBACK LOOP  (runs nightly before Zone 1)       │
│                                                                     │
│  WA webhook: delivered=0.1 · opened=0.3 · clicked=0.7             │
│  IVR: call-answered / keypress-1                                   │
│  POS: purchase within 14 days = 1.0                                │
│                                                                     │
│  LinUCB updates arm weights  ──▶  better targeting next night      │
└────────────────────────────────────────────────────────────────────┘
```

### Nightly Batch Pipeline (Zone 1 in detail)

```
External signal fetch (11 PM)
  IMD weather · ICAR-NCIPM pest · Agmarknet mandi prices
  └─▶ writes data/signal_cache.json
            │
            ▼
  Grower-360 feature store
  joins all 8 CSVs + signal cache → 1 row per grower
            │
            ▼
  ┌─────────────────── Four AI engines (parallel) ───────────────────┐
  │  E3 Receptivity    E4 Micro-segment   E2 Targeting   E1 Content  │
  │  XGBoost scores    HDBSCAN assigns    LinUCB picks   LLM renders │
  │  0–1, filter       segment + LLM      channel +      text +      │
  │  below 0.03        template/grower    time slot +    audio +     │
  │                                       creative       image       │
  └──────────────────────────────────────────────────────────────────┘
            │
            ▼
  Stock check gate
  suppress if retailer stock < 20% in grower's tehsil
            │
            ▼
  Pre-rendered message queue
  text + audio blob + image + send_at timestamp
  one row per grower, ready to flush
            │
            ▼
  delivery engine at scheduled time  (Zone 2)
```

---

## 4. The Four AI Engines

### Engine 1 — Context-Aware Content Generator (`engines/content_generator.py`)

Generates channel-specific content in 11 Indian languages for every grower segment.

**Generation priority:**

| Priority | Backend | Requirement |
|----------|---------|-------------|
| 1 | Google Gemini | `GEMINI_API_KEY` set |
| 2 | Rule-based fallback | Always works, zero dependencies |

**TTS priority for IVR audio:**

| Priority | Backend | Notes |
|----------|---------|-------|
| 1 | Bhashini | Govt-backed Indic TTS, 11 languages (`BHASHINI_API_KEY`) |
| 2 | Sarvam AI | High-quality cloud fallback (`SARVAM_API_KEY`) |

**Output formats per grower:**

| Format | Channel | Max length |
|--------|---------|-----------|
| WhatsApp rich (image + caption) | Smartphone 4G | 1,024 chars |
| WhatsApp text | Smartphone 2G | 512 chars |
| IVR voice script + TTS audio | Feature phone | ~30 sec |
| SMS | Any keypad | 160 chars |
| Field rep visit brief | Low-receptivity / high-LTV | Full brief |

**ICAR RAG:** An inline knowledge base of ICAR-NCIPM pest advisories (wheat rust, mustard aphid, potato late blight, chickpea pod borer) is injected into every LLM system prompt, grounding content in verified agronomic guidance.

---

### Engine 2 — Targeting & Timing Optimizer (`engines/targeting_optimizer.py`)

Decides the optimal `channel × time_of_day × creative_variant` arm for each grower across **60 arms** (5 channels × 3 time slots × 4 creative variants).

**Algorithms:**

- **LinUCB** (Contextual Multi-Armed Bandit) — for growers with existing WA engagement history. Each arm maintains its own ridge regression model that updates nightly from POS feedback.
- **Thompson Sampling** — for cold-start growers with zero WA history. Explores via Beta distribution sampling and hands off to LinUCB once ≥3 engagement events are recorded.

**Delivery cost per channel:**

| Channel | Cost / message |
|---------|---------------|
| WhatsApp text | ₹0.10 |
| WhatsApp rich (image) | ₹0.30 |
| SMS | ₹0.15 |
| IVR voice | ₹0.80 |
| Field rep visit | ₹50.00 |

**Device constraints:** Smartphones → WhatsApp channels only. Keypad/feature phones → IVR + SMS only. No device on record → field rep or SMS.

---

### Engine 3 — Campaign Receptivity Predictor (`engines/receptivity_predictor.py`)

Predicts `P(purchase within 14 days | grower, message, channel)` before any message is sent.

**Training:** Joins `whatsapp_campaign` ⟕ `growers` ⟕ `retailer_pos` on tehsil + date within a 14-day attribution window. Both XGBoost and LightGBM are trained; the model with the higher cross-validated ROC-AUC is saved as `models/receptivity_model.pkl`.

**23 features across four categories:**

| Category | Features |
|----------|---------|
| Demographic | `grower_age`, `grower_farm_size`, `gender`, `language`, `device_type`, `state` |
| Behavioural | `hist_open_rate`, `hist_click_rate`, `hist_delivery_rate`, `hist_pos_rate`, `cum_messages`, `offline_campaign_attended`, `product_scan` |
| Contextual | `tehsil_stock_rate`, `days_since_rep_visit`, `pest_pressure_index`, `whatsapp_propensity` |
| Temporal | `days_to_harvest`, `season_progress`, `month`, `day_of_week`, `week_of_year`, `crop_enc` |

**Tier routing:**

| Score | Tier | Action |
|-------|------|--------|
| ≥ 0.15 | High | WhatsApp rich, send immediately |
| 0.05–0.15 | Medium | WhatsApp text or IVR |
| < 0.05 + rep assigned | Low → Rep | Route to `field_rep_brief` channel |

---

### Engine 4 — Personalization Scaler (`engines/micro_segmentation.py`)

Clusters 6,000 growers into **~200 micro-segments** for scalable hyper-personalization.

**Approach:**

1. Each grower is encoded into a 64-dim embedding via PCA over one-hot categoricals + scaled numerics + behavioural flags.
2. **HDBSCAN** (`min_cluster_size=30`) clusters growers into ~200 segments vs. the traditional 4–5, enabling far more precise messaging.
3. One LLM prompt template per cluster, then personalized with four grower-level variables: `{name_token}`, `{tehsil}`, `{crop_stage}`, `{nearest_retailer}`.
4. Result: **200 templates → 6,000 personalized message variants** with zero additional human writing.

---

## 5. Data Inputs

| File | Rows | Role |
|------|------|------|
| `growers.csv` | 6,000 | Primary identity, crop calendar JSON, device type, language, tehsil |
| `whatsapp_campaign.csv` | 4,479 | Engagement ground truth — open / click labels for bandit training |
| `retailer_pos.csv` | 235,042 | Ultimate success metric — POS conversion signal for attribution |
| `retailer_inventory_weekly.csv` | 310,544 | Stock-aware messaging — blocks OOS promotions |
| `retailer_visit_log.csv` | 30,000 | Rep visit recency and frequency per tehsil |
| `retailers.csv` | 4,000 | Nearest-retailer injection into personalized messages |
| `reps_territory.csv` | 500 | Field rep routing for low-digital-literacy growers |
| `digital_funnel_weekly.csv` | 104 | Funnel benchmarks (impression → lead → conversion) |

---

## 6. Key Innovations

### Stock-Synchronized Messaging
Never promote a product when `retailer_inventory_weekly.sku_qty == 0` at all nearby retailers. Two-layer gate:

- **Hard block** (`stockout_risk = 1`): all retailers in tehsil are OOS → grower excluded from the entire campaign batch.
- **Soft filter** (`tehsil_stock_rate < 0.20`): fewer than 20% of retailers have stock → grower suppressed.

### Crop-Calendar-Triggered Campaigns
`grower_crop_calendar` JSON is parsed to compute `days_to_next_stage` for every grower. Growers within a **3–7 day pre-stage window** receive a **1.35× receptivity score boost**, surfacing them at the top of the selection list and ensuring messages land when agronomic urgency is highest.

### Digital + Field Rep Hybrid
When the receptivity model predicts low digital conversion probability AND the grower has an assigned rep in `reps_territory`, the system auto-routes to `field_rep_brief` channel — the rep receives a personalized talking-points brief for a face-to-face visit, capturing growers who would not respond to digital messages.

### Pest Surveillance Emergency Dispatch
An ICAR-NCIPM severe-alert flag in any tehsil propagates to all affected growers within **2 hours** (`EMERGENCY_CONTENT_WINDOW_HRS = 2`), bypassing the normal nightly schedule. Content is generated with the pest advisory injected and the queue record is flagged `is_emergency = true`.

### Zero-Compute Delivery
All LLM inference, TTS synthesis, and bandit decisions happen in Zone 1 at night. By the time a message is delivered, the Zone 2 engine simply reads the pre-rendered payload and routes it — no model is called. This makes the delivery path resilient to network latency, API downtime, and cost spikes.

---

## 7. Setup & Installation

**Requirements:** Python 3.10+

```bash
# 1. Clone / unzip the project

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Copy environment template and fill in your keys
cp env.example .env
```

---

## 8. Running the Pipeline

Run scripts in order from the `agri_marketing/` directory.

```bash
# Step 1 — Exploratory data analysis (optional)
python scripts/01_eda.py

# Step 2 — Build Grower-360 feature store (required before training)
python scripts/02_feature_store.py
# → results/grower_feature_store.parquet

# Step 3 — Train receptivity model
python scripts/03_train_receptivity.py
# → models/receptivity_model.pkl  (target AUC ≥ 0.78)

# Step 4 — Cluster growers into ~200 micro-segments
python scripts/04_micro_segmentation.py
# → models/embedding_pipeline.pkl, results/segment_profiles.parquet

# Step 5 — Attribution diagnostics (optional)
python scripts/05_attribution_analysis.py

# Step 6 — Full campaign batch [Runs all 5 scripts, with a sample campaign plan generated] (Optional)
python scripts/06_run_pipeline.py
# → results/campaign_plan_CMP_RABI25_AI_<date>.csv
```

**Quick single-engine demos:**
```bash
python engines/content_generator.py      # Engine 1
python engines/targeting_optimizer.py   # Engine 2
python engines/receptivity_predictor.py # Engine 3
python engines/micro_segmentation.py    # Engine 4
```


---

## 9. Three-Zone Runner (New)

The `zones/` package implements the architecture shown in Section 3 as three cleanly separated, independently runnable modules.

### Zone 1 — Nightly Batch

Runs all intelligence: signal fetch → feature store → 4 engines → stock gate → queue write.

```bash
# Full run (all crops, up to 500 growers)
python -m zones.zone1_nightly_batch

# Filter to a single crop, dry-run (skip queue write)
python -m zones.zone1_nightly_batch --crop wheat --max 200 --dry-run

# With API keys
python -m zones.zone1_nightly_batch \
    --api-key $GEMINI_API_KEY \
    --sarvam-key $SARVAM_API_KEY \
    --imd-key $IMD_API_KEY
```

**Output:** `results/message_queue_<YYYYMMDD>.jsonl` — one self-contained JSON record per grower, with all content pre-rendered and `send_at` pre-computed.

**Cron (11 PM nightly):**
```cron
0 23 * * * cd /opt/agri_marketing && python -m zones.zone1_nightly_batch
```

### Zone 2 — Delivery Engine

Polls the queue, routes due records to the correct channel adapter, and logs receipts. No model calls.

```bash
# Start the daemon (polls every 60 s)
python -m zones.zone2_delivery_engine

# Flush once and exit (for testing / cron)
python -m zones.zone2_delivery_engine --once

# Target a specific queue file
python -m zones.zone2_delivery_engine --queue results/message_queue_20260520.jsonl

# Simulate an engagement receipt (for testing Zone 3)
python -m zones.zone2_delivery_engine \
    --receipt GRW001 \
    --delivered --opened --clicked
```

**Channel routing table:**

| Queue record `channel` | Device type | Adapter | Data cost |
|------------------------|-------------|---------|-----------|
| `whatsapp_rich` | Smartphone 4G | WhatsApp Business API | ~80 KB |
| `whatsapp_text` | Smartphone 2G | WhatsApp Business API | ~1 KB |
| `ivr_voice` | Feature phone | Sarvam IVR / Exotel | 0 (GSM) |
| `sms` | Any | SMS gateway | 0 (GSM) |
| `field_rep_brief` | No device | Rep mobile app push | — |

**Output:** `results/dispatch_log_<YYYYMMDD>.jsonl` — per-message audit trail with gateway IDs and dispatch timestamps.

### Zone 3 — Feedback Loop

Aggregates delivery receipts and POS data, computes rewards, and updates the LinUCB bandit before the next Zone 1 run.

```bash
# Full run (reads last 7 days of queue files automatically)
python -m zones.zone3_feedback_loop

# Custom attribution window
python -m zones.zone3_feedback_loop --window 14

# Target a specific queue file, dry-run (don't save bandit)
python -m zones.zone3_feedback_loop \
    --queue results/message_queue_20260520.jsonl \
    --dry-run
```

**Reward table (LinUCB arm update values):**

| Signal | Source | Reward |
|--------|--------|--------|
| Message delivered | WA webhook / IVR log | 0.1 |
| Message opened / read | WA webhook | 0.3 |
| Link clicked | WA webhook | 0.7 |
| POS purchase within 14 days | Nightly POS join | 1.0 |

**Output:** `results/feedback_report_<YYYYMMDD>.json` — conversion rates, top arms, and channel-level breakdown.

**Cron (10:30 PM — runs before Zone 1 so the bandit is refreshed):**
```cron
30 22 * * * cd /opt/agri_marketing && python -m zones.zone3_feedback_loop
0  23 * * * cd /opt/agri_marketing && python -m zones.zone1_nightly_batch
```

### Full nightly schedule

```
10:30 PM  Zone 3 — Feedback loop (reads yesterday's queue + POS)
              └─▶ updates models/linucb_bandit.pkl

11:00 PM  Zone 1 — Nightly batch (uses the refreshed bandit)
              └─▶ writes results/message_queue_<today>.jsonl

08:00 AM  Zone 2 — Delivery engine flushes morning slot
01:00 PM  Zone 2 — Delivery engine flushes midday slot
07:00 PM  Zone 2 — Delivery engine flushes evening slot
              (daemon mode checks every 60 s — all three slots auto-flush)
```

---

## 10. API Server

All four engines and the orchestrator are exposed as REST endpoints.

```bash
cd agri_marketing
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Open frontend/dashboard.html**
For the KPI dashboard, and to run features.


Interactive docs: `http://localhost:8000/docs`

**Key endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/content/generate` | Engine 1 — generate content for one grower |
| `POST` | `/api/v1/content/batch` | Engine 1 — batch content generation |
| `POST` | `/api/v1/targeting/decide` | Engine 2 — bandit arm selection |
| `POST` | `/api/v1/targeting/feedback` | Engine 2 — inject reward, update bandit |
| `GET`  | `/api/v1/targeting/stats` | Engine 2 — arm-level performance stats |
| `POST` | `/api/v1/receptivity/score` | Engine 3 — score a single grower |
| `POST` | `/api/v1/receptivity/batch` | Engine 3 — batch scoring |
| `POST` | `/api/v1/segment/assign` | Engine 4 — assign micro-segment |
| `GET`  | `/api/v1/segment/profiles` | Engine 4 — list segment profiles |
| `POST` | `/api/v1/campaign/plan` | Full orchestrator batch plan |
| `POST` | `/api/v1/campaign/auto-feedback-pos` | Automated POS feedback loop |
| `POST` | `/api/v1/signals/external` | Fetch IMD / ICAR / Agmarknet signals |
| `POST` | `/api/v1/stock/check` | Tehsil stock availability |
| `GET`  | `/api/v1/stock/oos-alerts` | Out-of-stock alert list |
| `GET`  | `/api/v1/growers/{grower_id}` | Grower profile + engagement history |
| `GET`  | `/health` | Health check |

---

## 11. Configuration & API Keys

All keys are optional. The system degrades gracefully when none are set.

| Environment Variable | Purpose | Fallback behaviour |
|---------------------|---------|-------------------|
| `GEMINI_API_KEY` | Richer LLM content via Google Gemini | Ollama local → rule-based templates |
| `BHASHINI_API_KEY` | Bhashini IVR TTS (`userID\|ulcaApiKey`) | Sarvam AI → pyttsx3 |
| `SARVAM_API_KEY` | High-quality Indic TTS | pyttsx3 offline |
| `STABILITY_API_KEY` | AI-generated crop infographics | SVG placeholder |
| `IMD_API_KEY` | Live weather signals | `signal_cache.json` → defaults |
| `NCIPM_API_KEY` | Live pest pressure index | `signal_cache.json` → 0.0 |
| `AGMARKNET_KEY` | Mandi price data | `signal_cache.json` → crop defaults |

Create a `.env` file in the project root — `python-dotenv` picks it up automatically:

```env
GEMINI_API_KEY=your_key_here
BHASHINI_API_KEY=userID|ulcaApiKey
SARVAM_API_KEY=your_key_here
IMD_API_KEY=your_key_here
NCIPM_API_KEY=your_key_here
```

---

## 12. Expected Outcomes

| Metric | Baseline (mass campaign) | KisanAI | Lift |
|--------|--------------------------|---------|------|
| WhatsApp open rate | ~22% | 45–55% | ~2× |
| Click-through rate | ~3% | 8–12% | ~3× |
| Campaign-to-POS conversion | ~0.8% | 3–5% | ~4× |
| Cost per conversion | ₹180 | ₹55 | −70% |
| Creative variants per campaign | 4 | 800–2,000+ | 500× |
| Human content writing effort | Linear with variants | Near-flat (LLM) | — |

---
