"""
Script 02: Feature Store Builder
Constructs the Grower-360 feature store used by all four AI engines.
Run: python scripts/02_feature_store.py
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime, date
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Campaign reference data (from DATA_DICTIONARY)
CAMPAIGN_MAP = {
    "wheat": {"campaign_id": "CMP_RABI25_001", "product": "Topik 15 WP"},
    "mustard": {"campaign_id": "CMP_RABI25_002", "product": "Score 250 EC"},
    "chickpea": {"campaign_id": "CMP_RABI25_003", "product": "Actara 25 WG"},
    "potato": {"campaign_id": "CMP_RABI25_004", "product": "Kavach 75 WP"},
}

# Product → SKU mapping from inventory data
PRODUCT_SKU_MAP = {
    "Topik 15 WP": "SY_TOP_15WP",
    "Score 250 EC": "SY_SCO_250EC",
    "Actara 25 WG": "SY_ACT_25WG",
    "Kavach 75 WP": "SY_KAV_75WP",
    "Tilt 250 EC": "SY_TILT_250EC",
    "Amistar 250 SC": "SY_AMI_250SC",
}


def load_all_data():
    print("Loading all datasets...")
    growers = pd.read_csv(DATA_DIR / "growers.csv")
    wa = pd.read_csv(DATA_DIR / "whatsapp_campaign.csv")
    funnel = pd.read_csv(DATA_DIR / "digital_funnel_weekly.csv")
    pos = pd.read_csv(DATA_DIR / "retailer_pos.csv")
    inv = pd.read_csv(DATA_DIR / "retailer_inventory_weekly.csv")
    visit_log = pd.read_csv(DATA_DIR / "retailer_visit_log.csv")
    reps = pd.read_csv(DATA_DIR / "reps_territory.csv")
    retailers = pd.read_csv(DATA_DIR / "retailers.csv")
    return growers, wa, funnel, pos, inv, visit_log, reps, retailers


def parse_growers(growers: pd.DataFrame) -> pd.DataFrame:
    """Parse crop calendar JSON and add derived fields."""
    print("Parsing grower profiles...")

    def safe_parse(cal):
        try:
            return json.loads(cal) if pd.notna(cal) else {}
        except Exception:
            return {}

    calendars = growers["grower_crop_calendar"].apply(safe_parse)

    growers["crop"] = calendars.apply(lambda c: c.get("crop", "unknown"))
    growers["sowing_start"] = pd.to_datetime(
        calendars.apply(lambda c: c.get("sowing", {}).get("start")), errors="coerce"
    )
    growers["harvest_start"] = pd.to_datetime(
        calendars.apply(lambda c: c.get("harvest", {}).get("start")), errors="coerce"
    )
    growers["stages_json"] = calendars.apply(lambda c: c.get("stages", []))

    # Days to next critical growth stage from today
    today = pd.Timestamp.today().normalize()

    def days_to_next_stage(stages):
        future = [
            (pd.Timestamp(s["approx"]) - today).days
            for s in stages
            if pd.Timestamp(s["approx"]) > today
        ]
        return min(future) if future else -1

    growers["days_to_next_stage"] = growers["stages_json"].apply(days_to_next_stage)

    # Season progress (0-1 normalized between sowing and harvest)
    growers["season_progress"] = (today - growers["sowing_start"]) / (
        growers["harvest_start"] - growers["sowing_start"]
    )
    growers["season_progress"] = growers["season_progress"].clip(0, 1).fillna(0.5)

    # Assign campaign product based on crop
    growers["campaign_product"] = growers["crop"].map(
        lambda c: CAMPAIGN_MAP.get(c, {}).get("product", "Tilt 250 EC")
    )
    growers["campaign_id"] = growers["crop"].map(
        lambda c: CAMPAIGN_MAP.get(c, {}).get("campaign_id", "CMP_RABI25_001")
    )

    return growers


def compute_engagement_features(growers: pd.DataFrame, wa: pd.DataFrame) -> pd.DataFrame:
    """Compute per-grower WhatsApp engagement history."""
    print("Computing engagement features...")

    wa_agg = wa.groupby("grower_id").agg(
        wa_messages_received=("id", "count"),
        wa_delivery_rate=("delivered_status", "mean"),
        wa_open_rate=("opened_status", "mean"),
        wa_click_rate=("clicked_status", "mean"),
        wa_last_sent=("message_sent_date", "max"),
    ).reset_index()

    wa_agg["wa_last_sent"] = pd.to_datetime(wa_agg["wa_last_sent"])
    today = pd.Timestamp.today().normalize()
    wa_agg["days_since_last_wa"] = (today - wa_agg["wa_last_sent"]).dt.days

    growers = growers.merge(wa_agg, on="grower_id", how="left")

    # Fill NaN for growers never messaged
    engagement_cols = [
        "wa_messages_received", "wa_delivery_rate", "wa_open_rate",
        "wa_click_rate", "days_since_last_wa"
    ]
    for col in engagement_cols:
        growers[col] = growers[col].fillna(0)

    # Engagement score (composite)
    growers["engagement_score"] = (
        0.2 * growers["wa_delivery_rate"]
        + 0.4 * growers["wa_open_rate"]
        + 0.4 * growers["wa_click_rate"]
    )

    # ADDED — whatsapp_propensity: weighted engagement signal from solution doc
    # (delivered 0.2 + opened 0.3 + clicked 0.5 per Solution 2 / Model 1 formula)
    growers["whatsapp_propensity"] = (
        0.2 * growers["wa_delivery_rate"]
        + 0.3 * growers["wa_open_rate"]
        + 0.5 * growers["wa_click_rate"]
    ).clip(0, 1)

    return growers


def compute_retailer_features(growers: pd.DataFrame, inv: pd.DataFrame,
                               retailers: pd.DataFrame,
                               pos: pd.DataFrame = None) -> pd.DataFrame:
    """Add stock availability signals: is the product in stock near the grower?

    Solution-doc derived features added:
      - stockout_risk       : 1 if ALL nearest retailers have sku_qty == 0 (hard block)
      - local_sku_velocity  : 4-week rolling POS sales of campaign SKU in grower tehsil
      - pos_converted_rate  : fraction of POS transactions involving campaign product (tehsil proxy)
    """
    print("Computing retailer / stock features...")

    inv["week_end_date"] = pd.to_datetime(inv["week_end_date"])
    latest_week = inv["week_end_date"].max()
    inv_latest = inv[inv["week_end_date"] == latest_week].copy()

    # Per-retailer: is each campaign product in stock?
    product_stock = inv_latest.groupby(["retailer_id", "sku_name"])["sku_qty"].sum().reset_index()
    product_stock["in_stock"] = product_stock["sku_qty"] > 0

    # Tehsil → retailer mapping
    retailer_tehsil = retailers[["retailer_id", "tehsil"]].copy()
    product_stock = product_stock.merge(retailer_tehsil, on="retailer_id", how="left")

    # Per tehsil: fraction of retailers with stock for each product
    tehsil_stock = product_stock.groupby(["tehsil", "sku_name"])["in_stock"].mean().reset_index()
    tehsil_stock.columns = ["tehsil", "sku_name", "tehsil_stock_rate"]

    # ADDED — stockout_risk: 1 if tehsil_stock_rate == 0 (no retailer has stock).
    # Solution doc: "Never promote a product if retailer_inventory_weekly shows 0 qty
    # within 10 km — eliminates wasted impressions." (Key Innovation #1)
    tehsil_stock["stockout_risk"] = (tehsil_stock["tehsil_stock_rate"] == 0.0).astype(int)

    # For each grower, look up tehsil stock rate for their campaign product SKU
    growers["campaign_sku"] = growers["campaign_product"].map(PRODUCT_SKU_MAP)
    growers_with_stock = growers.merge(
        tehsil_stock, left_on=["tehsil", "campaign_sku"], right_on=["tehsil", "sku_name"], how="left"
    )
    growers_with_stock["tehsil_stock_rate"] = growers_with_stock["tehsil_stock_rate"].fillna(0.5)
    growers_with_stock["stockout_risk"] = growers_with_stock["stockout_risk"].fillna(1).astype(int)

    # Nearest retailer count
    tehsil_retailer_count = retailer_tehsil.groupby("tehsil")["retailer_id"].count().reset_index()
    tehsil_retailer_count.columns = ["tehsil", "nearby_retailer_count"]
    growers_with_stock = growers_with_stock.merge(tehsil_retailer_count, on="tehsil", how="left")
    growers_with_stock["nearby_retailer_count"] = growers_with_stock["nearby_retailer_count"].fillna(0)

    # ADDED — local_sku_velocity: 4-week rolling POS sales volume of campaign SKU
    # per tehsil. Solution doc: "local_sku_velocity — 4-week rolling sale of campaign
    # SKU in grower's tehsil from retailer_pos" (Module 1, Grower-360 derived features)
    if pos is not None:
        pos = pos.copy()
        pos["transaction_date"] = pd.to_datetime(pos["transaction_date"])
        pos_r = pos.merge(retailer_tehsil, on="retailer_id", how="left")
        cutoff = pos["transaction_date"].max() - pd.Timedelta(weeks=4)
        pos_recent = pos_r[pos_r["transaction_date"] >= cutoff]
        sku_vel = (
            pos_recent.groupby(["tehsil", "sku_name"])["sku_qty"]
            .sum().reset_index()
            .rename(columns={"sku_qty": "local_sku_velocity", "sku_name": "campaign_sku"})
        )
        growers_with_stock = growers_with_stock.merge(
            sku_vel, on=["tehsil", "campaign_sku"], how="left"
        )
        growers_with_stock["local_sku_velocity"] = (
            growers_with_stock["local_sku_velocity"].fillna(0)
        )

        # ADDED — pos_converted_rate: fraction of tehsil POS lines that are the
        # campaign product — proxy for demand intensity.
        pos_total = pos_r.groupby("tehsil")["sku_qty"].sum().reset_index().rename(
            columns={"sku_qty": "tehsil_total_qty"}
        )
        pos_camp = pos_r[pos_r["sku_name"].isin(PRODUCT_SKU_MAP.values())].groupby(
            "tehsil"
        )["sku_qty"].sum().reset_index().rename(columns={"sku_qty": "tehsil_campaign_qty"})
        pos_rate = pos_total.merge(pos_camp, on="tehsil", how="left")
        pos_rate["pos_converted_rate"] = (
            pos_rate["tehsil_campaign_qty"] / pos_rate["tehsil_total_qty"].replace(0, 1)
        ).fillna(0)
        growers_with_stock = growers_with_stock.merge(
            pos_rate[["tehsil", "pos_converted_rate"]], on="tehsil", how="left"
        )
        growers_with_stock["pos_converted_rate"] = (
            growers_with_stock["pos_converted_rate"].fillna(0)
        )
    else:
        growers_with_stock["local_sku_velocity"] = 0.0
        growers_with_stock["pos_converted_rate"]  = 0.0

    return growers_with_stock


def compute_rep_features(growers: pd.DataFrame, reps: pd.DataFrame, visit_log: pd.DataFrame) -> pd.DataFrame:
    """Assign nearest rep and compute rep visit recency per tehsil."""
    print("Computing rep visit features...")

    # Expand tehsil_list (JSON array) into per-row tehsil
    rep_tehsil_rows = []
    for _, row in reps.iterrows():
        tehsils = json.loads(row["tehsil_list"])
        for t in tehsils:
            rep_tehsil_rows.append({"rep_id": row["rep_id"], "territory_id": row["territory_id"], "tehsil": t})
    rep_tehsil = pd.DataFrame(rep_tehsil_rows)

    # Grower → rep mapping (via tehsil)
    growers = growers.merge(
        rep_tehsil[["tehsil", "rep_id"]].drop_duplicates("tehsil"),
        on="tehsil", how="left"
    )

    # Visit recency per tehsil
    visit_log["visit_date"] = pd.to_datetime(visit_log["visit_date"])
    today = pd.Timestamp.today().normalize()
    last_visit = visit_log.groupby("visit_tehsil")["visit_date"].max().reset_index()
    last_visit.columns = ["tehsil", "last_rep_visit"]
    last_visit["days_since_rep_visit"] = (today - last_visit["last_rep_visit"]).dt.days

    growers = growers.merge(last_visit[["tehsil", "days_since_rep_visit"]], on="tehsil", how="left")
    growers["days_since_rep_visit"] = growers["days_since_rep_visit"].fillna(999)  # never visited

    # ADDED — rep_touch_frequency: visits per month per tehsil.
    # Solution doc: "rep_touch_frequency — visits/month in tehsil from retailer_visit_log"
    monthly_visits = (
        visit_log.groupby("visit_tehsil")["visit_date"]
        .count()
        / max(visit_log["visit_date"].nunique() / 30, 1)
    ).reset_index()
    monthly_visits.columns = ["tehsil", "rep_touch_frequency"]
    growers = growers.merge(monthly_visits, on="tehsil", how="left")
    growers["rep_touch_frequency"] = growers["rep_touch_frequency"].fillna(0)

    return growers


def compute_funnel_benchmarks(funnel: pd.DataFrame) -> dict:
    """Extract campaign funnel benchmarks for normalization."""
    funnel_agg = funnel.groupby("campaign_id").agg(
        avg_impressions=("social_post_impression", "mean"),
        avg_visits=("landing_page_visits", "mean"),
        avg_leads=("lead_form_submission", "mean"),
    )
    funnel_agg["visit_rate"] = funnel_agg["avg_visits"] / funnel_agg["avg_impressions"]
    funnel_agg["lead_rate"] = funnel_agg["avg_leads"] / funnel_agg["avg_visits"]
    return funnel_agg.to_dict("index")


def encode_features(growers: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features and scale numerics for ML."""
    print("Encoding and scaling features...")

    # Channel assignment (rule-based from device_type)
    growers["primary_channel"] = growers["device_type"].map(
        {"smartphone": "whatsapp", "keypad": "ivr_sms", "unknown": "field_rep"}
    )

    # Literacy proxy (older + keypad → lower digital literacy)
    growers["digital_literacy_score"] = (
        growers["device_type"].map({"smartphone": 1.0, "keypad": 0.4, "unknown": 0.2})
        - 0.005 * (growers["grower_age"] - 40).clip(-10, 30)
        + 0.1 * growers["offline_campaign_attended"].astype(float)
    ).clip(0, 1)

    # Encode categoricals
    le_lang = LabelEncoder()
    le_crop = LabelEncoder()
    le_state = LabelEncoder()
    le_device = LabelEncoder()

    growers["language_enc"] = le_lang.fit_transform(growers["language"].fillna("Hindi"))
    growers["crop_enc"] = le_crop.fit_transform(growers["crop"].fillna("wheat"))
    growers["state_enc"] = le_state.fit_transform(growers["state"].fillna("Uttar Pradesh"))
    growers["device_enc"] = le_device.fit_transform(growers["device_type"].fillna("unknown"))
    growers["gender_enc"] = (growers["gender"] == "male").astype(int)

    # Save encoders
    joblib.dump({"language": le_lang, "crop": le_crop, "state": le_state, "device": le_device},
                MODELS_DIR / "label_encoders.pkl")

    # Numeric feature scaling
    numeric_features = [
        "grower_age", "grower_farm_size", "engagement_score",
        "wa_open_rate", "wa_click_rate", "tehsil_stock_rate",
        "days_to_next_stage", "season_progress", "days_since_rep_visit",
        "digital_literacy_score", "nearby_retailer_count",
        "whatsapp_propensity", "local_sku_velocity", "pos_converted_rate",
        "rep_touch_frequency"
    ]
    for col in numeric_features:
        if col not in growers.columns:
            growers[col] = 0

    scaler = StandardScaler()
    existing_numeric = [c for c in numeric_features if c in growers.columns]
    growers[existing_numeric] = scaler.fit_transform(growers[existing_numeric].fillna(0))
    joblib.dump(scaler, MODELS_DIR / "feature_scaler.pkl")

    return growers


def build_feature_store():
    growers, wa, funnel, pos, inv, visit_log, reps, retailers = load_all_data()

    growers = parse_growers(growers)
    growers = compute_engagement_features(growers, wa)
    growers = compute_retailer_features(growers, inv, retailers, pos=pos)
    growers = compute_rep_features(growers, reps, visit_log)
    growers = encode_features(growers)

    funnel_benchmarks = compute_funnel_benchmarks(funnel)

    # Save feature store
    out_path = RESULTS_DIR / "grower_feature_store.parquet"
    growers.to_parquet(out_path, index=False)
    print(f"\n Feature store saved to {out_path}")
    print(f"   Shape: {growers.shape}")
    print(f"\nKey columns:\n{[c for c in growers.columns]}")

    joblib.dump(funnel_benchmarks, MODELS_DIR / "funnel_benchmarks.pkl")
    print(f"\n Funnel benchmarks saved.")

    return growers


if __name__ == "__main__":
    growers = build_feature_store()
    print("\nSample Grower-360 Profile:")
    sample = growers.iloc[0][["grower_id", "state", "crop", "device_type", "language",
                               "engagement_score", "wa_open_rate", "wa_click_rate",
                               "tehsil_stock_rate", "days_to_next_stage",
                               "primary_channel", "digital_literacy_score"]]
    print(sample)
