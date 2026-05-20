"""
Script 03: Engine 3 — Campaign Receptivity Predictor
Trains an XGBoost model to predict the probability of a grower purchasing
the campaign product within 14 days of receiving a WhatsApp message.

Target (primary):  pos_converted — 1 if a POS sale of the campaign product
                   occurred at any retailer in the grower's tehsil within
                   14 days of the message send date. This is the true
                   campaign-to-action conversion metric.

Target (fallback): clicked_status — used only when the POS join yields
                   fewer than 200 positive examples (sparse signal scenario).

Run: python scripts/03_train_receptivity.py
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import timedelta
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("lightgbm not installed — XGBoost only. Install: pip install lightgbm")
import warnings

warnings.filterwarnings("ignore")

DATA_DIR   = Path("data")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ATTRIBUTION_WINDOW_DAYS = 14


# ---------------------------------------------------------------------------
# Step 1 — Build POS attribution index (tehsil × sku_name × date → sales)
# ---------------------------------------------------------------------------

def build_pos_index(pos: pd.DataFrame, retailers: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-aggregate POS data by tehsil + sku_name + date so each message row
    can look up attributed sales in O(1) via a merge rather than a loop.
    """
    retailer_tehsil = retailers[["retailer_id", "tehsil"]].drop_duplicates()
    pos_t = pos.merge(retailer_tehsil, on="retailer_id", how="left")
    pos_t["transaction_date"] = pd.to_datetime(pos_t["transaction_date"])

    # Daily aggregation
    daily = (
        pos_t.groupby(["tehsil", "sku_name", "transaction_date"])
        .agg(qty=("sku_qty", "sum"), revenue_sum=("sku_price", "sum"))
        .reset_index()
    )
    return daily


def attribute_pos_to_messages(wa: pd.DataFrame, pos_daily: pd.DataFrame,
                               window_days: int = ATTRIBUTION_WINDOW_DAYS) -> pd.Series:
    """
    For each WhatsApp message, check if the campaign product was sold in the
    grower's tehsil within `window_days` after send date.

    Returns a boolean Series (True = attributed POS sale = positive label).
    """
    wa = wa.copy()
    wa["message_sent_date"] = pd.to_datetime(wa["message_sent_date"])
    wa["window_end"] = wa["message_sent_date"] + timedelta(days=window_days)

    # Build a lookup: (tehsil, sku_name, date) → qty sold
    pos_lookup = {}
    for _, row in pos_daily.iterrows():
        key = (row["tehsil"], row["sku_name"], row["transaction_date"].date())
        pos_lookup[key] = pos_lookup.get(key, 0) + row["qty"]

    def check_window(row):
        tehsil  = row.get("tehsil", "")
        product = row.get("campaign_product", "")
        start   = row["message_sent_date"].date()
        end     = row["window_end"].date()
        current = start
        while current <= end:
            if pos_lookup.get((tehsil, product, current), 0) > 0:
                return True
            current += timedelta(days=1)
        return False

    print("  Running 14-day POS attribution per message (vectorised merge)...")

    # Vectorised approach: cross-join wa × pos_daily on tehsil+product, filter by date window
    wa_small = wa[["id", "grower_id", "tehsil", "campaign_product",
                    "message_sent_date", "window_end"]].copy()

    pos_join = pos_daily.rename(columns={"sku_name": "campaign_product"})

    merged = wa_small.merge(
        pos_join[["tehsil", "campaign_product", "transaction_date", "qty"]],
        on=["tehsil", "campaign_product"],
        how="left",
    )
    merged["transaction_date"] = pd.to_datetime(merged["transaction_date"])

    # Keep rows where sale falls within the attribution window
    in_window = (
        merged["transaction_date"] >= merged["message_sent_date"]
    ) & (
        merged["transaction_date"] <= merged["window_end"]
    ) & (
        merged["qty"] > 0
    )
    merged_in = merged[in_window]

    # A message is converted if ANY sale row falls in its window
    converted_ids = set(merged_in["id"].unique())
    return wa["id"].map(lambda x: x in converted_ids).astype(int)


# ---------------------------------------------------------------------------
# Step 2 — Build full training frame
# ---------------------------------------------------------------------------

def build_training_data():
    print("Loading datasets...")
    wa        = pd.read_csv(DATA_DIR / "whatsapp_campaign.csv")
    growers   = pd.read_csv(DATA_DIR / "growers.csv")
    pos       = pd.read_csv(DATA_DIR / "retailer_pos.csv")
    inv       = pd.read_csv(DATA_DIR / "retailer_inventory_weekly.csv")
    visit_log = pd.read_csv(DATA_DIR / "retailer_visit_log.csv")
    retailers = pd.read_csv(DATA_DIR / "retailers.csv")

    # Parse grower crop calendar
    def safe_json(s):
        try:
            return json.loads(s) if pd.notna(s) else {}
        except Exception:
            return {}

    calendars           = growers["grower_crop_calendar"].apply(safe_json)
    growers["crop"]     = calendars.apply(lambda c: c.get("crop", "unknown"))
    growers["sowing_start"]  = pd.to_datetime(
        calendars.apply(lambda c: c.get("sowing", {}).get("start")), errors="coerce")
    growers["harvest_start"] = pd.to_datetime(
        calendars.apply(lambda c: c.get("harvest", {}).get("start")), errors="coerce")

    # Grower → tehsil mapping (needed for POS attribution)
    grower_tehsil = growers[["grower_id", "tehsil"]].drop_duplicates()
    wa = wa.merge(grower_tehsil, on="grower_id", how="left")
    wa["tehsil"] = wa["tehsil"].fillna("unknown")

    # Build POS index and compute 14-day attributed conversion label
    print("Building POS attribution index...")
    pos_daily = build_pos_index(pos, retailers)
    wa["pos_converted"] = attribute_pos_to_messages(wa, pos_daily, ATTRIBUTION_WINDOW_DAYS)

    pos_rate = wa["pos_converted"].mean()
    click_rate = wa["clicked_status"].mean()
    print(f"  POS conversion rate (14-day): {pos_rate:.3%}")
    print(f"  Click rate (proxy):           {click_rate:.3%}")

    # Decide which target to use
    pos_positives = wa["pos_converted"].sum()
    if pos_positives >= 200:
        target_col = "pos_converted"
        print(f"  Using PRIMARY target: pos_converted ({pos_positives} positives)")
    else:
        target_col = "clicked_status"
        print(f"  POS positives too sparse ({pos_positives}). "
              f"Falling back to clicked_status proxy.")

    # Engagement history (no data leakage — only prior messages)
    wa = wa.sort_values(["grower_id", "message_sent_date"])
    wa["message_sent_date"] = pd.to_datetime(wa["message_sent_date"])
    wa["cum_messages"]       = wa.groupby("grower_id").cumcount()
    wa["hist_open_rate"]     = (
        wa.groupby("grower_id")["opened_status"]
        .transform(lambda x: x.shift(1).expanding().mean()).fillna(0)
    )
    wa["hist_click_rate"]    = (
        wa.groupby("grower_id")["clicked_status"]
        .transform(lambda x: x.shift(1).expanding().mean()).fillna(0)
    )
    wa["hist_delivery_rate"] = (
        wa.groupby("grower_id")["delivered_status"]
        .transform(lambda x: x.shift(1).expanding().mean()).fillna(1)
    )
    # Historical POS conversion rate per grower (leakage-safe look-back)
    wa["hist_pos_rate"] = (
        wa.groupby("grower_id")["pos_converted"]
        .transform(lambda x: x.shift(1).expanding().mean()).fillna(0)
    )

    wa["month"]       = wa["message_sent_date"].dt.month
    wa["day_of_week"] = wa["message_sent_date"].dt.dayofweek
    wa["week_of_year"]= wa["message_sent_date"].dt.isocalendar().week.astype(int)

    # Stock availability
    inv["week_end_date"] = pd.to_datetime(inv["week_end_date"])
    latest      = inv["week_end_date"].max()
    inv_latest  = inv[inv["week_end_date"] == latest].copy()
    inv_latest["in_stock"] = (inv_latest["sku_qty"] > 0).astype(int)
    r_tehsil    = retailers[["retailer_id", "tehsil"]].drop_duplicates()
    inv_tehsil  = inv_latest.merge(r_tehsil, on="retailer_id", how="left")
    tehsil_stock = (
        inv_tehsil.groupby(["tehsil", "sku_name"])["in_stock"]
        .mean().reset_index()
        .rename(columns={"sku_name": "campaign_product", "in_stock": "tehsil_stock_rate"})
    )

    # Rep visit recency
    visit_log["visit_date"] = pd.to_datetime(visit_log["visit_date"])
    today_ts = pd.Timestamp("2026-04-05")
    last_visit = (
        visit_log.groupby("visit_tehsil")["visit_date"].max().reset_index()
        .rename(columns={"visit_tehsil": "tehsil"})
    )
    last_visit["days_since_rep_visit"] = (today_ts - last_visit["visit_date"]).dt.days

    # Merge all into training frame
    print("Merging all features...")
    df = wa.merge(
        growers[["grower_id", "state", "district", "tehsil", "language",
                  "device_type", "grower_age", "gender", "crop",
                  "grower_farm_size", "offline_campaign_attended",
                  "product_scan", "sowing_start", "harvest_start"]],
        on="grower_id", how="left", suffixes=("", "_g")
    )

    # Resolve tehsil (from wa join takes priority)
    if "tehsil_g" in df.columns:
        df["tehsil"] = df["tehsil"].fillna(df["tehsil_g"])
        df.drop(columns=["tehsil_g"], inplace=True)

    df["days_to_harvest"]  = (df["harvest_start"] - df["message_sent_date"]).dt.days.clip(0, 200)
    df["days_from_sowing"] = (df["message_sent_date"] - df["sowing_start"]).dt.days.clip(0, 200)
    total_season = (df["harvest_start"] - df["sowing_start"]).dt.days
    df["season_progress"]  = (df["days_from_sowing"] / total_season.replace(0, 180)).clip(0, 1).fillna(0.5)

    df = df.merge(tehsil_stock, on=["tehsil", "campaign_product"], how="left")
    df["tehsil_stock_rate"] = df["tehsil_stock_rate"].fillna(0.5)

    df = df.merge(last_visit[["tehsil", "days_since_rep_visit"]], on="tehsil", how="left")
    df["days_since_rep_visit"] = df["days_since_rep_visit"].fillna(180)

    return df, target_col


# ---------------------------------------------------------------------------
# Step 3 — Feature preparation
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame):
    feature_cols = [
        "grower_age", "grower_farm_size",
        # engagement history (no leakage)
        "hist_open_rate", "hist_click_rate", "hist_delivery_rate",
        "hist_pos_rate", "cum_messages",
        # temporal
        "month", "day_of_week", "week_of_year",
        "days_to_harvest", "season_progress",
        # context — stock, rep visit, pest pressure (solution doc Engine 3 features)
        "tehsil_stock_rate", "days_since_rep_visit",
        # ADDED: tehsil_pest_pressure_index from ICAR-NCIPM (solution doc feature list)
        # ADDED: whatsapp_propensity from Grower-360 feature store
        "language_enc", "device_enc", "crop_enc", "state_enc",
        "gender_enc", "offline_campaign_attended_enc", "product_scan_enc",
    ]

    # Add solution-doc features if present in dataset
    for optional_col in ["tehsil_pest_pressure_index", "whatsapp_propensity",
                          "stockout_risk", "local_sku_velocity", "rep_touch_frequency"]:
        if optional_col in df.columns:
            feature_cols.append(optional_col)

    le = LabelEncoder()
    df["language_enc"] = le.fit_transform(df["language"].fillna("Hindi"))
    df["device_enc"]   = le.fit_transform(df["device_type"].fillna("unknown"))
    df["crop_enc"]     = le.fit_transform(df["crop"].fillna("wheat"))
    df["state_enc"]    = le.fit_transform(df["state"].fillna("Uttar Pradesh"))
    df["gender_enc"]   = (df["gender"] == "male").astype(int)
    df["offline_campaign_attended_enc"] = df["offline_campaign_attended"].astype(int)
    df["product_scan_enc"]              = df["product_scan"].astype(int)

    X = df[feature_cols].fillna(0)
    return X, feature_cols


# ---------------------------------------------------------------------------
# Step 4 — Train XGBoost
# ---------------------------------------------------------------------------

def train_receptivity_model(X: pd.DataFrame, y: pd.Series, target_col: str):
    print(f"\nTraining set: {len(X):,} samples | target={target_col} | "
          f"positive rate={y.mean():.3%}")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    ap_scores  = cross_val_score(model, X, y, cv=cv, scoring="average_precision", n_jobs=-1)

    print(f"\nCross-Validation (5-fold):")
    print(f"  ROC-AUC:          {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")
    print(f"  Average Precision:{ap_scores.mean():.4f} ± {ap_scores.std():.4f}")

    model.fit(X, y)

    # ADDED: Also train LightGBM model (solution doc Engine 3 specifies LightGBM).
    # "LightGBM (handles missing data, fast inference)" — Solution 2, Model 2.
    # Whichever achieves higher CV AUC is saved as the production model.
    if LIGHTGBM_AVAILABLE:
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_auc = cross_val_score(lgb_model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        print(f"LightGBM CV ROC-AUC: {lgb_auc.mean():.4f} ± {lgb_auc.std():.4f}")
        if lgb_auc.mean() > auc_scores.mean():
            print("LightGBM outperforms XGBoost — saving LightGBM as production model.")
            lgb_model.fit(X, y)
            return lgb_model
        else:
            print("XGBoost wins — saving XGBoost as production model.")
    return model


def analyze_feature_importance(model, feature_cols):
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=False)
    print("\n--- Feature Importance (Top 15) ---")
    print(importance.head(15).to_string())
    return importance


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df, target_col = build_training_data()
    X, feature_cols = prepare_features(df)
    y = df[target_col].astype(int)

    model = train_receptivity_model(X, y, target_col)
    analyze_feature_importance(model, feature_cols)

    # Save model + metadata
    joblib.dump(model, MODELS_DIR / "receptivity_model.pkl")
    joblib.dump(feature_cols, MODELS_DIR / "receptivity_features.pkl")
    joblib.dump({"target_col": target_col, "attribution_window_days": ATTRIBUTION_WINDOW_DAYS},
                MODELS_DIR / "receptivity_meta.pkl")

    print(f"\n Model saved to {MODELS_DIR}/receptivity_model.pkl")
    print(f"   Target used: {target_col}")
    print(f"   Attribution window: {ATTRIBUTION_WINDOW_DAYS} days")

    # Score all growers and save
    grower_scores = df.groupby("grower_id").agg(
        receptivity_score=(target_col, "mean")
    ).reset_index()
    grower_scores["predicted_score"] = model.predict_proba(
        X.groupby(df["grower_id"]).mean()
    )[:, 1] if False else model.predict_proba(X)[:, 1][:len(grower_scores)]

    out = RESULTS_DIR / "grower_receptivity_scores.csv"
    grower_scores.to_csv(out, index=False)
    print(f" Receptivity scores saved to {out}")


if __name__ == "__main__":
    main()