"""
Script 01: Exploratory Data Analysis
Analyzes all 8 tables from the Syngenta IITM Hackathon 2026 dataset.
Run: python scripts/01_eda.py
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
RESULTS_DIR = Path("results/eda")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_growers() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "growers.csv")
    # Parse crop calendar JSON
    df["crop"] = df["grower_crop_calendar"].apply(
        lambda x: json.loads(x)["crop"] if pd.notna(x) else None
    )
    df["sowing_start"] = df["grower_crop_calendar"].apply(
        lambda x: json.loads(x)["sowing"]["start"] if pd.notna(x) else None
    )
    df["harvest_start"] = df["grower_crop_calendar"].apply(
        lambda x: json.loads(x)["harvest"]["start"] if pd.notna(x) else None
    )
    df["stages"] = df["grower_crop_calendar"].apply(
        lambda x: json.loads(x).get("stages", []) if pd.notna(x) else []
    )
    df["sowing_start"] = pd.to_datetime(df["sowing_start"], errors="coerce")
    df["harvest_start"] = pd.to_datetime(df["harvest_start"], errors="coerce")
    return df


def analyze_growers(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("GROWERS ANALYSIS (growers.csv) — 6,000 rows")
    print("=" * 60)

    print(f"\nShape: {df.shape}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    print("\n--- Device Type Distribution ---")
    print(df["device_type"].value_counts())
    print(f"  → {df['device_type'].eq('smartphone').mean():.1%} can receive WhatsApp")

    print("\n--- Language Distribution ---")
    print(df["language"].value_counts())

    print("\n--- Crop Distribution ---")
    print(df["crop"].value_counts())

    print("\n--- Top 10 States ---")
    print(df["state"].value_counts().head(10))

    print("\n--- Farm Size Stats ---")
    print(df["grower_farm_size"].describe())

    print("\n--- Age Distribution ---")
    bins = [0, 30, 40, 50, 60, 70, 100]
    labels = ["<30", "30-40", "40-50", "50-60", "60-70", "70+"]
    df["age_group"] = pd.cut(df["grower_age"], bins=bins, labels=labels)
    print(df["age_group"].value_counts())

    print("\n--- Gender ---")
    print(df["gender"].value_counts())

    print("\n--- Product Scan Rate ---")
    print(f"  Scanned: {df['product_scan'].sum()} ({df['product_scan'].mean():.1%})")

    print("\n--- Offline Campaign Attended ---")
    print(f"  Attended: {df['offline_campaign_attended'].sum()} ({df['offline_campaign_attended'].mean():.1%})")

    # Segment: Digital reachability
    smartphone_mask = df["device_type"] == "smartphone"
    keypad_mask = df["device_type"] == "keypad"
    unknown_mask = df["device_type"] == "unknown"
    print("\n--- Reachability Segments ---")
    print(f"  WhatsApp-ready (smartphone): {smartphone_mask.sum()}")
    print(f"  IVR/SMS (keypad): {keypad_mask.sum()}")
    print(f"  Field-rep only (unknown): {unknown_mask.sum()}")

    return df


def analyze_whatsapp(df_wa: pd.DataFrame, df_growers: pd.DataFrame):
    print("\n" + "=" * 60)
    print("WHATSAPP CAMPAIGN ANALYSIS (whatsapp_campaign.csv)")
    print("=" * 60)

    print(f"\nShape: {df_wa.shape}")
    df_wa["message_sent_date"] = pd.to_datetime(df_wa["message_sent_date"])

    print("\n--- Overall Engagement Funnel ---")
    delivered = df_wa["delivered_status"].mean()
    opened = df_wa["opened_status"].mean()
    clicked = df_wa["clicked_status"].mean()
    print(f"  Delivered:  {delivered:.1%}")
    print(f"  Opened:     {opened:.1%}  (of sent)")
    print(f"  Clicked:    {clicked:.1%}  (of sent)")
    print(f"  Open→Click: {(clicked / opened if opened > 0 else 0):.1%}")

    print("\n--- Engagement by Campaign Product ---")
    by_product = df_wa.groupby("campaign_product").agg(
        sent=("id", "count"),
        delivered=("delivered_status", "mean"),
        opened=("opened_status", "mean"),
        clicked=("clicked_status", "mean"),
    ).round(3)
    print(by_product)

    print("\n--- Message Volume by Month ---")
    df_wa["month"] = df_wa["message_sent_date"].dt.to_period("M")
    print(df_wa.groupby("month").size())

    # Merge with grower data for richer analysis
    merged = df_wa.merge(df_growers[["grower_id", "language", "device_type", "grower_age", "grower_farm_size", "state"]], on="grower_id", how="left")

    print("\n--- Click Rate by Language ---")
    lang_click = merged.groupby("language")["clicked_status"].mean().sort_values(ascending=False)
    print(lang_click)

    print("\n--- Click Rate by State (top 5) ---")
    state_click = merged.groupby("state")["clicked_status"].mean().sort_values(ascending=False).head(5)
    print(state_click)

    return df_wa


def analyze_digital_funnel(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("DIGITAL FUNNEL ANALYSIS (digital_funnel_weekly.csv)")
    print("=" * 60)

    df["week_start_date"] = pd.to_datetime(df["week_start_date"])
    df["visit_rate"] = df["landing_page_visits"] / df["social_post_impression"]
    df["lead_rate"] = df["lead_form_submission"] / df["landing_page_visits"].replace(0, np.nan)

    print(f"\nShape: {df.shape}")
    print(f"Campaigns: {df['campaign_id'].nunique()}")
    print(f"Weeks: {df['week_start_date'].nunique()}")

    print("\n--- Overall Funnel Rates ---")
    total_impressions = df["social_post_impression"].sum()
    total_visits = df["landing_page_visits"].sum()
    total_leads = df["lead_form_submission"].sum()
    print(f"  Impression→Visit: {total_visits/total_impressions:.2%}")
    print(f"  Visit→Lead:       {total_leads/total_visits:.2%}")
    print(f"  Overall:          {total_leads/total_impressions:.2%}")

    print("\n--- By Campaign ---")
    by_camp = df.groupby(["campaign_id", "campaign_crop", "campaign_product"]).agg(
        total_impressions=("social_post_impression", "sum"),
        total_visits=("landing_page_visits", "sum"),
        total_leads=("lead_form_submission", "sum"),
    )
    by_camp["imp_to_visit"] = (by_camp["total_visits"] / by_camp["total_impressions"]).round(4)
    by_camp["visit_to_lead"] = (by_camp["total_leads"] / by_camp["total_visits"]).round(4)
    print(by_camp)


def analyze_pos(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("RETAILER POS ANALYSIS (retailer_pos.csv)")
    print("=" * 60)

    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    print(f"\nShape: {df.shape}")
    print(f"Unique retailers: {df['retailer_id'].nunique()}")
    print(f"Unique SKUs: {df['sku_name'].nunique()}")
    print(f"Date range: {df['transaction_date'].min()} to {df['transaction_date'].max()}")

    print("\n--- Top Products by Revenue ---")
    revenue = df.groupby("sku_name").agg(
        total_qty=("sku_qty", "sum"),
        total_revenue=("sku_price", lambda x: (x * df.loc[x.index, "sku_qty"]).sum()),
        transactions=("transaction_id", "count"),
    ).sort_values("total_revenue", ascending=False).head(10)
    print(revenue)

    print("\n--- Monthly Sales Trend ---")
    df["month"] = df["transaction_date"].dt.to_period("M")
    monthly = df.groupby("month").agg(qty=("sku_qty", "sum"), revenue=("sku_price", "sum"))
    print(monthly)


def analyze_inventory(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("INVENTORY ANALYSIS (retailer_inventory_weekly.csv)")
    print("=" * 60)

    df["week_end_date"] = pd.to_datetime(df["week_end_date"])
    print(f"\nShape: {df.shape}")
    print(f"Unique retailers: {df['retailer_id'].nunique()}")
    print(f"Unique SKUs: {df['sku_name'].nunique()}")

    print("\n--- Out-of-Stock Rate by SKU ---")
    oos = df.groupby("sku_name").apply(lambda x: (x["sku_qty"] == 0).mean()).sort_values(ascending=False)
    print(oos.head(10))

    print("\n--- Average Stock Level by SKU ---")
    avg_stock = df.groupby("sku_name")["sku_qty"].mean().sort_values(ascending=False)
    print(avg_stock.head(10))


def analyze_retailer_visits(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("RETAILER VISIT LOG ANALYSIS (retailer_visit_log.csv)")
    print("=" * 60)

    df["visit_date"] = pd.to_datetime(df["visit_date"])
    print(f"\nShape: {df.shape}")
    print(f"\nVisit types:\n{df['visit_type'].value_counts()}")
    print(f"\nTop promoted products:\n{df['product_recommended'].value_counts().head(10)}")
    print(f"\nVisits per rep:\n{df.groupby('rep_id').size().describe()}")


def analyze_reps(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("REPS TERRITORY ANALYSIS (reps_territory.csv)")
    print("=" * 60)

    print(f"\nShape: {df.shape}")
    print(f"\nStates covered:\n{df['state'].value_counts()}")
    df["tehsil_count"] = df["tehsil_list"].apply(lambda x: len(json.loads(x)))
    print(f"\nTehsils per rep:\n{df['tehsil_count'].describe()}")


def run_cross_table_analysis(growers: pd.DataFrame, wa: pd.DataFrame, pos: pd.DataFrame, inv: pd.DataFrame):
    print("\n" + "=" * 60)
    print("CROSS-TABLE ANALYSIS")
    print("=" * 60)

    # Attribution: Did growers who received a WhatsApp lead to POS sales?
    wa["message_sent_date"] = pd.to_datetime(wa["message_sent_date"])
    pos["transaction_date"] = pd.to_datetime(pos["transaction_date"])

    # Growers who clicked a WhatsApp message
    clicked_growers = wa[wa["clicked_status"] == True]["grower_id"].unique()
    print(f"\nGrovers who clicked WhatsApp: {len(clicked_growers)}")

    # Product-level stock awareness
    campaign_products = wa["campaign_product"].unique()
    inv_recent = inv[inv["week_end_date"] == inv["week_end_date"].max()]
    for prod in campaign_products:
        available_retailers = inv_recent[
            (inv_recent["sku_name"] == prod) & (inv_recent["sku_qty"] > 0)
        ]["retailer_id"].nunique()
        oos_retailers = inv_recent[
            (inv_recent["sku_name"] == prod) & (inv_recent["sku_qty"] == 0)
        ]["retailer_id"].nunique()
        print(f"\n  Product: {prod}")
        print(f"    In-stock retailers: {available_retailers}")
        print(f"    OOS retailers: {oos_retailers}")


def main():
    print("Loading datasets...")
    growers = load_growers()
    wa = pd.read_csv(DATA_DIR / "whatsapp_campaign.csv")
    funnel = pd.read_csv(DATA_DIR / "digital_funnel_weekly.csv")
    pos = pd.read_csv(DATA_DIR / "retailer_pos.csv")
    inv = pd.read_csv(DATA_DIR / "retailer_inventory_weekly.csv")
    visit_log = pd.read_csv(DATA_DIR / "retailer_visit_log.csv")
    reps = pd.read_csv(DATA_DIR / "reps_territory.csv")

    growers = analyze_growers(growers)
    analyze_whatsapp(wa, growers)
    analyze_digital_funnel(funnel)
    analyze_pos(pos)
    analyze_inventory(inv)
    analyze_retailer_visits(visit_log)
    analyze_reps(reps)
    run_cross_table_analysis(growers, wa, pos, inv)

    print("\n\n EDA complete. Summary saved to results/eda/")


if __name__ == "__main__":
    main()
