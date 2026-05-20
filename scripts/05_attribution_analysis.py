"""
Script 05: POS Attribution Analysis
Computes campaign-to-action conversion rates using a 14-day attribution window.
Joins WhatsApp messages → retailer POS sales to measure campaign effectiveness.

Run: python scripts/05_attribution_analysis.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ATTRIBUTION_WINDOW_DAYS = 14  # standard marketing attribution window

# Map campaign products to POS SKU names (from dataset exploration)
CAMPAIGN_TO_SKU = {
    "Tilt 250 EC": "Tilt 250 EC",
    "Amistar 250 SC": "Amistar 250 SC",
    "Score 250 EC": "Score 250 EC",
    "Kavach 75 WP": "Kavach 75 WP",
    "Topik 15 WP": "Topik 15 WP",
    "Actara 25 WG": "Actara 25 WG",
}


def load_data():
    print("Loading WhatsApp and POS data...")
    wa = pd.read_csv(DATA_DIR / "whatsapp_campaign.csv")
    pos = pd.read_csv(DATA_DIR / "retailer_pos.csv")
    growers = pd.read_csv(DATA_DIR / "growers.csv")
    retailers = pd.read_csv(DATA_DIR / "retailers.csv")

    wa["message_sent_date"] = pd.to_datetime(wa["message_sent_date"])
    pos["transaction_date"] = pd.to_datetime(pos["transaction_date"])

    return wa, pos, growers, retailers


def compute_attribution(wa: pd.DataFrame, pos: pd.DataFrame,
                         growers: pd.DataFrame, retailers: pd.DataFrame) -> pd.DataFrame:
    """
    Attribution logic:
    1. For each WhatsApp message sent to a grower
    2. Look at POS sales of the same product within ATTRIBUTION_WINDOW_DAYS
    3. At retailers in the same tehsil as the grower
    4. Flag as 'attributed conversion' if sale exists
    
    Note: This is tehsil-level attribution (no grower→retailer purchase linkage in dataset).
    """
    print(f"Computing {ATTRIBUTION_WINDOW_DAYS}-day attribution window...")

    # Grower tehsil
    grower_tehsil = growers[["grower_id", "tehsil"]].copy()

    # Retailer tehsil
    retailer_tehsil = retailers[["retailer_id", "tehsil"]].copy()

    # POS with retailer tehsil
    pos_with_tehsil = pos.merge(retailer_tehsil, on="retailer_id", how="left")

    # WhatsApp with grower tehsil
    wa_with_tehsil = wa.merge(grower_tehsil, on="grower_id", how="left")

    # For each message: check if product sold in tehsil within window
    attributed = []
    for _, msg in wa_with_tehsil.iterrows():
        tehsil = msg["tehsil"]
        product = msg["campaign_product"]
        sent_date = msg["message_sent_date"]
        window_end = sent_date + timedelta(days=ATTRIBUTION_WINDOW_DAYS)

        sku_name = CAMPAIGN_TO_SKU.get(product, product)

        # Sales of same product in same tehsil within attribution window
        matching_sales = pos_with_tehsil[
            (pos_with_tehsil["tehsil"] == tehsil)
            & (pos_with_tehsil["sku_name"] == sku_name)
            & (pos_with_tehsil["transaction_date"] >= sent_date)
            & (pos_with_tehsil["transaction_date"] <= window_end)
        ]

        attributed.append({
            "message_id": msg["id"],
            "grower_id": msg["grower_id"],
            "campaign_product": product,
            "campaign_crop": msg["campaign_crop"],
            "tehsil": tehsil,
            "message_sent_date": sent_date,
            "delivered": msg["delivered_status"],
            "opened": msg["opened_status"],
            "clicked": msg["clicked_status"],
            "attributed_sales_count": len(matching_sales),
            "attributed_qty_sold": matching_sales["sku_qty"].sum() if len(matching_sales) > 0 else 0,
            "attributed_revenue": (matching_sales["sku_qty"] * matching_sales["sku_price"]).sum()
                                   if len(matching_sales) > 0 else 0,
            "converted": len(matching_sales) > 0,
        })

    attr_df = pd.DataFrame(attributed)
    return attr_df


def analyze_attribution(attr_df: pd.DataFrame):
    """Print conversion funnel and breakdowns."""
    total = len(attr_df)
    delivered = attr_df["delivered"].sum()
    opened = attr_df["opened"].sum()
    clicked = attr_df["clicked"].sum()
    converted = attr_df["converted"].sum()

    print("\n" + "=" * 60)
    print("CAMPAIGN ATTRIBUTION FUNNEL (14-day window)")
    print("=" * 60)
    print(f"\n  Messages Sent:       {total:,}  (100.0%)")
    print(f"  Delivered:           {delivered:,}  ({delivered/total:.1%})")
    print(f"  Opened:              {opened:,}  ({opened/total:.1%})")
    print(f"  Clicked:             {clicked:,}  ({clicked/total:.1%})")
    print(f"  Attributed Conv.:    {converted:,}  ({converted/total:.1%})")
    print(f"\n  Total Attributed Revenue: ₹{attr_df['attributed_revenue'].sum():,.0f}")
    print(f"  Avg Revenue/Message:      ₹{attr_df['attributed_revenue'].mean():,.1f}")

    print("\n--- Conversion by Campaign Product ---")
    by_product = attr_df.groupby("campaign_product").agg(
        messages=("message_id", "count"),
        delivered_rate=("delivered", "mean"),
        open_rate=("opened", "mean"),
        click_rate=("clicked", "mean"),
        conversion_rate=("converted", "mean"),
        total_attributed_qty=("attributed_qty_sold", "sum"),
        total_attributed_rev=("attributed_revenue", "sum"),
    ).round(4)
    print(by_product.to_string())

    print("\n--- Conversion by Crop ---")
    by_crop = attr_df.groupby("campaign_crop").agg(
        messages=("message_id", "count"),
        conversion_rate=("converted", "mean"),
    ).round(4)
    print(by_crop)

    print("\n--- Click-to-Convert Rate (among clicked messages) ---")
    clicked_df = attr_df[attr_df["clicked"] == True]
    if len(clicked_df) > 0:
        c2c = clicked_df["converted"].mean()
        print(f"  {c2c:.1%} of clicked messages led to attributed sales")

    return attr_df


def compute_incrementality(attr_df: pd.DataFrame, pos: pd.DataFrame, retailers: pd.DataFrame):
    """
    Estimate incremental lift:
    Compare sales in tehsils that received messages vs those that didn't.
    Simple pre-post / treated-control proxy (not a true RCT).
    """
    print("\n--- Incrementality Proxy Analysis ---")

    pos["transaction_date"] = pd.to_datetime(pos["transaction_date"])
    retailer_tehsil = retailers[["retailer_id", "tehsil"]].drop_duplicates()
    pos_t = pos.merge(retailer_tehsil, on="retailer_id", how="left")

    treated_tehsils = attr_df["tehsil"].dropna().unique()
    campaign_start = attr_df["message_sent_date"].min()
    campaign_end = attr_df["message_sent_date"].max()

    # Sales during campaign period
    campaign_sales = pos_t[
        (pos_t["transaction_date"] >= campaign_start)
        & (pos_t["transaction_date"] <= campaign_end)
    ].copy()

    treated = campaign_sales[campaign_sales["tehsil"].isin(treated_tehsils)]
    control = campaign_sales[~campaign_sales["tehsil"].isin(treated_tehsils)]

    treated_daily_qty = treated["sku_qty"].sum() / max((campaign_end - campaign_start).days, 1)
    control_daily_qty = control["sku_qty"].sum() / max((campaign_end - campaign_start).days, 1)

    print(f"  Treated tehsils:  {len(treated_tehsils)}")
    print(f"  Control tehsils:  {pos_t['tehsil'].nunique() - len(treated_tehsils)}")
    print(f"  Treated daily qty:  {treated_daily_qty:.1f} units/day")
    print(f"  Control daily qty:  {control_daily_qty:.1f} units/day")
    if control_daily_qty > 0:
        lift = (treated_daily_qty - control_daily_qty) / control_daily_qty
        print(f"  Estimated Lift:  {lift:+.1%}")


def main():
    wa, pos, growers, retailers = load_data()

    # For speed: use a vectorized approach on smaller datasets
    # For full 235K POS rows, we use a tehsil-date index
    print("Building tehsil-date-product sales index for fast lookup...")
    retailer_tehsil = retailers[["retailer_id", "tehsil"]].drop_duplicates()
    pos_indexed = pos.merge(retailer_tehsil, on="retailer_id", how="left")
    pos_indexed["month_year"] = pos_indexed["transaction_date"].astype(str).str[:7]

    # Tehsil-product-month aggregation for fast lookup
    tehsil_monthly_sales = pos_indexed.groupby(["tehsil", "sku_name", "month_year"]).agg(
        qty=("sku_qty", "sum"),
        revenue=("sku_price", lambda x: (x * pos_indexed.loc[x.index, "sku_qty"]).sum()),
        transaction_count=("transaction_id", "count"),
    ).reset_index()

    out = RESULTS_DIR / "tehsil_monthly_sales.csv"
    tehsil_monthly_sales.to_csv(out, index=False)
    print(f"Saved tehsil monthly sales index to {out}")

    # Full attribution (may be slow on 4479 messages × large POS — sample for demo)
    sample_wa = wa.sample(min(500, len(wa)), random_state=42)
    attr_df = compute_attribution(sample_wa, pos, growers, retailers)
    attr_df = analyze_attribution(attr_df)
    compute_incrementality(attr_df, pos, retailers)

    # Save attribution results
    out_path = RESULTS_DIR / "attribution_results.csv"
    attr_df.to_csv(out_path, index=False)
    print(f"\n Attribution results saved to {out_path}")

    # Overall campaign performance summary
    summary = {
        "total_messages": len(wa),
        "delivery_rate": wa["delivered_status"].mean(),
        "open_rate": wa["opened_status"].mean(),
        "click_rate": wa["clicked_status"].mean(),
        "attribution_window_days": ATTRIBUTION_WINDOW_DAYS,
        "sample_conversion_rate": attr_df["converted"].mean(),
    }
    print("\n--- Campaign Performance Summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
