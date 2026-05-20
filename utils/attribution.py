"""
Utility: Attribution Engine
Computes 14-day campaign-to-purchase attribution windows.
Ties WhatsApp message sends to downstream POS transactions at tehsil level.
"""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

DATA_DIR = Path("data")

ATTRIBUTION_WINDOW_DAYS = 14

# Campaign product → POS SKU name mapping
CAMPAIGN_TO_SKU = {
    "Tilt 250 EC": "Tilt 250 EC",
    "Amistar 250 SC": "Amistar 250 SC",
    "Score 250 EC": "Score 250 EC",
    "Kavach 75 WP": "Kavach 75 WP",
    "Topik 15 WP": "Topik 15 WP",
    "Actara 25 WG": "Actara 25 WG",
}


@dataclass
class AttributionResult:
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


class AttributionEngine:
    """
    Computes conversion attribution by joining message sends with
    nearby POS transactions within a rolling time window.
    """

    def __init__(self, window_days: int = ATTRIBUTION_WINDOW_DAYS):
        self.window_days = window_days
        self._pos_index: Optional[dict] = None      # (tehsil, sku_name, date) → sales
        self._retailer_tehsil: Optional[dict] = None
        self._load_pos_index()

    def _load_pos_index(self):
        """Pre-build a (tehsil, sku_name, date) → (qty, revenue) index for O(1) lookups."""
        pos_path = DATA_DIR / "retailer_pos.csv"
        ret_path = DATA_DIR / "retailers.csv"

        if not pos_path.exists():
            print("POS data not found. Attribution will use zero-sales fallback.")
            self._pos_index = {}
            return

        print("Building POS attribution index...")
        pos = pd.read_csv(pos_path)
        retailers = pd.read_csv(ret_path)

        pos["transaction_date"] = pd.to_datetime(pos["transaction_date"])
        retailer_tehsil = retailers[["retailer_id", "tehsil"]].drop_duplicates()
        pos = pos.merge(retailer_tehsil, on="retailer_id", how="left")
        pos["revenue"] = pos["sku_qty"] * pos["sku_price"]

        # Aggregate by tehsil + sku_name + date
        daily = (
            pos.groupby(["tehsil", "sku_name", "transaction_date"])
            .agg(qty=("sku_qty", "sum"), revenue=("revenue", "sum"), count=("transaction_id", "count"))
            .reset_index()
        )

        self._pos_index = {}
        for _, row in daily.iterrows():
            key = (str(row["tehsil"]), str(row["sku_name"]), row["transaction_date"].date())
            self._pos_index[key] = {
                "qty": float(row["qty"]),
                "revenue": float(row["revenue"]),
                "count": int(row["count"]),
            }

        print(f"  POS index: {len(self._pos_index):,} (tehsil, sku, date) records")

    def attribute(
        self,
        grower_id: str,
        message_id: str,
        campaign_product: str,
        tehsil: str,
        sent_date: datetime,
    ) -> AttributionResult:
        """
        Attribute POS sales to a specific message send.
        Searches for sales in the same tehsil within the attribution window.
        """
        sku_name = CAMPAIGN_TO_SKU.get(campaign_product, campaign_product)
        window_end = sent_date + timedelta(days=self.window_days)

        total_qty = 0.0
        total_revenue = 0.0
        total_count = 0
        days_to_first: Optional[int] = None

        if self._pos_index:
            current_date = sent_date.date()
            end_date = window_end.date()

            while current_date <= end_date:
                key = (tehsil, sku_name, current_date)
                if key in self._pos_index:
                    sale = self._pos_index[key]
                    total_qty += sale["qty"]
                    total_revenue += sale["revenue"]
                    total_count += sale["count"]
                    if days_to_first is None:
                        days_to_first = (current_date - sent_date.date()).days
                current_date += timedelta(days=1)

        return AttributionResult(
            grower_id=grower_id,
            message_id=message_id,
            campaign_product=campaign_product,
            tehsil=tehsil,
            sent_date=sent_date.isoformat(),
            window_end=window_end.isoformat(),
            attributed_sales_count=total_count,
            attributed_qty=total_qty,
            attributed_revenue=total_revenue,
            converted=total_count > 0,
            days_to_first_sale=days_to_first,
        )

    def batch_attribute(self, campaign_results: pd.DataFrame,
                         grower_tehsil: pd.DataFrame) -> pd.DataFrame:
        """
        Attribute a batch of campaign results.

        Args:
            campaign_results: DataFrame with columns [grower_id, message_id,
                              campaign_product, send_scheduled_at]
            grower_tehsil: DataFrame mapping grower_id → tehsil

        Returns:
            campaign_results with attribution columns appended
        """
        merged = campaign_results.merge(grower_tehsil[["grower_id", "tehsil"]],
                                         on="grower_id", how="left")
        merged["tehsil"] = merged["tehsil"].fillna("unknown")
        merged["send_scheduled_at"] = pd.to_datetime(merged["send_scheduled_at"])

        results = []
        for _, row in merged.iterrows():
            attr = self.attribute(
                grower_id=str(row["grower_id"]),
                message_id=str(row.get("message_id", row.get("id", ""))),
                campaign_product=str(row.get("campaign_product", "")),
                tehsil=str(row.get("tehsil", "")),
                sent_date=row["send_scheduled_at"],
            )
            results.append({
                "attributed_sales_count": attr.attributed_sales_count,
                "attributed_qty": attr.attributed_qty,
                "attributed_revenue": attr.attributed_revenue,
                "converted": attr.converted,
                "days_to_first_sale": attr.days_to_first_sale,
            })

        attr_df = pd.DataFrame(results)
        return pd.concat([campaign_results.reset_index(drop=True), attr_df], axis=1)

    def conversion_report(self, attributed: pd.DataFrame) -> dict:
        """
        Generate a summary conversion report from attributed results.
        """
        total = len(attributed)
        if total == 0:
            return {}

        converted = attributed["converted"].sum() if "converted" in attributed.columns else 0

        report = {
            "total_messages": total,
            "total_converted": int(converted),
            "conversion_rate": round(converted / total, 4),
            "total_attributed_qty": float(attributed.get("attributed_qty", pd.Series([0])).sum()),
            "total_attributed_revenue": float(attributed.get("attributed_revenue", pd.Series([0])).sum()),
            "avg_days_to_sale": float(attributed["days_to_first_sale"].dropna().mean())
                                if "days_to_first_sale" in attributed.columns else None,
            "attribution_window_days": self.window_days,
        }

        if "campaign_product" in attributed.columns:
            by_product = attributed.groupby("campaign_product").agg(
                messages=("converted", "count"),
                conversions=("converted", "sum"),
            )
            by_product["conversion_rate"] = by_product["conversions"] / by_product["messages"]
            report["by_product"] = by_product.to_dict("index")

        return report


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    engine = AttributionEngine(window_days=14)

    test_cases = [
        ("GRW_00001", "WAM_RABI25_00001", "Tilt 250 EC", "Bharatpur_T023", datetime(2026, 3, 20)),
        ("GRW_00003", "WAM_RABI25_00003", "Tilt 250 EC", "Patiala_T104", datetime(2026, 3, 3)),
    ]

    print("\nAttribution Results:")
    for grower_id, msg_id, product, tehsil, sent_date in test_cases:
        result = engine.attribute(grower_id, msg_id, product, tehsil, sent_date)
        print(f"\n  Grower: {result.grower_id} | Product: {result.campaign_product}")
        print(f"  Window: {result.sent_date[:10]} → {result.window_end[:10]}")
        print(f"  Sales count: {result.attributed_sales_count}")
        print(f"  Qty sold: {result.attributed_qty:.0f} | Revenue: ₹{result.attributed_revenue:,.0f}")
        print(f"  Converted: {result.converted}")
        if result.days_to_first_sale is not None:
            print(f"  Days to first sale: {result.days_to_first_sale}")
