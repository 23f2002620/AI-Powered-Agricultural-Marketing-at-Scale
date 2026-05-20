"""
Utility: Stock Checker
Queries retailer_inventory_weekly.csv to determine whether campaign products
are available at retailers near a grower's tehsil.
Used by the orchestrator to suppress sends when stock is unavailable.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

DATA_DIR = Path("data")

# Crop → most commonly promoted product (from campaign data)
CROP_TO_PRODUCT = {
    "wheat": "Tilt 250 EC",
    "mustard": "Score 250 EC",
    "chickpea": "Amistar 250 SC",
    "potato": "Kavach 75 WP",
    "barley": "Tilt 250 EC",
    "lentil": "Amistar 250 SC",
    "safflower": "Score 250 EC",
    "cumin": "Tilt 250 EC",
    "maize": "Amistar 250 SC",
}

# Low-stock threshold: minimum fraction of in-stock retailers in tehsil
LOW_STOCK_THRESHOLD = 0.2


class StockChecker:
    """
    Loads the latest week's retailer inventory and provides per-tehsil
    stock availability lookups for any SKU.

    Caches the data in memory to avoid repeated CSV reads.
    """

    def __init__(self):
        self._tehsil_stock_index: Optional[dict] = None  # (tehsil, sku_name) → stock_rate
        self._loaded_date: Optional[str] = None
        self._load()

    def _load(self):
        """Load retailer inventory and build tehsil-level stock index."""
        inv_path = DATA_DIR / "retailer_inventory_weekly.csv"
        ret_path = DATA_DIR / "retailers.csv"

        if not inv_path.exists() or not ret_path.exists():
            print("Stock data not found. Using default stock rate 0.7 for all tehsils.")
            self._tehsil_stock_index = {}
            return

        print("Loading inventory data for stock checker...")
        inv = pd.read_csv(inv_path)
        retailers = pd.read_csv(ret_path)

        inv["week_end_date"] = pd.to_datetime(inv["week_end_date"])
        latest_week = inv["week_end_date"].max()
        self._loaded_date = str(latest_week.date())

        inv_latest = inv[inv["week_end_date"] == latest_week].copy()
        inv_latest["in_stock"] = (inv_latest["sku_qty"] > 0).astype(int)

        # Join with retailer tehsil
        retailer_tehsil = retailers[["retailer_id", "tehsil"]].drop_duplicates()
        inv_with_tehsil = inv_latest.merge(retailer_tehsil, on="retailer_id", how="left")

        # Aggregate: per (tehsil, sku_name) → fraction of retailers with stock
        tehsil_stock = (
            inv_with_tehsil.groupby(["tehsil", "sku_name"])["in_stock"]
            .mean()
            .reset_index()
        )
        tehsil_stock.columns = ["tehsil", "sku_name", "stock_rate"]

        self._tehsil_stock_index = {
            (row["tehsil"], row["sku_name"]): round(float(row["stock_rate"]), 4)
            for _, row in tehsil_stock.iterrows()
        }

        n_tehsils = tehsil_stock["tehsil"].nunique()
        n_skus = tehsil_stock["sku_name"].nunique()
        print(f"  Stock index built: {n_tehsils} tehsils × {n_skus} SKUs (week: {self._loaded_date})")

    def get_stock_rate(self, tehsil: str, crop_or_product: str) -> float:
        """
        Returns the fraction of retailers in the tehsil that have the product in stock.

        Args:
            tehsil: Tehsil identifier (e.g., "Patiala_T104")
            crop_or_product: Either a crop name ("wheat") or direct product name ("Tilt 250 EC")

        Returns:
            Float between 0.0 (all OOS) and 1.0 (all in stock). Default 0.7 if unknown.
        """
        if not self._tehsil_stock_index:
            return 0.7  # default fallback

        # Resolve crop → product if needed
        product = CROP_TO_PRODUCT.get(crop_or_product, crop_or_product)

        rate = self._tehsil_stock_index.get((tehsil, product))
        if rate is not None:
            return rate

        # Try alternate common products for the crop
        for alt_product in ["Tilt 250 EC", "Amistar 250 SC", "Score 250 EC"]:
            rate = self._tehsil_stock_index.get((tehsil, alt_product))
            if rate is not None:
                return rate

        return 0.7  # default when tehsil not in index

    def is_in_stock(self, tehsil: str, crop_or_product: str,
                    threshold: float = LOW_STOCK_THRESHOLD) -> bool:
        """
        Returns True if stock availability is above threshold.

        Args:
            tehsil: Tehsil identifier
            crop_or_product: Crop name or product name
            threshold: Minimum fraction of retailers to be "in stock"
        """
        return self.get_stock_rate(tehsil, crop_or_product) >= threshold

    def get_best_stocked_product(self, tehsil: str, crop: str) -> dict:
        """
        For a given crop and tehsil, find which product has the best stock availability.
        Useful when multiple products protect against the same threat.

        Returns:
            {"product": "Tilt 250 EC", "stock_rate": 0.85}
        """
        if not self._tehsil_stock_index:
            return {"product": CROP_TO_PRODUCT.get(crop, "Tilt 250 EC"), "stock_rate": 0.7}

        candidates = {
            sku: rate
            for (t, sku), rate in self._tehsil_stock_index.items()
            if t == tehsil
        }

        if not candidates:
            return {"product": CROP_TO_PRODUCT.get(crop, "Tilt 250 EC"), "stock_rate": 0.7}

        best_product = max(candidates, key=candidates.get)
        return {"product": best_product, "stock_rate": candidates[best_product]}

    def tehsil_stock_summary(self, tehsil: str) -> pd.DataFrame:
        """
        Return a DataFrame of all products and their stock rates for a tehsil.
        """
        if not self._tehsil_stock_index:
            return pd.DataFrame(columns=["product", "stock_rate"])

        rows = [
            {"product": sku, "stock_rate": rate}
            for (t, sku), rate in self._tehsil_stock_index.items()
            if t == tehsil
        ]
        df = pd.DataFrame(rows).sort_values("stock_rate", ascending=False)
        return df

    def bulk_check(self, growers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a 'tehsil_stock_rate' column to a growers DataFrame.
        Expects columns: 'tehsil', 'crop'
        """
        growers_df = growers_df.copy()
        growers_df["tehsil_stock_rate"] = growers_df.apply(
            lambda row: self.get_stock_rate(
                str(row.get("tehsil", "")),
                str(row.get("crop", "wheat"))
            ),
            axis=1
        )
        return growers_df

    def oos_tehsils(self, crop: str, threshold: float = LOW_STOCK_THRESHOLD) -> list:
        """
        Return list of tehsils where the crop's product is out of stock.
        Used to redirect spend to in-stock areas.
        """
        if not self._tehsil_stock_index:
            return []

        product = CROP_TO_PRODUCT.get(crop, crop)
        return [
            tehsil for (tehsil, sku), rate in self._tehsil_stock_index.items()
            if sku == product and rate < threshold
        ]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    checker = StockChecker()

    test_cases = [
        ("Patiala_T104", "wheat"),
        ("Patna_T001", "chickpea"),
        ("Hisar_T003", "mustard"),
    ]

    print("\nStock Availability Check:")
    for tehsil, crop in test_cases:
        rate = checker.get_stock_rate(tehsil, crop)
        in_stock = checker.is_in_stock(tehsil, crop)
        best = checker.get_best_stocked_product(tehsil, crop)
        print(f"\n  Tehsil: {tehsil} | Crop: {crop}")
        print(f"    Primary product stock rate: {rate:.1%}")
        print(f"    In stock (>20%): {in_stock}")
        print(f"    Best stocked product: {best['product']} ({best['stock_rate']:.1%})")
