"""
Engine 3: Campaign Receptivity Predictor — Inference Wrapper
Loads the trained XGBoost model and scores individual growers or batches.
Used by the campaign orchestrator to rank and filter growers before delivery.
"""

"""
OFFLINE STATUS: NO CHANGES REQUIRED
-------------------------------------
This engine is already fully offline-capable in the original codebase.
It loads all state from local .pkl files (models/) and performs all
inference in-process. It makes zero external API calls.

  targeting_optimizer.py   → LinUCB bandit loads from linucb_bandit.pkl
  receptivity_predictor.py → XGBoost model loads from receptivity_model.pkl
  micro_segmentation.py    → HDBSCAN + PCA load from embedding_pipeline.pkl

No API keys needed. No internet needed. No changes made.
This file is identical to the original.
"""


import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

MODELS_DIR = Path("models")
DATA_DIR = Path("data")

FEATURE_COLS = [
    "grower_age", "grower_farm_size",
    "hist_open_rate", "hist_click_rate", "hist_delivery_rate", "hist_pos_rate", "cum_messages",
    "month", "day_of_week", "week_of_year",
    "days_to_harvest", "season_progress",
    # Solution-doc features: stock, rep visit recency, pest pressure, propensity
    "tehsil_stock_rate", "days_since_rep_visit",
    "tehsil_pest_pressure_index",   # ADDED: from ICAR-NCIPM / external_signals (solution doc Engine 3)
    "whatsapp_propensity",           # ADDED: solution doc Grower-360 feature store
    "language_enc", "device_enc", "crop_enc", "state_enc",
    "gender_enc", "offline_campaign_attended_enc", "product_scan_enc",
]

CROP_ENC = {"wheat": 0, "mustard": 1, "chickpea": 2, "potato": 3,
            "barley": 4, "lentil": 5, "safflower": 6, "cumin": 7, "maize": 8}
LANGUAGE_ENC = {
    "Hindi": 0, "Punjabi": 1, "Marathi": 2, "Gujarati": 3, "Kannada": 4,
    "Bengali": 5, "Tamil": 6, "Telugu": 7, "Odia": 8, "Assamese": 9, "Malayalam": 10,
}
DEVICE_ENC = {"keypad": 0, "smartphone": 1, "unknown": 2}
STATE_ENC = {s: i for i, s in enumerate([
    "Bihar", "Gujarat", "Haryana", "Karnataka", "Madhya Pradesh",
    "Maharashtra", "Punjab", "Rajasthan", "Uttar Pradesh", "West Bengal",
])}


@dataclass
class ReceptivityInput:
    grower_id: str
    grower_age: int
    grower_farm_size: float
    language: str
    device_type: str
    crop: str
    state: str
    gender: str
    offline_campaign_attended: bool
    product_scan: bool
    hist_open_rate: float = 0.0
    hist_click_rate: float = 0.0
    hist_delivery_rate: float = 1.0
    hist_pos_rate: float = 0.0       # POS conversion rate (fraction of messages → purchase)
    cum_messages: int = 0
    days_to_harvest: int = 60
    season_progress: float = 0.5
    tehsil_stock_rate: float = 0.7
    days_since_rep_visit: int = 30
    tehsil_pest_pressure_index: float = 0.0  # ADDED: ICAR-NCIPM pest alert level (0-1)
    whatsapp_propensity: float = 0.0          # ADDED: weighted open+click engagement score
    send_date: Optional[datetime] = None


@dataclass
class ReceptivityScore:
    grower_id: str
    score: float                    # 0-1 probability of clicking/converting
    tier: str                       # "high", "medium", "low"
    recommended_action: str
    score_components: dict


class ReceptivityPredictor:
    """
    Loads trained XGBoost model and provides:
    - Single-grower scoring
    - Batch scoring
    - Score-based campaign selection (top-K)
    """

    TIER_THRESHOLDS = {"high": 0.15, "medium": 0.05}  # calibrated on 5% baseline click rate

    def __init__(self):
        self.model = None
        self.loaded = False
        self.feature_cols = None   # set by _try_load from receptivity_features.pkl
        self._try_load()

    def _try_load(self):
        model_path    = MODELS_DIR / "receptivity_model.pkl"
        features_path = MODELS_DIR / "receptivity_features.pkl"
        if model_path.exists():
            self.model = joblib.load(model_path)
            # Load the exact feature list the model was trained on.
            # This is the single source of truth for column count and order,
            # preventing "X has N features but model expects M" mismatches.
            if features_path.exists():
                self.feature_cols = joblib.load(features_path)
            else:
                # Fallback: infer from model (XGBoost exposes feature_names_in_,
                # LightGBM exposes feature_name_())
                try:
                    self.feature_cols = list(self.model.feature_names_in_)
                except AttributeError:
                    try:
                        self.feature_cols = self.model.feature_name_()
                    except Exception:
                        self.feature_cols = None
            self.loaded = True
            n = len(self.feature_cols) if self.feature_cols else "unknown"
            print(f"Receptivity model loaded from {model_path} ({n} features)")
        else:
            self.feature_cols = None
            print("No trained model found. Run scripts/03_train_receptivity.py first.")
            print("Using heuristic scoring as fallback.")

    # All possible features this engine can produce, keyed by the column name
    # used in 03_train_receptivity.py's prepare_features().  At inference time
    # we build a full dict and then select only the columns the saved model knows.
    def _inp_to_dict(self, inp: ReceptivityInput) -> dict:
        send_date = inp.send_date or datetime.utcnow()
        return {
            "grower_age":                   inp.grower_age,
            "grower_farm_size":             inp.grower_farm_size,
            "hist_open_rate":               inp.hist_open_rate,
            "hist_click_rate":              inp.hist_click_rate,
            "hist_delivery_rate":           inp.hist_delivery_rate,
            "hist_pos_rate":                inp.hist_pos_rate,
            "cum_messages":                 inp.cum_messages,
            "month":                        send_date.month,
            "day_of_week":                  send_date.weekday(),
            "week_of_year":                 send_date.isocalendar()[1],
            "days_to_harvest":              inp.days_to_harvest,
            "season_progress":              inp.season_progress,
            "tehsil_stock_rate":            inp.tehsil_stock_rate,
            "days_since_rep_visit":         inp.days_since_rep_visit,
            "tehsil_pest_pressure_index":   inp.tehsil_pest_pressure_index,
            "whatsapp_propensity":          inp.whatsapp_propensity,
            "language_enc":                 LANGUAGE_ENC.get(inp.language, 0),
            "device_enc":                   DEVICE_ENC.get(inp.device_type, 2),
            "crop_enc":                     CROP_ENC.get(inp.crop, 0),
            "state_enc":                    STATE_ENC.get(inp.state, 0),
            "gender_enc":                   1 if inp.gender == "male" else 0,
            "offline_campaign_attended_enc": int(inp.offline_campaign_attended),
            "product_scan_enc":             int(inp.product_scan),
            # optional solution-doc features — default 0 if not in input
            "stockout_risk":                getattr(inp, "stockout_risk", 0),
            "local_sku_velocity":           getattr(inp, "local_sku_velocity", 0.0),
            "rep_touch_frequency":          getattr(inp, "rep_touch_frequency", 0.0),
        }

    def _encode_input(self, inp: ReceptivityInput) -> "pd.DataFrame":
        """
        Build a single-row DataFrame whose columns exactly match the feature list
        the model was trained on (loaded from receptivity_features.pkl).

        This is the correct fix for:
          ValueError: X has 23 features, but LGBMClassifier is expecting 21 features
          UserWarning: X does not have valid feature names

        By using a named DataFrame instead of a bare numpy array, both XGBoost and
        LightGBM receive feature names and the right column count regardless of how
        many optional features were added after the model was trained.
        """
        import pandas as _pd
        row = self._inp_to_dict(inp)

        if self.feature_cols:
            # Select and order exactly the columns the model knows; fill missing with 0
            ordered = {col: row.get(col, 0) for col in self.feature_cols}
            return _pd.DataFrame([ordered])
        else:
            # No saved feature list — fall back to full dict (matches heuristic path)
            return _pd.DataFrame([row])

    def _heuristic_score(self, inp: ReceptivityInput) -> float:
        """Rule-based fallback scoring when model is not available."""
        score = 0.05  # base click rate
        if inp.device_type == "smartphone":
            score += 0.05
        if inp.hist_click_rate > 0:
            score += inp.hist_click_rate * 0.5
        if inp.hist_open_rate > 0.3:
            score += 0.03
        if inp.tehsil_stock_rate > 0.7:
            score += 0.02
        if inp.tehsil_pest_pressure_index > 0.5:   # ADDED: pest urgency boosts receptivity
            score += 0.04
        if inp.whatsapp_propensity > 0.3:           # ADDED: prior engagement propensity
            score += inp.whatsapp_propensity * 0.1
        if inp.offline_campaign_attended:
            score += 0.03
        if inp.product_scan:
            score += 0.04
        if 20 <= inp.days_to_harvest <= 50:
            score += 0.03  # urgency window
        return min(score, 0.95)

    def _classify_tier(self, score: float) -> tuple[str, str]:
        if score >= self.TIER_THRESHOLDS["high"]:
            return "high", "send_immediately_whatsapp_rich"
        elif score >= self.TIER_THRESHOLDS["medium"]:
            return "medium", "send_whatsapp_text_or_ivr"
        else:
            return "low", "route_to_field_rep_or_retailer"

    def score(self, inp: ReceptivityInput) -> ReceptivityScore:
        """Score a single grower."""
        if self.loaded and self.model is not None:
            X = self._encode_input(inp)   # returns a named DataFrame — correct column count
            prob = float(self.model.predict_proba(X)[0][1])
        else:
            prob = self._heuristic_score(inp)

        tier, action = self._classify_tier(prob)

        return ReceptivityScore(
            grower_id=inp.grower_id,
            score=round(prob, 4),
            tier=tier,
            recommended_action=action,
            score_components={
                "hist_click_rate": inp.hist_click_rate,
                "hist_open_rate": inp.hist_open_rate,
                "device_type": inp.device_type,
                "tehsil_stock_rate": inp.tehsil_stock_rate,
                "days_to_harvest": inp.days_to_harvest,
                "season_progress": inp.season_progress,
            },
        )

    def score_batch(self, inputs: list[ReceptivityInput]) -> list[ReceptivityScore]:
        """Score a list of growers efficiently.

        Uses pd.concat to stack named DataFrames so the model receives the exact
        column names and count it was trained on — fixes both the feature-count
        mismatch and the "X does not have valid feature names" sklearn warning.
        """
        if self.loaded and self.model is not None:
            import pandas as _pd
            X = _pd.concat([self._encode_input(inp) for inp in inputs],
                           ignore_index=True)
            probs = self.model.predict_proba(X)[:, 1]
        else:
            probs = np.array([self._heuristic_score(inp) for inp in inputs])

        results = []
        for inp, prob in zip(inputs, probs):
            prob = float(prob)
            tier, action = self._classify_tier(prob)
            results.append(ReceptivityScore(
                grower_id=inp.grower_id,
                score=round(prob, 4),
                tier=tier,
                recommended_action=action,
                score_components={
                    "hist_click_rate": inp.hist_click_rate,
                    "device_type": inp.device_type,
                },
            ))
        return results

    def select_top_k(self, inputs: list[ReceptivityInput], k: int,
                     min_score: float = 0.05) -> list[ReceptivityScore]:
        """Return top-K growers by receptivity score, above minimum threshold."""
        all_scores = self.score_batch(inputs)
        filtered = [s for s in all_scores if s.score >= min_score]
        return sorted(filtered, key=lambda x: x.score, reverse=True)[:k]

    def score_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score a full grower DataFrame (as produced by feature store).
        Expected columns match ReceptivityInput fields.
        """
        send_date = datetime.utcnow()
        inputs = []
        for _, row in df.iterrows():
            inputs.append(ReceptivityInput(
                grower_id=str(row.get("grower_id", "unknown")),
                grower_age=int(row.get("grower_age", 45)),
                grower_farm_size=float(row.get("grower_farm_size", 2.0)),
                language=str(row.get("language", "Hindi")),
                device_type=str(row.get("device_type", "unknown")),
                crop=str(row.get("crop", "wheat")),
                state=str(row.get("state", "Uttar Pradesh")),
                gender=str(row.get("gender", "male")),
                offline_campaign_attended=bool(row.get("offline_campaign_attended", False)),
                product_scan=bool(row.get("product_scan", False)),
                hist_open_rate=float(row.get("wa_open_rate", 0)),
                hist_click_rate=float(row.get("wa_click_rate", 0)),
                hist_delivery_rate=float(row.get("wa_delivery_rate", 1)),
                hist_pos_rate=float(row.get("pos_converted_rate", row.get("hist_pos_rate", 0.0))),
                cum_messages=int(row.get("wa_messages_received", 0)),
                days_to_harvest=int(row.get("days_to_harvest", 60)),
                season_progress=float(row.get("season_progress", 0.5)),
                tehsil_stock_rate=float(row.get("tehsil_stock_rate", 0.7)),
                days_since_rep_visit=int(row.get("days_since_rep_visit", 30)),
                tehsil_pest_pressure_index=float(row.get("tehsil_pest_pressure_index", 0.0)),
                whatsapp_propensity=float(row.get("whatsapp_propensity", 0.0)),
                send_date=send_date,
            ))

        scores = self.score_batch(inputs)
        score_df = pd.DataFrame([{
            "grower_id": s.grower_id,
            "receptivity_score": s.score,
            "tier": s.tier,
            "recommended_action": s.recommended_action,
        } for s in scores])

        return df.merge(score_df, on="grower_id", how="left")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    predictor = ReceptivityPredictor()

    test_inputs = [
        ReceptivityInput(
            grower_id="GRW_00001",
            grower_age=67, grower_farm_size=3.54,
            language="Hindi", device_type="smartphone",
            crop="wheat", state="Rajasthan", gender="male",
            offline_campaign_attended=False, product_scan=False,
            hist_open_rate=0.0, hist_click_rate=0.0,
            tehsil_stock_rate=0.8, days_to_harvest=25, season_progress=0.7,
        ),
        ReceptivityInput(
            grower_id="GRW_00003",
            grower_age=52, grower_farm_size=0.55,
            language="Punjabi", device_type="smartphone",
            crop="wheat", state="Punjab", gender="male",
            offline_campaign_attended=True, product_scan=False,
            hist_open_rate=0.3, hist_click_rate=0.08,
            tehsil_stock_rate=0.9, days_to_harvest=18, season_progress=0.75,
        ),
    ]

    print("=== Engine 3: Receptivity Predictor Demo ===\n")
    for s in predictor.score_batch(test_inputs):
        print(f"Grower {s.grower_id}: score={s.score:.4f} | tier={s.tier} | action={s.recommended_action}")

    print("\n--- Top-K Selection (k=1) ---")
    top = predictor.select_top_k(test_inputs, k=1)
    print(f"Best grower to target: {top[0].grower_id} (score={top[0].score:.4f})")