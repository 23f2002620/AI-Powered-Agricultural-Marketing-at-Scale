"""
Engine 4: Micro-Segmentation — Inference Wrapper
Assigns new/existing growers to micro-segments and returns
the associated content template and channel strategy.
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

MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")

LANGUAGE_META = {
    "Hindi": {"tts_code": "hi-IN", "iso": "hi"},
    "Punjabi": {"tts_code": "pa-IN", "iso": "pa"},
    "Marathi": {"tts_code": "mr-IN", "iso": "mr"},
    "Gujarati": {"tts_code": "gu-IN", "iso": "gu"},
    "Kannada": {"tts_code": "kn-IN", "iso": "kn"},
    "Bengali": {"tts_code": "bn-IN", "iso": "bn"},
}

CROP_PRODUCT_MAP = {
    "wheat": "Topik 15 WP",
    "mustard": "Score 250 EC",
    "chickpea": "Actara 25 WG",
    "potato": "Kavach 75 WP",
    "barley": "Tilt 250 EC",
    "lentil": "Amistar 250 SC",
    "safflower": "Score 250 EC",
    "cumin": "Tilt 250 EC",
    "maize": "Actara 25 WG",
}


@dataclass
class SegmentAssignment:
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


class MicroSegmentationEngine:
    """
    Assigns a grower to the nearest micro-segment using saved clustering artifacts,
    and retrieves the associated content template.
    """

    def __init__(self):
        self.embedding_pipeline = None
        self.clusterer = None
        self.seg_profiles = None
        self.seg_templates = None
        self.grower_segments = None
        self._try_load()

    def _try_load(self):
        # Load embedding pipeline
        ep_path = MODELS_DIR / "embedding_pipeline.pkl"
        if ep_path.exists():
            self.embedding_pipeline = joblib.load(ep_path)

        # Load clusterer (HDBSCAN or KMeans)
        for name in ["hdbscan_clusterer.pkl", "kmeans_clusterer.pkl"]:
            path = MODELS_DIR / name
            if path.exists():
                self.clusterer = joblib.load(path)
                self.clusterer_type = "hdbscan" if "hdbscan" in name else "kmeans"
                break

        # Load segment profiles and templates
        sp_path = RESULTS_DIR / "segment_profiles.parquet"
        if sp_path.exists():
            self.seg_profiles = pd.read_parquet(sp_path)
            self.seg_profiles["content_brief"] = self.seg_profiles["content_brief"].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )

        st_path = RESULTS_DIR / "segment_llm_templates.csv"
        if st_path.exists():
            self.seg_templates = pd.read_csv(st_path)

        gs_path = RESULTS_DIR / "grower_segments.csv"
        if gs_path.exists():
            self.grower_segments = pd.read_csv(gs_path).set_index("grower_id")

        loaded = all([
            self.embedding_pipeline is not None,
            self.clusterer is not None,
            self.seg_profiles is not None,
        ])
        if loaded:
            print(f"Micro-segmentation engine loaded: {len(self.seg_profiles)} segments")
        else:
            print("Micro-segmentation artifacts not found. Run scripts/04_micro_segmentation.py first.")
            print("Using rule-based segment assignment as fallback.")

    def _lookup_existing(self, grower_id: str) -> Optional[str]:
        """Check if grower already has a segment assignment."""
        if self.grower_segments is not None and grower_id in self.grower_segments.index:
            return self.grower_segments.loc[grower_id, "segment_id"]
        return None

    def _rule_based_assign(self, crop: str, device_type: str, language: str) -> str:
        """Fallback: assign segment based on crop+device+language key."""
        channel = (
            "whatsapp_rich" if device_type == "smartphone"
            else "ivr_voice" if device_type == "keypad"
            else "field_rep_brief"
        )
        return f"RULE_{crop[:3].upper()}_{channel[:3].upper()}_{language[:3].upper()}"

    def assign(
        self,
        grower_id: str,
        crop: str,
        device_type: str,
        language: str,
        state: str,
        grower_age: int = 45,
        farm_size: float = 2.0,
        open_rate: float = 0.0,
        click_rate: float = 0.0,
        offline_attended: bool = False,
        product_scan: bool = False,
    ) -> SegmentAssignment:
        """Assign a grower to a micro-segment and return content strategy."""

        # 1. Try existing lookup
        existing_seg = self._lookup_existing(grower_id)
        segment_id = existing_seg

        # 2. If no existing segment and model is loaded, predict with clusterer
        if segment_id is None and self.clusterer is not None and self.embedding_pipeline is not None:
            try:
                segment_id = self._predict_segment(
                    crop, device_type, language, state,
                    grower_age, farm_size, open_rate, click_rate,
                    offline_attended, product_scan
                )
            except Exception as e:
                print(f"Clustering inference failed: {e}")
                segment_id = None

        # 3. Fallback
        if segment_id is None:
            segment_id = self._rule_based_assign(crop, device_type, language)

        # 4. Retrieve segment profile + template
        return self._build_assignment(grower_id, segment_id, crop, language)

    def _predict_segment(self, crop, device_type, language, state,
                          age, farm_size, open_rate, click_rate,
                          offline_attended, product_scan) -> str:
        """Run embedding pipeline + clusterer to predict segment."""
        pca = self.embedding_pipeline["pca"]
        scaler = self.embedding_pipeline["scaler"]

        # Reconstruct simple feature vector (must match training)
        # Build one-hot manually using known categories
        crop_cats = ["barley","chickpea","cumin","lentil","maize","mustard","potato","safflower","wheat"]
        lang_cats = ["Bengali","Gujarati","Hindi","Kannada","Marathi","Punjabi"]
        state_cats = ["Bihar","Gujarat","Haryana","Karnataka","Madhya Pradesh",
                      "Maharashtra","Punjab","Rajasthan","Uttar Pradesh","West Bengal"]
        dev_cats = ["keypad","smartphone","unknown"]
        gen_cats = ["female","male"]

        def one_hot(val, cats):
            return [1.0 if val == c else 0.0 for c in cats]

        ohe = (
            one_hot(crop, crop_cats) +
            one_hot(language, lang_cats) +
            one_hot(state, state_cats) +
            one_hot(device_type, dev_cats) +
            one_hot("male", gen_cats)  # gender unknown → assume male as mode
        )
        numeric = scaler.transform([[age, farm_size, open_rate, click_rate, 0]])[0].tolist()
        flags = [float(offline_attended), float(product_scan)]

        features = np.array(ohe + numeric + flags).reshape(1, -1)
        # Pad/trim to match PCA input dims
        n_in = pca.n_features_in_
        if features.shape[1] < n_in:
            features = np.pad(features, ((0, 0), (0, n_in - features.shape[1])))
        else:
            features = features[:, :n_in]

        reduced = pca.transform(features)

        if self.clusterer_type == "hdbscan":
            try:
                import hdbscan
                labels, _ = hdbscan.approximate_predict(self.clusterer, reduced)
                label = int(labels[0])
            except Exception:
                label = 0
        else:
            label = int(self.clusterer.predict(reduced)[0])

        return f"SEG_{label:04d}"

    def _build_assignment(self, grower_id: str, segment_id: str,
                           crop: str, language: str) -> SegmentAssignment:
        """Look up template and build SegmentAssignment response."""
        product = CROP_PRODUCT_MAP.get(crop, "Tilt 250 EC")
        lang_meta = LANGUAGE_META.get(language, {"tts_code": "hi-IN"})

        # Find matching segment in templates
        if self.seg_templates is not None and segment_id in self.seg_templates["segment_id"].values:
            row = self.seg_templates[self.seg_templates["segment_id"] == segment_id].iloc[0]
            channel = row["channel"]
            content_format = row["channel"]
            llm_prompt = row["llm_prompt"]
            seg_size = int(row["size"])
        else:
            # Fallback: generate LLM prompt with the 4 hyper-personalization variables
            # from solution doc Engine 4: {name_token, tehsil, crop_stage, nearest_retailer}
            # "Generate one creative template per cluster, then LLM hyper-personalizes
            # with variables: {name_token, tehsil, crop_stage, nearest_retailer}"
            channel = "whatsapp_rich"
            content_format = "whatsapp_rich"
            llm_prompt = (
                f"Write a WhatsApp message in {language} for {{name_token}}, "
                f"a {crop} farmer in {{tehsil}}. "
                f"Their crop is currently at the {{crop_stage}} stage. "
                f"Recommend {product} by Syngenta. "
                f"Nearest retailer with stock: {{nearest_retailer}}. "
                f"Max 512 characters. Warm, advisory tone. No jargon."
            )
            seg_size = 0

        # Average click rate for segment
        avg_click = 0.0
        if self.seg_profiles is not None and segment_id in self.seg_profiles["segment_id"].values:
            row_p = self.seg_profiles[self.seg_profiles["segment_id"] == segment_id].iloc[0]
            avg_click = float(row_p.get("avg_click_rate", 0.0))

        return SegmentAssignment(
            grower_id=grower_id,
            segment_id=segment_id,
            recommended_channel=channel,
            recommended_product=product,
            content_format=content_format,
            language=language,
            tts_code=lang_meta["tts_code"],
            llm_prompt_template=llm_prompt,
            segment_size=seg_size,
            avg_segment_click_rate=round(avg_click, 4),
        )

    def batch_assign(self, growers_df: pd.DataFrame) -> pd.DataFrame:
        """Assign micro-segments to a full grower DataFrame."""
        results = []
        for _, row in growers_df.iterrows():
            asgn = self.assign(
                grower_id=str(row.get("grower_id", "")),
                crop=str(row.get("crop", "wheat")),
                device_type=str(row.get("device_type", "unknown")),
                language=str(row.get("language", "Hindi")),
                state=str(row.get("state", "Uttar Pradesh")),
                grower_age=int(row.get("grower_age", 45)),
                farm_size=float(row.get("grower_farm_size", 2.0)),
                open_rate=float(row.get("wa_open_rate", 0)),
                click_rate=float(row.get("wa_click_rate", 0)),
                offline_attended=bool(row.get("offline_campaign_attended", False)),
                product_scan=bool(row.get("product_scan", False)),
            )
            results.append({
                "grower_id": asgn.grower_id,
                "segment_id": asgn.segment_id,
                "recommended_channel": asgn.recommended_channel,
                "recommended_product": asgn.recommended_product,
                "content_format": asgn.content_format,
                "avg_segment_click_rate": asgn.avg_segment_click_rate,
            })

        return growers_df.merge(pd.DataFrame(results), on="grower_id", how="left")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    engine = MicroSegmentationEngine()

    test_cases = [
        dict(grower_id="GRW_TEST_01", crop="wheat", device_type="smartphone",
             language="Punjabi", state="Punjab", grower_age=52, farm_size=0.55,
             open_rate=0.3, click_rate=0.08, offline_attended=True, product_scan=False),
        dict(grower_id="GRW_TEST_02", crop="chickpea", device_type="keypad",
             language="Hindi", state="Rajasthan", grower_age=65, farm_size=4.0,
             open_rate=0.0, click_rate=0.0, offline_attended=False, product_scan=False),
        dict(grower_id="GRW_TEST_03", crop="potato", device_type="smartphone",
             language="Bengali", state="West Bengal", grower_age=38, farm_size=1.2,
             open_rate=0.5, click_rate=0.2, offline_attended=False, product_scan=True),
    ]

    print("=== Engine 4: Micro-Segmentation Demo ===\n")
    for tc in test_cases:
        result = engine.assign(**tc)
        print(f"Grower: {result.grower_id}")
        print(f"  Segment:   {result.segment_id} (size={result.segment_size})")
        print(f"  Channel:   {result.recommended_channel}")
        print(f"  Product:   {result.recommended_product}")
        print(f"  Avg CTR:   {result.avg_segment_click_rate:.4f}")
        print(f"  LLM Prompt (first 100 chars): {result.llm_prompt_template[:100]}...")
        print()