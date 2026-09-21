"""XGBoost / Gradient Boosted Model Scoring Operator.

Loads registered model artifact from model/registry/fraud_xgb_v1.joblib,
transforms incoming feature dictionaries into aligned feature vectors,
and computes real-time fraud probability scores.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from core.interfaces import BaseScorerOperator
from core.logger import get_logger

logger = get_logger("xgboost_scorer")

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "model" / "registry" / "fraud_xgb_v1.joblib"
DEFAULT_METADATA_PATH = Path(__file__).resolve().parent.parent.parent / "model" / "registry" / "model_metadata.json"

FEATURE_COLUMNS = [
    *[f"v{i}" for i in range(1, 29)],
    "amount",
    "velocity_5m",
    "velocity_60m",
    "amount_deviation",
    "time_since_last_tx_sec",
    "geo_distance_km"
]

class XGBoostScorerOperator(BaseScorerOperator):
    """Production ML model inference operator."""

    def __init__(self, model_path: str = None, version: str = "fraud-xgb-v1"):
        self.version = version
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.metadata_path = DEFAULT_METADATA_PATH
        self.model = None
        self.feature_names = FEATURE_COLUMNS
        self.calibrated_threshold = 0.5
        self.load_model()

    def load_model(self):
        """Load trained model and metadata from disk."""
        if self.model_path.exists():
            try:
                import joblib
                self.model = joblib.load(self.model_path)
                logger.info("Successfully loaded champion fraud model from: {path}", path=self.model_path)

                if self.metadata_path.exists():
                    with open(self.metadata_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        self.feature_names = meta.get("features", FEATURE_COLUMNS)
                        self.calibrated_threshold = meta.get("optimal_threshold", 0.5)
                        logger.info("Loaded model metadata: Strategy={strat} | PR-AUC={pr:.4f} | Optimal Thresh={th}",
                                    strat=meta.get("champion_strategy"),
                                    pr=meta.get("pr_auc", 0.0),
                                    th=self.calibrated_threshold)
            except Exception as e:
                logger.error("Failed to load model artifact at {path}: {err}", path=self.model_path, err=str(e))
                self.model = None
        else:
            logger.warning("No pre-trained model found at {path}. Scorer will use fallback heuristic.",
                           path=self.model_path)

    def _extract_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Align feature dictionary to model's expected feature vector."""
        vec = []
        for col in self.feature_names:
            val = features.get(col, 0.0)
            try:
                vec.append(float(val) if val is not None else 0.0)
            except (ValueError, TypeError):
                vec.append(0.0)
        return np.array([vec], dtype=np.float32)

    def predict_proba(self, features: Dict[str, Any]) -> float:
        """Compute fraud probability in range [0.0, 1.0]."""
        if self.model is not None:
            try:
                import pandas as pd
                # Feed as DataFrame with column names to avoid feature name mismatch warnings
                X_df = pd.DataFrame([features]).reindex(columns=self.feature_names, fill_value=0.0)
                probs = self.model.predict_proba(X_df)
                return round(float(probs[0, 1]), 4)
            except Exception as e:
                logger.warning("Model inference error ({err}). Reverting to heuristic fallback.", err=str(e))

        # Graceful fallback heuristic if model is uninitialized
        amount = float(features.get("amount", 0.0))
        velocity = int(features.get("velocity_5m", 0))
        geo_dist = float(features.get("geo_distance_km", 0.0))

        score = 0.05
        if velocity >= 3:
            score += 0.40
        if amount > 1000.0:
            score += 0.35
        if geo_dist > 500.0:
            score += 0.30

        return min(1.0, round(score, 4))

    def get_model_version(self) -> str:
        return self.version
