"""ONNX Runtime Accelerated Model Scoring Operator.

Provides sub-5ms inline inference on the hot path using ONNX Runtime:
- Ingests tabular transaction feature vectors.
- Executes optimized C++ inference kernels via CPUExecutionProvider.
- Emits calibrated fraud class probabilities.
- Gracefully falls back to standard XGBoost / heuristic scorer if ONNX runtime is unconfigured.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from core.interfaces import BaseScorerOperator
from core.logger import get_logger

logger = get_logger("onnx_scorer")

DEFAULT_ONNX_PATH = Path(__file__).resolve().parent / "registry" / "fraud_xgb_v1.onnx"
DEFAULT_METADATA_PATH = Path(__file__).resolve().parent / "registry" / "model_metadata.json"

FEATURE_COLUMNS = [
    *[f"v{i}" for i in range(1, 29)],
    "amount",
    "velocity_5m",
    "velocity_60m",
    "amount_deviation",
    "time_since_last_tx_sec",
    "geo_distance_km"
]


class ONNXModelScorer(BaseScorerOperator):
    """Ultra-fast ONNX Runtime inference operator."""

    def __init__(self, onnx_path: Optional[str] = None, version: str = "fraud-xgb-v1-onnx"):
        self.version = version
        self.onnx_path = Path(onnx_path) if onnx_path else DEFAULT_ONNX_PATH
        self.metadata_path = DEFAULT_METADATA_PATH
        self.session = None
        self.input_name = "float_input"
        self.feature_names = FEATURE_COLUMNS
        self.calibrated_threshold = 0.38
        self._fallback_scorer = None
        self._initialize_session()

    def _initialize_session(self) -> None:
        """Load ONNX inference session or prepare fallback."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    self.feature_names = meta.get("features", FEATURE_COLUMNS)
                    self.calibrated_threshold = meta.get("optimal_threshold", 0.38)
            except Exception:
                pass

        if self.onnx_path.exists():
            try:
                import onnxruntime as ort
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = 2
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                self.session = ort.InferenceSession(
                    str(self.onnx_path),
                    sess_options,
                    providers=["CPUExecutionProvider"]
                )
                self.input_name = self.session.get_inputs()[0].name
                logger.info("ONNX Runtime session initialized successfully from {path}", path=self.onnx_path)
                return
            except Exception as e:
                logger.warning("Could not initialize ONNX session ({err}). Reverting to standard XGBoost.", err=str(e))

        # Fallback to standard XGBoost operator
        from operators.scoring.xgboost_operator import XGBoostScorerOperator
        self._fallback_scorer = XGBoostScorerOperator()

    def _extract_feature_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Align input dictionary to 34-feature float32 tensor."""
        vec = []
        for col in self.feature_names:
            val = features.get(col, 0.0)
            try:
                vec.append(float(val) if val is not None else 0.0)
            except (ValueError, TypeError):
                vec.append(0.0)
        return np.array([vec], dtype=np.float32)

    def predict_proba(self, features: Dict[str, Any]) -> float:
        """Compute fraud probability in range [0.0, 1.0]. Sub-5ms execution."""
        if self.session is not None:
            try:
                X_tensor = self._extract_feature_vector(features)
                outputs = self.session.run(None, {self.input_name: X_tensor})
                # outputs[1] contains [[P(non_fraud), P(fraud)]]
                prob_fraud = float(outputs[1][0, 1])
                return round(prob_fraud, 4)
            except Exception as e:
                logger.warning("ONNX inference failed ({err}). Delegating to fallback scorer.", err=str(e))

        if self._fallback_scorer:
            return self._fallback_scorer.predict_proba(features)

        return 0.05

    def get_model_version(self) -> str:
        """Return registered model version."""
        return self.version
