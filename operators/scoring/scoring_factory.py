"""Model Scoring Operator Factory."""

import os
from core.interfaces import BaseScorerOperator
from operators.scoring.heuristic_operator import HeuristicScorerOperator
from operators.scoring.xgboost_operator import XGBoostScorerOperator
from model.onnx_scorer import ONNXModelScorer

def get_scoring_operator(model_type: str = None) -> BaseScorerOperator:
    selected = (model_type or os.getenv("MODEL_SCORER", "xgboost")).lower().strip()
    if selected in ["onnx", "onnxruntime"]:
        return ONNXModelScorer()
    elif selected == "xgboost":
        return XGBoostScorerOperator()
    return HeuristicScorerOperator()

get_scorer_operator = get_scoring_operator

