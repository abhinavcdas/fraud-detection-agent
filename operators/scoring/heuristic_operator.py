"""Heuristic Scoring Operator for cold-start and baseline scoring."""

from typing import Dict, Any
from core.interfaces import BaseScorerOperator

class HeuristicScorerOperator(BaseScorerOperator):
    """Heuristic rule-based scorer used before ML model training is completed."""

    def predict_proba(self, features: Dict[str, Any]) -> float:
        score = 0.05 # Baseline benign score
        amount = float(features.get("amount", 0.0))
        velocity_5m = int(features.get("velocity_5m", 0))
        geo_dist = float(features.get("geo_distance_km", 0.0))
        amount_dev = float(features.get("amount_deviation", 0.0))

        if velocity_5m >= 3:
            score += 0.35
        if geo_dist > 500.0:
            score += 0.30
        if amount > 5000.0 or amount_dev > 4.0:
            score += 0.25

        return min(1.0, round(score, 4))

    def get_model_version(self) -> str:
        return "heuristic-v1.0"
