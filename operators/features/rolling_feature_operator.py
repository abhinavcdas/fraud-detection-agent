"""Rolling Feature Extraction Operator.

Computes real-time streaming velocity, amount deviation, and geo-spatial jumps
based on incoming transaction and historical customer events.
"""

from typing import Dict, Any, List
from core.interfaces import BaseFeatureOperator
from consumer.feature_engineering import compute_rolling_features

class RollingFeatureOperator(BaseFeatureOperator):
    """Production feature engineering operator delegating to central feature logic."""

    def extract_features(self, current_tx: Dict[str, Any], customer_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute rolling features from customer history."""
        return compute_rolling_features(current_tx, customer_history)

