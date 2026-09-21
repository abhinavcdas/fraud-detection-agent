"""Rolling Feature Extraction Operator.

Computes real-time streaming velocity, amount deviation, and geo-spatial jumps
based on incoming transaction and historical customer events.
"""

import math
from datetime import datetime
from typing import Dict, Any, List
from core.interfaces import BaseFeatureOperator
from consumer.feature_engineering import haversine_distance

class RollingFeatureOperator(BaseFeatureOperator):
    """Production feature engineering operator."""

    def extract_features(self, current_tx: Dict[str, Any], customer_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute rolling features from customer history."""
        curr_amount = float(current_tx.get("amount", 0.0))
        curr_time = current_tx.get("time_step", 0.0) # Kaggle Time field is seconds elapsed
        curr_lat = current_tx.get("lat")
        curr_lon = current_tx.get("lon")

        velocity_5m = 0
        velocity_60m = 0
        past_amounts = []
        time_since_last = 0.0
        geo_distance = 0.0

        if customer_history:
            # Sort history by time descending
            sorted_history = sorted(
                customer_history,
                key=lambda x: float(x.get("time_step", 0.0)),
                reverse=True
            )
            
            last_tx = sorted_history[0]
            last_time = float(last_tx.get("time_step", 0.0))
            time_since_last = max(0.0, curr_time - last_time)

            # Geo distance from immediate previous location
            if curr_lat is not None and curr_lon is not None:
                prev_lat = last_tx.get("lat")
                prev_lon = last_tx.get("lon")
                if prev_lat is not None and prev_lon is not None:
                    geo_distance = haversine_distance(curr_lat, curr_lon, prev_lat, prev_lon)

            # Windowed aggregations
            for h in sorted_history:
                h_time = float(h.get("time_step", 0.0))
                diff_sec = curr_time - h_time
                if diff_sec <= 300.0:  # 5 minutes
                    velocity_5m += 1
                if diff_sec <= 3600.0: # 60 minutes
                    velocity_60m += 1
                past_amounts.append(float(h.get("amount", 0.0)))

        # Amount deviation calculation
        if past_amounts:
            avg_amt = sum(past_amounts) / len(past_amounts)
            var_amt = sum((x - avg_amt) ** 2 for x in past_amounts) / len(past_amounts)
            std_amt = math.sqrt(var_amt)
            amount_deviation = (curr_amount - avg_amt) / (std_amt + 1.0)
        else:
            amount_deviation = 0.0

        # Respect incoming features if customer_history is not yet accumulated
        final_velocity_5m = max(velocity_5m, int(current_tx.get("velocity_5m", 0)))
        final_velocity_60m = max(velocity_60m, int(current_tx.get("velocity_60m", 0)))
        final_geo_dist = max(geo_distance, float(current_tx.get("geo_distance_km", 0.0)))
        final_amount_dev = amount_deviation if past_amounts else float(current_tx.get("amount_deviation", 0.0))

        # Construct complete feature dictionary
        features = {
            "transaction_id": current_tx["transaction_id"],
            "customer_id": current_tx["customer_id"],
            "timestamp": current_tx.get("timestamp", ""),
            "time_step": curr_time,
            "amount": curr_amount,
            "velocity_5m": final_velocity_5m,
            "velocity_60m": final_velocity_60m,
            "amount_deviation": round(final_amount_dev, 4),
            "time_since_last_tx_sec": round(time_since_last, 2),
            "geo_distance_km": round(final_geo_dist, 2),
            "is_fraud": int(current_tx.get("is_fraud", 0))
        }

        # Pass through PCA anonymized features V1-V28 if present
        for i in range(1, 29):
            key = f"v{i}"
            if key in current_tx:
                features[key] = current_tx[key]

        return features
