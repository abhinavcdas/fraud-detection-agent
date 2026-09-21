"""Streaming Feature Engineering for Real-Time Fraud Detection.

Computes rolling behavioral and velocity features per customer:
1. velocity_5m: transaction count in the last 5 minutes (300 seconds).
2. velocity_60m: transaction count in the last 60 minutes (3,600 seconds).
3. amount_deviation: z-score or normalized distance from customer's historical mean & std spend.
4. time_since_last_tx_sec: delta seconds elapsed since previous transaction.
5. geo_distance_km: haversine geographic distance (in kilometers) from previous transaction location.
"""

import math
from datetime import datetime
from typing import Dict, Any, List, Optional

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in kilometers between two geographic coordinates."""
    if any(v is None for v in (lat1, lon1, lat2, lon2)):
        return 0.0
    r = 6371.0 # Mean Earth radius in kilometers
    try:
        phi1, phi2 = math.radians(float(lat1)), math.radians(float(lat2))
        dphi = math.radians(float(lat2) - float(lat1))
        dlambda = math.radians(float(lon2) - float(lon1))
        
        a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
        # Clamp 'a' to [0.0, 1.0] to eliminate floating-point precision domain errors
        a = max(0.0, min(1.0, a))
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return round(r * c, 3)
    except (ValueError, TypeError):
        return 0.0

def parse_time_seconds(tx: Dict[str, Any]) -> float:
    """Extract or parse continuous time in seconds from transaction event."""
    if "time_step" in tx and tx["time_step"] is not None:
        try:
            return float(tx["time_step"])
        except (ValueError, TypeError):
            pass
    if "timestamp" in tx and tx["timestamp"]:
        try:
            dt = datetime.fromisoformat(str(tx["timestamp"]).replace("Z", "+00:00"))
            return dt.timestamp()
        except Exception:
            pass
    return 0.0

def compute_rolling_features(current_tx: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute rolling features given current transaction and historical customer transactions.
    
    Args:
        current_tx: Dictionary containing the incoming transaction attributes.
        history: List of historical transaction dictionaries for this customer.
        
    Returns:
        Dictionary of enriched rolling features.
    """
    curr_amount = float(current_tx.get("amount", 0.0))
    curr_time = parse_time_seconds(current_tx)
    curr_lat = current_tx.get("lat")
    curr_lon = current_tx.get("lon")

    velocity_5m = 0
    velocity_60m = 0
    past_amounts = []
    time_since_last = 0.0
    geo_distance = 0.0

    if history:
        # Sort history by time descending (most recent first)
        sorted_history = sorted(
            history,
            key=lambda x: parse_time_seconds(x),
            reverse=True
        )

        # Most recent prior transaction
        last_tx = sorted_history[0]
        last_time = parse_time_seconds(last_tx)
        time_since_last = max(0.0, curr_time - last_time)

        # Geo-distance jump from immediate previous location
        if curr_lat is not None and curr_lon is not None:
            prev_lat = last_tx.get("lat")
            prev_lon = last_tx.get("lon")
            if prev_lat is not None and prev_lon is not None:
                geo_distance = haversine_distance(curr_lat, curr_lon, prev_lat, prev_lon)

        # Windowed aggregations over history
        for past_tx in sorted_history:
            h_time = parse_time_seconds(past_tx)
            diff_sec = curr_time - h_time
            if diff_sec <= 300.0:  # 5 minutes
                velocity_5m += 1
            if diff_sec <= 3600.0: # 60 minutes
                velocity_60m += 1
            past_amounts.append(float(past_tx.get("amount", 0.0)))

    # Amount deviation calculation
    if past_amounts:
        avg_amt = sum(past_amounts) / len(past_amounts)
        var_amt = sum((x - avg_amt) ** 2 for x in past_amounts) / len(past_amounts)
        std_amt = math.sqrt(var_amt)
        amount_deviation = (curr_amount - avg_amt) / (std_amt + 1.0)
    else:
        amount_deviation = 0.0

    # Respect pre-calculated values if history has not accumulated yet
    final_velocity_5m = max(velocity_5m, int(current_tx.get("velocity_5m", 0)))
    final_velocity_60m = max(velocity_60m, int(current_tx.get("velocity_60m", 0)))
    final_geo_dist = max(geo_distance, float(current_tx.get("geo_distance_km", 0.0)))
    final_amount_dev = amount_deviation if past_amounts else float(current_tx.get("amount_deviation", 0.0))

    # Calculate travel speed (km/h) for impossible speed detection
    if time_since_last > 0:
        travel_speed_kmh = round(final_geo_dist / (time_since_last / 3600.0), 2)
    elif final_geo_dist > 0:
        travel_speed_kmh = 999999.0
    else:
        travel_speed_kmh = 0.0

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
        "travel_speed_kmh": travel_speed_kmh,
        "is_fraud": int(current_tx.get("is_fraud", 0)),
        "merchant_id": current_tx.get("merchant_id", "MERCH_001")
    }

    # Pass through PCA anonymized features V1-V28
    for i in range(1, 29):
        key = f"v{i}"
        if key in current_tx:
            features[key] = current_tx[key]

    return features

