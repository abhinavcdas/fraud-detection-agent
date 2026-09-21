"""Low-Latency Redis Sliding-Window Feature Store Operator.

Implements sub-millisecond hot-path feature extraction:
1. Real-time sliding window counts via Redis Sorted Sets (ZSET):
   - 10-second velocity
   - 60-second velocity
   - 5-minute velocity
2. Impossible Travel calculation:
   - Tracks most recent location (lat, lon, timestamp)
   - Computes Haversine distance and implied travel speed in km/h
3. Amount deviation z-score against customer's recent window
4. Seamless fallback: Uses fakeredis or in-memory dict when Redis daemon is offline.
"""

import os
import json
import math
import time
from typing import Dict, Any, List, Optional
from core.interfaces import BaseFeatureStoreOperator
from core.logger import get_logger

logger = get_logger("redis_feature_store")


def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points on Earth in kilometers."""
    if any(v is None for v in (lat1, lon1, lat2, lon2)):
        return 0.0
    try:
        R = 6371.0  # Earth radius in km
        phi1, phi2 = math.radians(float(lat1)), math.radians(float(lat2))
        delta_phi = math.radians(float(lat2) - float(lat1))
        delta_lambda = math.radians(float(lon2) - float(lon1))

        a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
        # Clamp 'a' to [0.0, 1.0] to eliminate floating-point precision domain errors
        a = max(0.0, min(1.0, a))
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return float(round(R * c, 3))
    except (ValueError, TypeError):
        return 0.0


class RedisFeatureStore(BaseFeatureStoreOperator):
    """High-performance Redis-backed sliding-window feature store with runtime fallback."""

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.client = None
        self.backend_type = "redis"
        self._memory_fallback: Dict[str, List[Dict[str, Any]]] = {}
        self._init_connection()

    def _init_connection(self) -> None:
        """Attempt to connect to live Redis; fall back to fakeredis or in-memory emulation."""
        try:
            import redis
            client = redis.Redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=1.0,
                socket_connect_timeout=1.0,
                retry_on_timeout=True
            )
            client.ping()
            self.client = client
            self.backend_type = "live_redis"
            logger.info("Connected to live Redis feature store at {url}", url=self.redis_url)
        except Exception as live_err:
            logger.info("Live Redis unavailable ({err}). Attempting fakeredis fallback...", err=str(live_err))
            try:
                import fakeredis
                self.client = fakeredis.FakeRedis(decode_responses=True)
                self.backend_type = "fakeredis"
                logger.info("Initialized in-process fakeredis sliding-window feature store.")
            except ImportError:
                self.client = None
                self.backend_type = "memory"
                logger.info("Initialized pure in-memory sliding-window feature store fallback.")

    def is_healthy(self) -> bool:
        """Check if feature store backend is available."""
        if self.client:
            try:
                return bool(self.client.ping())
            except Exception:
                return False
        return True

    async def record_event(
        self,
        customer_id: str,
        timestamp: float,
        amount: float,
        ip: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None
    ) -> None:
        """Record an event in the sliding-window sorted set with resilient fallback."""
        event_data = {
            "ts": timestamp,
            "amt": amount,
            "ip": ip or "",
            "lat": lat if lat is not None else 0.0,
            "lon": lon if lon is not None else 0.0
        }

        if self.client:
            try:
                zset_key = f"feat:cust:{customer_id}:txs"
                member = json.dumps(event_data)
                self.client.zadd(zset_key, {member: timestamp})
                cutoff_24h = timestamp - 86400.0
                self.client.zremrangebyscore(zset_key, "-inf", cutoff_24h)
                return
            except Exception as exc:
                logger.warning("Redis write error ({err}). Failing over to memory buffer.", err=str(exc))

        # Memory buffer fallback
        if customer_id not in self._memory_fallback:
            self._memory_fallback[customer_id] = []
        self._memory_fallback[customer_id].append(event_data)
        cutoff_24h = timestamp - 86400.0
        self._memory_fallback[customer_id] = [
            e for e in self._memory_fallback[customer_id] if e["ts"] >= cutoff_24h
        ]

    async def get_sliding_window_features(
        self,
        customer_id: str,
        current_timestamp: float,
        current_lat: Optional[float] = None,
        current_lon: Optional[float] = None
    ) -> Dict[str, Any]:
        """Compute rolling velocity counts and impossible travel metrics in sub-2ms."""
        start_t = time.perf_counter()

        events: List[Dict[str, Any]] = []
        if self.client:
            try:
                zset_key = f"feat:cust:{customer_id}:txs"
                cutoff_1h = current_timestamp - 3600.0
                raw_members = self.client.zrangebyscore(zset_key, cutoff_1h, current_timestamp)
                for m in raw_members:
                    try:
                        events.append(json.loads(m))
                    except Exception:
                        pass
            except Exception as exc:
                logger.warning("Redis read error ({err}). Reading from memory buffer fallback.", err=str(exc))
                events = []

        if not events:
            cust_events = self._memory_fallback.get(customer_id, [])
            cutoff_1h = current_timestamp - 3600.0
            events = [e for e in cust_events if cutoff_1h <= e["ts"] <= current_timestamp]


        # Velocity counts across standard banking sliding windows
        window_10s_cutoff = current_timestamp - 10.0
        window_60s_cutoff = current_timestamp - 60.0
        window_5m_cutoff = current_timestamp - 300.0

        vel_10s = sum(1 for e in events if e["ts"] >= window_10s_cutoff)
        vel_60s = sum(1 for e in events if e["ts"] >= window_60s_cutoff)
        vel_5m = sum(1 for e in events if e["ts"] >= window_5m_cutoff)

        # Geolocation & impossible travel calculations
        geo_distance_km = 0.0
        time_since_last_tx = 0.0
        travel_speed_kmh = 0.0

        # Find the most recent previous event
        prior_events = [e for e in events if e["ts"] < current_timestamp]
        if prior_events:
            last_event = max(prior_events, key=lambda x: x["ts"])
            time_since_last_tx = max(0.1, current_timestamp - last_event["ts"])
            last_lat = last_event.get("lat", 0.0)
            last_lon = last_event.get("lon", 0.0)

            if current_lat is not None and current_lon is not None and (last_lat != 0.0 or last_lon != 0.0):
                geo_distance_km = calculate_haversine_distance(last_lat, last_lon, current_lat, current_lon)
                travel_hours = time_since_last_tx / 3600.0
                if travel_hours > 0.0001:
                    travel_speed_kmh = geo_distance_km / travel_hours

        # Amount deviation z-score
        amounts = [e["amt"] for e in prior_events]
        amount_deviation = 0.0
        if len(amounts) >= 3:
            mean_amt = sum(amounts) / len(amounts)
            variance = sum((x - mean_amt) ** 2 for x in amounts) / len(amounts)
            std_amt = math.sqrt(variance)
            current_amt = events[-1]["amt"] if events else 0.0
            if std_amt > 1.0:
                amount_deviation = (current_amt - mean_amt) / std_amt

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        return {
            "customer_id": customer_id,
            "velocity_10s": vel_10s,
            "velocity_60s": vel_60s,
            "velocity_5m": vel_5m,
            "time_since_last_tx_sec": round(time_since_last_tx, 2),
            "geo_distance_km": round(geo_distance_km, 2),
            "travel_speed_kmh": round(travel_speed_kmh, 1),
            "amount_deviation": round(amount_deviation, 2),
            "feature_store_backend": self.backend_type,
            "lookup_latency_ms": round(elapsed_ms, 3)
        }

    async def close(self) -> None:
        """Close Redis connection pools cleanly."""
        if self.client and hasattr(self.client, "close"):
            try:
                self.client.close()
            except Exception:
                pass
