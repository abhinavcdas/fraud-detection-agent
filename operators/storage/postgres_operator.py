"""PostgreSQL Storage Operator for production environment."""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional
from core.interfaces import BaseStorageOperator
from core.logger import get_logger

logger = get_logger("postgres_storage")

class PostgresStorageOperator(BaseStorageOperator):
    """PostgreSQL implementation with connection pooling."""

    def __init__(self, database_url: str = None):
        self.database_url = database_url or os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:postgres@localhost:5432/fraud_detection"
        )
        self.pool = None

    async def initialize(self) -> None:
        loop = asyncio.get_running_loop()
        def _connect():
            try:
                import psycopg2.pool
                self.pool = psycopg2.pool.ThreadedConnectionPool(minconn=2, maxconn=20, dsn=self.database_url)
                logger.info("Initialized PostgreSQL ThreadedConnectionPool.")

                # Ensure required tables exist
                conn = self._get_conn()
                try:
                    with conn.cursor() as cur:
                        cur.execute("""
                            CREATE TABLE IF NOT EXISTS raw_transactions (
                                transaction_id VARCHAR(64) PRIMARY KEY,
                                timestamp TIMESTAMP WITH TIME ZONE,
                                time_step DOUBLE PRECISION,
                                customer_id VARCHAR(64) NOT NULL,
                                merchant_id VARCHAR(64) NOT NULL,
                                amount DOUBLE PRECISION NOT NULL,
                                lat DOUBLE PRECISION,
                                lon DOUBLE PRECISION,
                                is_fraud INTEGER DEFAULT 0,
                                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                            );
                            CREATE INDEX IF NOT EXISTS idx_raw_cust_time ON raw_transactions (customer_id, timestamp DESC);

                            CREATE TABLE IF NOT EXISTS engineered_features (
                                transaction_id VARCHAR(64) PRIMARY KEY,
                                customer_id VARCHAR(64) NOT NULL,
                                timestamp TIMESTAMP WITH TIME ZONE,
                                amount DOUBLE PRECISION NOT NULL,
                                velocity_5m INTEGER NOT NULL DEFAULT 0,
                                velocity_60m INTEGER NOT NULL DEFAULT 0,
                                amount_deviation DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                                time_since_last_tx_sec DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                                geo_distance_km DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                                travel_speed_kmh DOUBLE PRECISION DEFAULT 0.0,
                                is_fraud INTEGER DEFAULT 0
                            );
                            CREATE INDEX IF NOT EXISTS idx_eng_cust ON engineered_features (customer_id);

                            CREATE TABLE IF NOT EXISTS audit_log (
                                audit_id SERIAL PRIMARY KEY,
                                transaction_id VARCHAR(64) NOT NULL,
                                timestamp TIMESTAMP WITH TIME ZONE,
                                input_hash VARCHAR(64) NOT NULL,
                                model_version VARCHAR(64) NOT NULL,
                                fraud_score DOUBLE PRECISION NOT NULL,
                                is_flagged BOOLEAN NOT NULL,
                                agent_decision VARCHAR(64),
                                agent_report JSONB,
                                guardrail_status VARCHAR(32),
                                faithfulness_score DOUBLE PRECISION,
                                latency_ms DOUBLE PRECISION
                            );
                            CREATE INDEX IF NOT EXISTS idx_audit_tx_id ON audit_log (transaction_id);
                        """)
                        conn.commit()
                except Exception as e:
                    conn.rollback()
                    logger.warning("Could not execute DDL schema check: {err}", err=str(e))
                finally:
                    self._put_conn(conn)

            except ImportError:
                raise ImportError("psycopg2 is required for PostgresStorageOperator. Install it via `pip install psycopg2-binary`.")
        await loop.run_in_executor(None, _connect)

    def _get_conn(self):
        return self.pool.getconn()

    def _put_conn(self, conn):
        self.pool.putconn(conn)

    async def save_raw_transaction(self, tx: Dict[str, Any]) -> None:
        await self.save_raw_transactions_batch([tx])

    async def save_raw_transactions_batch(self, txs: List[Dict[str, Any]]) -> None:
        if not txs:
            return
        loop = asyncio.get_running_loop()
        def _sync_save():
            conn = self._get_conn()
            try:
                with conn.cursor() as cur:
                    cols = [k for k in txs[0].keys() if k in {
                        "transaction_id", "timestamp", "time_step", "customer_id", "merchant_id", "amount",
                        "lat", "lon", "is_fraud", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                        "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
                        "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28"
                    }]
                    placeholders = ", ".join(["%s"] * len(cols))
                    col_names = ", ".join(cols)
                    query = f"""
                        INSERT INTO raw_transactions ({col_names})
                        VALUES ({placeholders})
                        ON CONFLICT (transaction_id) DO NOTHING
                    """
                    records = [[tx.get(c) for c in cols] for tx in txs]
                    cur.executemany(query, records)
                    conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                self._put_conn(conn)
        await loop.run_in_executor(None, _sync_save)

    async def save_engineered_features(self, features: Dict[str, Any]) -> None:
        await self.save_engineered_features_batch([features])

    async def save_engineered_features_batch(self, features_list: List[Dict[str, Any]]) -> None:
        if not features_list:
            return
        loop = asyncio.get_running_loop()
        def _sync_save():
            conn = self._get_conn()
            try:
                with conn.cursor() as cur:
                    cols = [k for k in features_list[0].keys() if k in {
                        "transaction_id", "customer_id", "timestamp", "amount", "velocity_5m", "velocity_60m",
                        "amount_deviation", "time_since_last_tx_sec", "geo_distance_km", "travel_speed_kmh", "is_fraud",
                        "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
                        "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
                        "v24", "v25", "v26", "v27", "v28"
                    }]
                    placeholders = ", ".join(["%s"] * len(cols))
                    col_names = ", ".join(cols)
                    query = f"""
                        INSERT INTO engineered_features ({col_names})
                        VALUES ({placeholders})
                        ON CONFLICT (transaction_id) DO UPDATE SET
                            velocity_5m = EXCLUDED.velocity_5m,
                            velocity_60m = EXCLUDED.velocity_60m,
                            amount_deviation = EXCLUDED.amount_deviation,
                            geo_distance_km = EXCLUDED.geo_distance_km
                    """
                    records = [[f.get(c) for c in cols] for f in features_list]
                    cur.executemany(query, records)
                    conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                self._put_conn(conn)
        await loop.run_in_executor(None, _sync_save)

    async def get_customer_history(self, customer_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        loop = asyncio.get_running_loop()
        def _sync_get():
            from psycopg2.extras import RealDictCursor
            conn = self._get_conn()
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        "SELECT * FROM raw_transactions WHERE customer_id = %s ORDER BY timestamp DESC LIMIT %s",
                        (customer_id, limit)
                    )
                    return [dict(r) for r in cur.fetchall()]
            except Exception:
                conn.rollback()
                raise
            finally:
                self._put_conn(conn)
        return await loop.run_in_executor(None, _sync_get)

    async def save_audit_log(self, audit_record: Dict[str, Any]) -> None:
        loop = asyncio.get_running_loop()
        def _sync_save():
            conn = self._get_conn()
            try:
                with conn.cursor() as cur:
                    query = """
                        INSERT INTO audit_log
                        (transaction_id, timestamp, input_hash, model_version, fraud_score, is_flagged,
                         agent_decision, agent_report, guardrail_status, faithfulness_score, latency_ms)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    cur.execute(query, (
                        audit_record["transaction_id"],
                        audit_record.get("timestamp"),
                        audit_record.get("input_hash", ""),
                        audit_record.get("model_version", "unknown"),
                        float(audit_record.get("fraud_score", 0.0)),
                        bool(audit_record.get("is_flagged")),
                        audit_record.get("agent_decision"),
                        json.dumps(audit_record.get("agent_report")),
                        audit_record.get("guardrail_status"),
                        audit_record.get("faithfulness_score"),
                        audit_record.get("latency_ms")
                    ))
                    conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                self._put_conn(conn)
        await loop.run_in_executor(None, _sync_save)

    async def get_recent_flagged_transactions(self, limit: int = 20) -> List[Dict[str, Any]]:
        loop = asyncio.get_running_loop()
        def _sync_get():
            from psycopg2.extras import RealDictCursor
            conn = self._get_conn()
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        "SELECT * FROM audit_log WHERE is_flagged = TRUE ORDER BY timestamp DESC LIMIT %s",
                        (limit,)
                    )
                    return [dict(r) for r in cur.fetchall()]
            except Exception:
                conn.rollback()
                raise
            finally:
                self._put_conn(conn)
        return await loop.run_in_executor(None, _sync_get)

    async def get_audit_record_by_tx(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        loop = asyncio.get_running_loop()
        def _sync_get():
            from psycopg2.extras import RealDictCursor
            conn = self._get_conn()
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        "SELECT * FROM audit_log WHERE transaction_id = %s ORDER BY timestamp DESC LIMIT 1",
                        (transaction_id,)
                    )
                    row = cur.fetchone()
                    return dict(row) if row else None
            except Exception:
                conn.rollback()
                raise
            finally:
                self._put_conn(conn)
        return await loop.run_in_executor(None, _sync_get)

    async def close(self) -> None:
        if self.pool:
            self.pool.closeall()
            logger.info("Closed PostgreSQL connection pool.")
