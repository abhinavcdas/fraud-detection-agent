"""SQLite Storage Operator for zero-setup persistent local development."""

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional
from core.interfaces import BaseStorageOperator
from core.logger import get_logger

logger = get_logger("sqlite_storage")

class SqliteStorageOperator(BaseStorageOperator):
    """SQLite implementation of BaseStorageOperator."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
            data_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(data_dir / "fraud_local.db")
        else:
            self.db_path = db_path

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    async def initialize(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._init_schema)
        logger.info("Initialized SQLite storage backend at: {path}", path=self.db_path)

    def _init_schema(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS raw_transactions (
                    transaction_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    customer_id TEXT NOT NULL,
                    merchant_id TEXT NOT NULL,
                    amount REAL NOT NULL,
                    lat REAL,
                    lon REAL,
                    is_fraud INTEGER DEFAULT 0,
                    raw_json TEXT
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_raw_cust ON raw_transactions(customer_id)")
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS engineered_features (
                    transaction_id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    amount REAL NOT NULL,
                    velocity_5m INTEGER,
                    velocity_60m INTEGER,
                    amount_deviation REAL,
                    time_since_last_tx_sec REAL,
                    geo_distance_km REAL,
                    is_fraud INTEGER DEFAULT 0,
                    features_json TEXT
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_eng_cust ON engineered_features(customer_id)")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transaction_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    fraud_score REAL NOT NULL,
                    is_flagged INTEGER NOT NULL,
                    agent_decision TEXT,
                    agent_report_json TEXT,
                    guardrail_status TEXT,
                    faithfulness_score REAL,
                    latency_ms REAL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_tx ON audit_log(transaction_id)")
            conn.commit()

    async def save_raw_transaction(self, tx: Dict[str, Any]) -> None:
        loop = asyncio.get_running_loop()
        def _sync_save():
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO raw_transactions 
                    (transaction_id, timestamp, customer_id, merchant_id, amount, lat, lon, is_fraud, raw_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tx["transaction_id"],
                        str(tx.get("timestamp", "")),
                        tx["customer_id"],
                        tx.get("merchant_id", "UNKNOWN"),
                        float(tx.get("amount", 0.0)),
                        float(tx.get("lat", 0.0)) if tx.get("lat") is not None else None,
                        float(tx.get("lon", 0.0)) if tx.get("lon") is not None else None,
                        int(tx.get("is_fraud", 0)),
                        json.dumps(tx)
                    )
                )
                conn.commit()
        await loop.run_in_executor(None, _sync_save)

    async def save_engineered_features(self, features: Dict[str, Any]) -> None:
        loop = asyncio.get_running_loop()
        def _sync_save():
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO engineered_features
                    (transaction_id, customer_id, timestamp, amount, velocity_5m, velocity_60m, 
                     amount_deviation, time_since_last_tx_sec, geo_distance_km, is_fraud, features_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        features["transaction_id"],
                        features["customer_id"],
                        str(features.get("timestamp", "")),
                        float(features.get("amount", 0.0)),
                        int(features.get("velocity_5m", 0)),
                        int(features.get("velocity_60m", 0)),
                        float(features.get("amount_deviation", 0.0)),
                        float(features.get("time_since_last_tx_sec", 0.0)),
                        float(features.get("geo_distance_km", 0.0)),
                        int(features.get("is_fraud", 0)),
                        json.dumps(features)
                    )
                )
                conn.commit()
        await loop.run_in_executor(None, _sync_save)

    async def get_customer_history(self, customer_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        loop = asyncio.get_running_loop()
        def _sync_get():
            with self._get_connection() as conn:
                cur = conn.execute(
                    "SELECT raw_json FROM raw_transactions WHERE customer_id = ? ORDER BY timestamp DESC LIMIT ?",
                    (customer_id, limit)
                )
                return [json.loads(row["raw_json"]) for row in cur.fetchall()]
        return await loop.run_in_executor(None, _sync_get)

    async def save_audit_log(self, audit_record: Dict[str, Any]) -> None:
        loop = asyncio.get_running_loop()
        def _sync_save():
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO audit_log
                    (transaction_id, timestamp, input_hash, model_version, fraud_score, is_flagged,
                     agent_decision, agent_report_json, guardrail_status, faithfulness_score, latency_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        audit_record["transaction_id"],
                        str(audit_record.get("timestamp", "")),
                        audit_record.get("input_hash", ""),
                        audit_record.get("model_version", "unknown"),
                        float(audit_record.get("fraud_score", 0.0)),
                        1 if audit_record.get("is_flagged") else 0,
                        audit_record.get("agent_decision", "NONE"),
                        json.dumps(audit_record.get("agent_report")),
                        audit_record.get("guardrail_status", "SKIPPED"),
                        float(audit_record.get("faithfulness_score", 0.0)) if audit_record.get("faithfulness_score") is not None else None,
                        float(audit_record.get("latency_ms", 0.0))
                    )
                )
                conn.commit()
        await loop.run_in_executor(None, _sync_save)

    async def get_recent_flagged_transactions(self, limit: int = 20) -> List[Dict[str, Any]]:
        loop = asyncio.get_running_loop()
        def _sync_get():
            with self._get_connection() as conn:
                cur = conn.execute(
                    "SELECT * FROM audit_log WHERE is_flagged = 1 ORDER BY audit_id DESC LIMIT ?",
                    (limit,)
                )
                return [dict(row) for row in cur.fetchall()]
        return await loop.run_in_executor(None, _sync_get)

    async def get_audit_record_by_tx(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        loop = asyncio.get_running_loop()
        def _sync_get():
            with self._get_connection() as conn:
                cur = conn.execute(
                    "SELECT * FROM audit_log WHERE transaction_id = ? ORDER BY audit_id DESC LIMIT 1",
                    (transaction_id,)
                )
                row = cur.fetchone()
                return dict(row) if row else None
        return await loop.run_in_executor(None, _sync_get)

    async def close(self) -> None:
        logger.info("Closed SQLite storage connection.")
