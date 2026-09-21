"""In-Memory Storage Operator for ultra-fast tests and mock environments."""

import asyncio
from typing import Dict, Any, List, Optional
from core.interfaces import BaseStorageOperator
from core.logger import get_logger

logger = get_logger("memory_storage")

class MemoryStorageOperator(BaseStorageOperator):
    """In-memory thread/async-safe storage backend."""

    def __init__(self):
        self.raw_transactions: Dict[str, Dict[str, Any]] = {}
        self.engineered_features: Dict[str, Dict[str, Any]] = {}
        self.audit_log: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        logger.info("Initialized In-Memory storage operator.")

    async def save_raw_transaction(self, tx: Dict[str, Any]) -> None:
        async with self._lock:
            tx_id = tx["transaction_id"]
            self.raw_transactions[tx_id] = tx.copy()

    async def save_engineered_features(self, features: Dict[str, Any]) -> None:
        async with self._lock:
            tx_id = features["transaction_id"]
            self.engineered_features[tx_id] = features.copy()

    async def get_customer_history(self, customer_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        async with self._lock:
            history = [
                tx for tx in self.raw_transactions.values()
                if tx.get("customer_id") == customer_id
            ]
            # Sort newest first if timestamp exists
            return sorted(history, key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]

    async def save_audit_log(self, audit_record: Dict[str, Any]) -> None:
        async with self._lock:
            self.audit_log.append(audit_record.copy())

    async def get_recent_flagged_transactions(self, limit: int = 20) -> List[Dict[str, Any]]:
        async with self._lock:
            flagged = [rec for rec in self.audit_log if rec.get("is_flagged")]
            return list(reversed(flagged))[:limit]

    async def get_audit_record_by_tx(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            for rec in reversed(self.audit_log):
                if rec.get("transaction_id") == transaction_id:
                    return rec.copy()
            return None

    async def close(self) -> None:
        async with self._lock:
            self.raw_transactions.clear()
            self.engineered_features.clear()
            self.audit_log.clear()
        logger.info("Closed In-Memory storage operator.")
