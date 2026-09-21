"""Storage Operator Factory.

Instantiates the designated storage backend:
- 'postgres' -> PostgresStorageOperator (production)
- 'sqlite'   -> SqliteStorageOperator (local persistent developer default)
- 'memory'   -> MemoryStorageOperator (ultra-fast tests)
"""

import os
import threading
from typing import Dict
from core.interfaces import BaseStorageOperator
from operators.storage.memory_operator import MemoryStorageOperator
from operators.storage.sqlite_operator import SqliteStorageOperator
from operators.storage.postgres_operator import PostgresStorageOperator
from core.logger import get_logger

logger = get_logger("storage_factory")

_storage_instances: Dict[str, BaseStorageOperator] = {}
_lock = threading.Lock()

def get_storage_operator(backend: str = None) -> BaseStorageOperator:
    """Factory method to get the configured storage operator (thread-safe singleton)."""
    selected_backend = (backend or os.getenv("STORAGE_BACKEND", "sqlite")).lower().strip()
    
    with _lock:
        if selected_backend in _storage_instances:
            return _storage_instances[selected_backend]
            
        logger.info("Initializing storage operator: {backend}", backend=selected_backend)
        
        if selected_backend == "postgres":
            operator = PostgresStorageOperator()
        elif selected_backend == "sqlite":
            operator = SqliteStorageOperator()
        elif selected_backend == "memory":
            operator = MemoryStorageOperator()
        else:
            logger.warning(
                "Unknown storage backend '{backend}'. Falling back to 'sqlite'.",
                backend=selected_backend
            )
            operator = SqliteStorageOperator()
            selected_backend = "sqlite"
            
        _storage_instances[selected_backend] = operator
        return operator

def reset_storage_operators() -> None:
    """Reset cached singleton storage operators (used mainly in testing)."""
    with _lock:
        _storage_instances.clear()

