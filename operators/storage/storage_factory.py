"""Storage Operator Factory.

Instantiates the designated storage backend:
- 'postgres' -> PostgresStorageOperator (production)
- 'sqlite'   -> SqliteStorageOperator (local persistent developer default)
- 'memory'   -> MemoryStorageOperator (ultra-fast tests)
"""

import os
from core.interfaces import BaseStorageOperator
from operators.storage.memory_operator import MemoryStorageOperator
from operators.storage.sqlite_operator import SqliteStorageOperator
from operators.storage.postgres_operator import PostgresStorageOperator
from core.logger import get_logger

logger = get_logger("storage_factory")

def get_storage_operator(backend: str = None) -> BaseStorageOperator:
    """Factory method to get the configured storage operator."""
    selected_backend = (backend or os.getenv("STORAGE_BACKEND", "sqlite")).lower().strip()
    
    logger.info("Initializing storage operator: {backend}", backend=selected_backend)
    
    if selected_backend == "postgres":
        return PostgresStorageOperator()
    elif selected_backend == "sqlite":
        return SqliteStorageOperator()
    elif selected_backend == "memory":
        return MemoryStorageOperator()
    else:
        logger.warning(
            "Unknown storage backend '{backend}'. Falling back to 'sqlite'.",
            backend=selected_backend
        )
        return SqliteStorageOperator()
