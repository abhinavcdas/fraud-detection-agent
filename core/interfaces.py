"""Abstract Base Classes (ABCs) defining the Operator Strategy Contracts."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncGenerator

class BaseStorageOperator(ABC):
    """Contract for data persistence backends (PostgreSQL, SQLite, In-Memory)."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize connection pool and verify/create required tables."""
        pass

    @abstractmethod
    async def save_raw_transaction(self, tx: Dict[str, Any]) -> None:
        """Persist ingested raw transaction."""
        pass

    @abstractmethod
    async def save_engineered_features(self, features: Dict[str, Any]) -> None:
        """Persist computed rolling features."""
        pass

    @abstractmethod
    async def get_customer_history(self, customer_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent historical transactions for rolling calculation and investigation."""
        pass

    @abstractmethod
    async def save_audit_log(self, audit_record: Dict[str, Any]) -> None:
        """Persist immutable audit trail for Model Risk Management."""
        pass

    @abstractmethod
    async def get_recent_flagged_transactions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve flagged transactions for analyst dashboard."""
        pass

    @abstractmethod
    async def get_audit_record_by_tx(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve audit record for a given transaction ID to reconstruct decisions."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close connection pools cleanly."""
        pass


class BaseStreamOperator(ABC):
    """Contract for streaming event buses (Kafka/Redpanda, Async Memory Queue)."""

    @abstractmethod
    async def start(self) -> None:
        """Initialize producer and consumer clients."""
        pass

    @abstractmethod
    async def publish(self, topic: str, message: Dict[str, Any]) -> None:
        """Publish an event to the message broker."""
        pass

    @abstractmethod
    async def consume(self, topic: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Consume stream of events asynchronously."""
        yield {} # Type stub

    @abstractmethod
    async def close(self) -> None:
        """Gracefully disconnect and flush buffers."""
        pass


class BaseLLMOperator(ABC):
    """Contract for LLM investigation providers (Groq, OpenAI, Mock)."""

    @abstractmethod
    async def investigate(self, transaction: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct investigation with tool execution, returning structured report & guardrails."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Verify API connectivity and authentication."""
        pass


class BaseScorerOperator(ABC):
    """Contract for machine learning inference operators (XGBoost, LightGBM, Heuristic)."""

    @abstractmethod
    def predict_proba(self, features: Dict[str, Any]) -> float:
        """Compute fraud probability score in range [0.0, 1.0]."""
        pass

    def score_transaction(self, features: Dict[str, Any]) -> float:
        """Convenience alias for predict_proba."""
        return self.predict_proba(features)

    @abstractmethod
    def get_model_version(self) -> str:
        """Return registered model version identifier."""
        pass


class BaseFeatureOperator(ABC):
    """Contract for feature engineering transformers."""

    @abstractmethod
    def extract_features(self, current_tx: Dict[str, Any], customer_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute rolling and behavioral features given current event and customer history."""
        pass


class BaseRulesEngineOperator(ABC):
    """Contract for pre-ML deterministic hard rules engines (Sanctions, Caps, Killswitches)."""

    @abstractmethod
    def evaluate_rules(self, tx: Dict[str, Any], features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evaluate deterministic rules prior to machine learning scoring.
        
        Returns:
            Dict containing:
                - action: "BLOCK" | "STEP_UP" | "PASS"
                - passed: bool (True if PASS, False if BLOCK or STEP_UP)
                - triggered_rules: List[str] rule identifiers
                - reason: Descriptive explanation for audit logs
        """
        pass


class BaseFeatureStoreOperator(ABC):
    """Contract for low-latency in-memory sliding-window feature stores (Redis, Memory)."""

    @abstractmethod
    async def record_event(
        self,
        customer_id: str,
        timestamp: float,
        amount: float,
        ip: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None
    ) -> None:
        """Record transaction timestamp and spatial coordinates in in-memory sliding window."""
        pass

    @abstractmethod
    async def get_sliding_window_features(
        self,
        customer_id: str,
        current_timestamp: float,
        current_lat: Optional[float] = None,
        current_lon: Optional[float] = None
    ) -> Dict[str, Any]:
        """Retrieve real-time rolling counts (10s, 60s, 5m) and impossible travel speed."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close connection pools cleanly."""
        pass

