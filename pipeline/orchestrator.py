"""Asynchronous Fraud Pipeline Orchestrator.

Orchestrates the lifecycle of transactions:
Stream -> Feature Extraction -> Deterministic Rules -> ML Scoring -> LLM Investigation -> Audit Logging.

Features:
- Complete component decoupling via Strategy Pattern.
- Deterministic rules evaluated first: immediate BLOCK on hard constraints.
- Total failure isolation: LLM network glitches or external API errors will NOT crash consumer.
- Dead-Letter Queue (DLQ) routing for poisoned messages.
- High-throughput asynchronous processing with concurrent storage persistence and structured Loguru observability.
"""

import os
import time
import asyncio
import hashlib
from typing import Dict, Any, Optional
from core.interfaces import (
    BaseStorageOperator,
    BaseStreamOperator,
    BaseLLMOperator,
    BaseScorerOperator,
    BaseFeatureOperator
)
from core.logger import get_logger, bind_tx_context
from core.resilience import global_dlq, isolated_async_execution
from core.rules_engine import DeterministicRulesEngine, RuleAction
from operators.storage.storage_factory import get_storage_operator
from operators.stream.stream_factory import get_stream_operator
from operators.llm.llm_factory import get_llm_operator
from operators.scoring.scoring_factory import get_scoring_operator
from operators.features.rolling_feature_operator import RollingFeatureOperator

logger = get_logger("pipeline_orchestrator")

FRAUD_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.38"))

class FraudPipelineOrchestrator:
    """Enterprise Pipeline Orchestrator."""

    def __init__(
        self,
        storage: Optional[BaseStorageOperator] = None,
        stream: Optional[BaseStreamOperator] = None,
        llm: Optional[BaseLLMOperator] = None,
        scorer: Optional[BaseScorerOperator] = None,
        feature_extractor: Optional[BaseFeatureOperator] = None,
        rules_engine: Optional[DeterministicRulesEngine] = None,
        fraud_threshold: float = FRAUD_THRESHOLD
    ):
        self.storage = storage or get_storage_operator()
        self.stream = stream or get_stream_operator()
        self.llm = llm or get_llm_operator()
        self.scorer = scorer or get_scoring_operator()
        self.feature_extractor = feature_extractor or RollingFeatureOperator()
        self.rules_engine = rules_engine or DeterministicRulesEngine()
        self.fraud_threshold = fraud_threshold
        self._running = False

    async def initialize(self) -> None:
        """Initialize all underlying operators."""
        logger.info("Initializing Fraud Pipeline Orchestrator operators...")
        await self.storage.initialize()
        await self.stream.start()
        logger.info("Pipeline Orchestrator initialized successfully.")

    def _fallback_investigation(self, exc: Exception, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Degraded fallback when LLM API call fails, preventing consumer crash."""
        tx = kwargs.get("transaction", {})
        logger.error("LLM Operator failed ({err}). Generating degraded fallback audit report.", err=str(exc))
        return {
            "report": {
                "risk_level": "HIGH",
                "recommendation": "ESCALATE",
                "summary": "Automated system escalation: LLM agent unavailable during investigation.",
                "evidence": ["System anomaly: Primary LLM provider failed, manual inspection mandated."],
                "cited_facts": []
            },
            "guardrails": {
                "status": "NEEDS_REVIEW",
                "faithfulness_score": 0.0,
                "reason": "Degraded fallback mode active."
            },
            "latency_ms": 0.0,
            "provider": "degraded_fallback"
        }

    async def _safe_investigate(self, transaction: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute LLM investigation inside an isolated error boundary."""
        try:
            return await self.llm.investigate(transaction, context)
        except Exception as e:
            return self._fallback_investigation(e, {"transaction": transaction})

    async def process_transaction(self, raw_tx: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single transaction through the full pipeline with failure isolation."""
        start_time = time.time()
        tx_id = raw_tx.get("transaction_id")
        cust_id = raw_tx.get("customer_id")

        if not tx_id or not cust_id:
            raise ValueError("Malformed transaction payload: missing transaction_id or customer_id")

        tx_logger = bind_tx_context(logger, tx_id, cust_id)
        tx_logger.debug("Ingesting transaction into pipeline | amount=${amt}", amt=raw_tx.get("amount"))

        # 1. Fetch customer history
        history = await self.storage.get_customer_history(cust_id, limit=50)

        # 2. Extract engineered features
        features = self.feature_extractor.extract_features(raw_tx, history)

        # 3. Persist raw transaction & computed features concurrently
        await asyncio.gather(
            self.storage.save_raw_transaction(raw_tx),
            self.storage.save_engineered_features(features)
        )

        # 4. Evaluate deterministic rules first
        rule_decision = self.rules_engine.evaluate_rules(raw_tx, features)
        rule_action = rule_decision.get("action", "PASS")
        rule_blocked = (rule_action == RuleAction.BLOCK or rule_action == "BLOCK")

        # 5. Score transaction using ML Scorer
        fraud_score = self.scorer.predict_proba(features)
        if rule_blocked:
            fraud_score = max(fraud_score, 1.0)
            is_flagged = True
        else:
            is_flagged = fraud_score >= self.fraud_threshold

        model_ver = self.scorer.get_model_version()

        tx_logger.info(
            "Transaction scored | score={score:.4f} | flagged={flagged} | rule_action={rule_act} | model={ver}",
            score=fraud_score,
            flagged=is_flagged,
            rule_act=rule_action,
            ver=model_ver
        )

        investigation_result = None
        if is_flagged:
            tx_logger.warning(
                "Transaction exceeded threshold ({thresh}) or blocked by rule. Dispatching LLM agent...",
                thresh=self.fraud_threshold
            )
            investigation_result = await self._safe_investigate(
                transaction=features,
                context={
                    "customer_history_count": len(history),
                    "rule_decision": rule_decision
                }
            )

        # 6. Persist immutable audit log record
        input_hash = hashlib.sha256(str(raw_tx).encode("utf-8")).hexdigest()[:16]
        agent_decision = "AUTO_PASS"
        if investigation_result:
            agent_decision = investigation_result["report"].get("recommendation", "MANUAL_REVIEW")
        elif rule_blocked:
            agent_decision = "DECLINE"

        audit_entry = {
            "transaction_id": tx_id,
            "timestamp": raw_tx.get("timestamp"),
            "input_hash": input_hash,
            "model_version": model_ver,
            "fraud_score": fraud_score,
            "is_flagged": is_flagged,
            "agent_decision": agent_decision,
            "agent_report": investigation_result["report"] if investigation_result else None,
            "guardrail_status": investigation_result["guardrails"].get("status") if investigation_result else "SKIPPED",
            "faithfulness_score": investigation_result["guardrails"].get("faithfulness_score") if investigation_result else None,
            "latency_ms": round((time.time() - start_time) * 1000, 2)
        }
        await self.storage.save_audit_log(audit_entry)

        return {
            "transaction_id": tx_id,
            "fraud_score": fraud_score,
            "is_flagged": is_flagged,
            "rule_decision": rule_decision,
            "investigation": investigation_result,
            "total_latency_ms": audit_entry["latency_ms"]
        }


    async def run_stream(self, topic: str = "transactions", max_events: Optional[int] = None) -> int:
        """Run continuous streaming consumer loop with DLQ protection."""
        self._running = True
        logger.info("Pipeline stream consumer active on topic '{topic}'", topic=topic)
        count = 0

        async for message in self.stream.consume(topic):
            if not self._running:
                break
            try:
                await self.process_transaction(message)
                count += 1
                if max_events and count >= max_events:
                    logger.info("Reached target max events ({max_events}). Halting consumer.", max_events=max_events)
                    break
            except Exception as exc:
                # Capture poison pills into DLQ without killing consumer loop!
                global_dlq.route_to_dlq(payload=message, error=exc, context={"topic": topic, "count": count})
                logger.error("Poisoned message caught and isolated to DLQ. Continuing stream.")

        return count

    async def close(self) -> None:
        """Gracefully close stream and storage connections."""
        self._running = False
        await self.stream.close()
        await self.storage.close()
        logger.info("Fraud Pipeline Orchestrator shut down cleanly.")
