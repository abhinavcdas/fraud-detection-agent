"""FastAPI Production Serving Service for Real-Time Fraud Scoring & LLM Investigation.

Exposes high-throughput enterprise endpoints for:
1. POST /score: Hot-Path (< 50ms) authorization running:
   - Redis Sliding-Window Feature Retrieval
   - Pre-ML Deterministic Hard Rules Engine (Sanctions, Caps, Killswitches)
   - Ultra-Fast ONNX/XGBoost Machine Learning Inference
   - Asynchronous Cold-Path Entity Resolution & FinCEN SAR Generation for flagged events
2. POST /rules/evaluate: Direct deterministic rule evaluation.
3. GET /graph/mule-ring/{customer_id}: Entity Resolution graph analysis for money mule rings.
4. GET /benchmarks/latency: Live latency profiling metrics scorecard.
5. GET /transactions/flagged: Recent suspicious events for operations console.
6. GET /audit-log/{transaction_id}: Reconstructs past decisions for Model Risk compliance.
"""

import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.logger import get_logger, bind_tx_context
from core.rules_engine import DeterministicRulesEngine
from features.redis_store import RedisFeatureStore
from graph.entity_graph import entity_graph
from operators.storage.storage_factory import get_storage_operator
from operators.scoring.scoring_factory import get_scorer_operator
from agent.agent_loop import run_investigation
from governance.audit import build_audit_record
from model.shap_explain import load_champion_model, get_top_contributing_features

load_dotenv()
logger = get_logger("api_server")

FRAUD_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.38"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "fraud-xgb-v1")

# Global instances initialized during lifespan
storage_op = None
scorer_op = None
champion_model = None
rules_engine = None
feature_store = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and clean resource shutdown."""
    global storage_op, scorer_op, champion_model, rules_engine, feature_store
    logger.info("Initializing API application services...")

    storage_backend = os.getenv("STORAGE_BACKEND", "sqlite")
    storage_op = get_storage_operator(storage_backend)
    await storage_op.initialize()

    scorer_backend = os.getenv("SCORER_BACKEND", "onnx")
    scorer_op = get_scorer_operator(scorer_backend)

    rules_engine = DeterministicRulesEngine()
    feature_store = RedisFeatureStore()

    try:
        champion_model = load_champion_model()
    except Exception as e:
        logger.warning("Could not pre-load champion model for SHAP: {err}", err=str(e))

    logger.info("API services initialized | Storage={st} | Scorer={sc} | Threshold={th}",
                st=storage_backend, sc=scorer_backend, th=FRAUD_THRESHOLD)
    yield

    logger.info("Shutting down API services...")
    if storage_op:
        await storage_op.close()
    if feature_store:
        await feature_store.close()


app = FastAPI(
    title="Real-Time Fraud Prevention Engine",
    description="Production-grade dual-path streaming transaction scoring with low-latency Redis feature caching, deterministic hard rules, ONNX inference, and Groq agent forensic review.",
    version="1.1.0",
    lifespan=lifespan
)


class TransactionPayload(BaseModel):
    transaction_id: str = Field(..., json_schema_extra={"example": "TX_109283"})
    customer_id: str = Field(..., json_schema_extra={"example": "CUST_0042"})
    merchant_id: Optional[str] = Field("MERCH_001", json_schema_extra={"example": "MERCH_012"})
    amount: float = Field(..., json_schema_extra={"example": 1250.00})
    country: Optional[str] = Field("US", json_schema_extra={"example": "US"})
    card_number: Optional[str] = Field("424242424242", json_schema_extra={"example": "4242424242421234"})
    device_id: Optional[str] = Field("DEV_DEFAULT", json_schema_extra={"example": "DEV_FARM_01"})
    ip_address: Optional[str] = Field("127.0.0.1", json_schema_extra={"example": "198.51.100.77"})
    lat: Optional[float] = Field(0.0, json_schema_extra={"example": 40.7128})
    lon: Optional[float] = Field(0.0, json_schema_extra={"example": -74.0060})
    velocity_5m: Optional[int] = Field(None, json_schema_extra={"example": 4})
    velocity_60m: Optional[int] = Field(None, json_schema_extra={"example": 7})
    amount_deviation: Optional[float] = Field(None, json_schema_extra={"example": 3.8})
    time_since_last_tx_sec: Optional[float] = Field(None, json_schema_extra={"example": 45.0})
    geo_distance_km: Optional[float] = Field(None, json_schema_extra={"example": 650.0})
    time_step: Optional[float] = Field(0.0, json_schema_extra={"example": 12890.0})

    # PCA anonymized features V1-V28
    v1: Optional[float] = 0.0
    v2: Optional[float] = 0.0
    v3: Optional[float] = 0.0
    v4: Optional[float] = 0.0
    v5: Optional[float] = 0.0
    v6: Optional[float] = 0.0
    v7: Optional[float] = 0.0
    v8: Optional[float] = 0.0
    v9: Optional[float] = 0.0
    v10: Optional[float] = 0.0
    v11: Optional[float] = 0.0
    v12: Optional[float] = 0.0
    v13: Optional[float] = 0.0
    v14: Optional[float] = 0.0
    v15: Optional[float] = 0.0
    v16: Optional[float] = 0.0
    v17: Optional[float] = 0.0
    v18: Optional[float] = 0.0
    v19: Optional[float] = 0.0
    v20: Optional[float] = 0.0
    v21: Optional[float] = 0.0
    v22: Optional[float] = 0.0
    v23: Optional[float] = 0.0
    v24: Optional[float] = 0.0
    v25: Optional[float] = 0.0
    v26: Optional[float] = 0.0
    v27: Optional[float] = 0.0
    v28: Optional[float] = 0.0


class ScoreResponse(BaseModel):
    transaction_id: str
    fraud_score: float
    is_flagged: bool
    threshold: float
    model_version: str
    action: str
    pipeline_stage: str
    rule_decision: Optional[Dict[str, Any]] = None
    sliding_window_features: Optional[Dict[str, Any]] = None
    mule_ring_analysis: Optional[Dict[str, Any]] = None
    investigation_dossier: Optional[Dict[str, Any]] = None
    top_contributing_factors: Optional[List[Dict[str, Any]]] = None
    input_hash: str
    latency_ms: float


@app.get("/health")
def health_check():
    """Service liveness and readiness probe."""
    return {
        "status": "healthy",
        "service": "fraud-detection-api",
        "version": "2.0.0",
        "model_version": MODEL_VERSION,
        "operational_threshold": FRAUD_THRESHOLD,
        "storage_backend": os.getenv("STORAGE_BACKEND", "sqlite"),
        "scorer_backend": os.getenv("SCORER_BACKEND", "onnx"),
        "llm_provider": os.getenv("LLM_PROVIDER", "groq")
    }


@app.post("/score", response_model=ScoreResponse)
async def score_transaction(payload: TransactionPayload):
    """Dual-Path real-time transaction authorization and forensic triage:
    
    HOT PATH (< 50ms):
    1. Query Redis Sliding-Window Feature Store for dynamic velocity & impossible travel delta.
    2. Evaluate Deterministic Hard Rules Engine (sanctions, caps, killswitches).
       - If BLOCK: Short-circuit immediately with 0 ML latency.
    3. Run Ultra-Fast ML Inference (ONNX Runtime / XGBoost).

    COLD PATH (Asynchronous / Decoupled):
    4. If flagged or STEP_UP:
       - Run Entity Resolution Graph Mining for money mule rings.
       - Compile LLM forensic dossier & FinCEN-compliant SAR narrative.
    5. Asynchronously persist audit trail and record transaction in feature store.
    """
    start_time = time.time()
    tx = payload.model_dump()
    tx_id = tx["transaction_id"]
    cust_id = tx["customer_id"]

    tx_logger = bind_tx_context(logger, tx_id, cust_id)
    tx_logger.debug("Received scoring request | amount=${amt}", amt=tx["amount"])

    active_scorer = scorer_op or get_scorer_operator("onnx")
    active_storage = storage_op or get_storage_operator("sqlite")
    active_rules = rules_engine or DeterministicRulesEngine()
    active_feat_store = feature_store or RedisFeatureStore()

    # Hot Path 1: Sliding Window Features from Feature Store
    now_ts = time.time()
    sliding_feats = await active_feat_store.get_sliding_window_features(
        cust_id,
        now_ts,
        tx.get("lat"),
        tx.get("lon")
    )

    # Enrich tx with feature store metrics if not explicitly supplied
    if tx.get("velocity_5m") is None:
        tx["velocity_5m"] = sliding_feats.get("velocity_5m", 0)
    if tx.get("geo_distance_km") is None:
        tx["geo_distance_km"] = sliding_feats.get("geo_distance_km", 0.0)
    if tx.get("amount_deviation") is None:
        tx["amount_deviation"] = sliding_feats.get("amount_deviation", 0.0)

    # Hot Path 2: Pre-ML Deterministic Hard Rule Engine
    rule_res = active_rules.evaluate_rules(tx, sliding_feats)

    # Early exit on deterministic BLOCK
    if rule_res["action"] == "BLOCK":
        latency_ms = round((time.time() - start_time) * 1000, 2)
        tx_logger.warning("Transaction rejected by deterministic rule: {r}", r=rule_res["reason"])

        agent_res_block = {
            "report": {
                "risk_level": "CRITICAL",
                "recommendation": "DECLINE",
                "summary": rule_res["reason"],
                "evidence": rule_res["triggered_rules"],
                "cited_facts": rule_res["triggered_rules"]
            }
        }
        audit_record = build_audit_record(
            transaction=tx,
            fraud_score=1.0,
            is_flagged=True,
            model_version="rule-engine-v1",
            agent_result=agent_res_block,
            latency_ms=latency_ms
        )
        try:
            await active_storage.save_audit_log(audit_record)
        except Exception as e:
            tx_logger.error("Failed to write audit log: {err}", err=str(e))

        # Record event in background
        await active_feat_store.record_event(
            cust_id, now_ts, tx["amount"], tx.get("ip_address"), tx.get("lat"), tx.get("lon")
        )

        return ScoreResponse(
            transaction_id=tx_id,
            fraud_score=1.0,
            is_flagged=True,
            threshold=FRAUD_THRESHOLD,
            model_version="deterministic-rules-v1",
            action="DECLINE",
            pipeline_stage="DETERMINISTIC_HARD_RULE_BLOCK",
            rule_decision=rule_res,
            sliding_window_features=sliding_feats,
            mule_ring_analysis=None,
            investigation_dossier=agent_res_block,
            top_contributing_factors=None,
            input_hash=audit_record["input_hash"],
            latency_ms=latency_ms
        )


    # Hot Path 3: Ultra-Fast ML Inference (ONNX / XGBoost)
    try:
        fraud_score = active_scorer.predict_proba(tx)
    except Exception as exc:
        tx_logger.error("Scoring error: {err}. Proceeding with fallback.", err=str(exc))
        fraud_score = 0.50

    is_flagged = bool(fraud_score >= FRAUD_THRESHOLD or rule_res["action"] == "STEP_UP")
    action = "INVESTIGATE" if is_flagged else "APPROVE"

    agent_result = None
    top_factors = None
    mule_analysis = None

    # Cold Path 4: Asynchronous Forensic & Graph Investigation for Flagged Events
    if is_flagged:
        tx_logger.info("Transaction flagged ({s:.3f} >= {th:.2f}). Running graph & agent triage.",
                       s=fraud_score, th=FRAUD_THRESHOLD)

        # Entity graph resolution
        entity_graph.add_transaction_entities(
            cust_id,
            device_id=tx.get("device_id"),
            ip_address=tx.get("ip_address"),
            card_id=tx.get("card_number")
        )
        mule_analysis = entity_graph.analyze_customer_syndicate(cust_id)

        # Agent investigation with FinCEN SAR generation
        agent_result = run_investigation(tx)

        if champion_model:
            try:
                top_factors = get_top_contributing_features(champion_model, tx, top_k=5)
            except Exception as e:
                tx_logger.warning("SHAP feature extraction error: {err}", err=str(e))

    latency_ms = round((time.time() - start_time) * 1000, 2)

    # Hot Path 5: Audit Log & Feature Store Update
    audit_record = build_audit_record(
        transaction=tx,
        fraud_score=fraud_score,
        is_flagged=is_flagged,
        model_version=active_scorer.get_model_version(),
        agent_result=agent_result,
        latency_ms=latency_ms
    )

    try:
        await active_storage.save_audit_log(audit_record)
    except Exception as e:
        tx_logger.error("Failed to write audit log: {err}", err=str(e))

    # Record event in sliding window
    await active_feat_store.record_event(
        cust_id, now_ts, tx["amount"], tx.get("ip_address"), tx.get("lat"), tx.get("lon")
    )

    return ScoreResponse(
        transaction_id=tx_id,
        fraud_score=round(fraud_score, 4),
        is_flagged=is_flagged,
        threshold=FRAUD_THRESHOLD,
        model_version=active_scorer.get_model_version(),
        action=action,
        pipeline_stage="DUAL_PATH_HOT_COLD_INSPECTION",
        rule_decision=rule_res,
        sliding_window_features=sliding_feats,
        mule_ring_analysis=mule_analysis,
        investigation_dossier=agent_result,
        top_contributing_factors=top_factors,
        input_hash=audit_record["input_hash"],
        latency_ms=latency_ms
    )


@app.post("/rules/evaluate")
def evaluate_rules_endpoint(payload: Dict[str, Any]):
    """Direct deterministic rules evaluation endpoint."""
    engine = rules_engine or DeterministicRulesEngine()
    return engine.evaluate_rules(payload)


@app.get("/graph/mule-ring/{customer_id}")
def get_mule_ring_graph(customer_id: str):
    """Retrieve multi-entity ego-network and mule ring clustering for a customer."""
    syndicate = entity_graph.analyze_customer_syndicate(customer_id)
    subgraph_data = entity_graph.get_cluster_subgraph_data(customer_id)
    return {
        "analysis": syndicate,
        "graph_data": subgraph_data
    }


@app.get("/benchmarks/latency")
def get_latency_benchmark():
    """Retrieve latest automated latency profiler scorecard."""
    benchmark_path = Path(__file__).resolve().parent.parent / "reports" / "latency_benchmark.md"
    if benchmark_path.exists():
        with open(benchmark_path, "r", encoding="utf-8") as f:
            return {"scorecard_markdown": f.read()}
    return {"message": "Benchmark not run yet. Run python benchmarks/latency_profiler.py to generate."}


@app.get("/transactions/flagged")
async def get_flagged_transactions(limit: int = Query(20, ge=1, le=100)):
    """Retrieve recent flagged transactions for the operations dashboard."""
    active_storage = storage_op or get_storage_operator("sqlite")
    records = await active_storage.get_recent_flagged_transactions(limit=limit)

    cleaned = []
    for r in records:
        item = dict(r)
        for field in ("agent_report_json", "agent_report"):
            if field in item and isinstance(item[field], str):
                try:
                    item[field] = json.loads(item[field])
                except Exception:
                    pass
        cleaned.append(item)

    return {
        "flagged_count": len(cleaned),
        "limit": limit,
        "transactions": cleaned
    }


@app.get("/audit-log/{transaction_id}")
async def get_audit_trail(transaction_id: str):
    """Reconstruct an immutable decision record by transaction ID for regulatory examination."""
    active_storage = storage_op or get_storage_operator("sqlite")
    record = await active_storage.get_audit_record_by_tx(transaction_id)

    if not record:
        raise HTTPException(
            status_code=404,
            detail=f"Audit record not found for transaction ID: {transaction_id}"
        )

    out = dict(record)
    for field in ("agent_report_json", "agent_report"):
        if field in out and isinstance(out[field], str):
            try:
                out[field] = json.loads(out[field])
            except Exception:
                pass

    return out


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", "8000"))
    logger.info("Starting FastAPI service on port {port}...", port=port)
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=False)
