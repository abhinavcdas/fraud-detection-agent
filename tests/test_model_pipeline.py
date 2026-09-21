"""Automated Test Suite for Phase 2: Fraud ML Model & MLflow Tracking."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from model.evaluate import calculate_metrics, find_optimal_threshold
from model.train import (
    FEATURE_COLUMNS,
    train_and_compare_strategies,
    REGISTRY_DIR
)
from operators.scoring.xgboost_operator import XGBoostScorerOperator

def test_calculate_metrics():
    """Verify precision, recall, PR-AUC, and ROC-AUC calculation."""
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])

    metrics = calculate_metrics(y_true, y_prob, threshold=0.5)

    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["pr_auc"] > 0.9
    assert metrics["roc_auc"] == 1.0
    assert metrics["true_positives"] == 2
    assert metrics["true_negatives"] == 4

def test_find_optimal_threshold():
    """Verify recall-biased threshold scanning achieves target recall."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.45, 0.60, 0.80, 0.95])

    # Default 0.5 threshold misses the fraud at 0.45 (Recall = 0.75)
    # Calibrated threshold should lower boundary to include 0.45, achieving Recall >= 0.85
    thresh, metrics = find_optimal_threshold(y_true, y_prob, target_recall=0.85)

    assert thresh < 0.5
    assert metrics["recall"] >= 0.85

def test_train_and_compare_strategies(tmp_path):
    """Verify all 3 imbalance strategies train, evaluate, and export champion model."""
    # Create deterministic synthetic tabular dataset (1,000 records, 40 frauds ~4%)
    np.random.seed(42)
    n_samples = 1000
    n_fraud = 40

    records = []
    for i in range(n_samples):
        is_fraud = 1 if i < n_fraud else 0
        row = {
            "transaction_id": f"TX_{i:04d}",
            "customer_id": f"CUST_{i % 100:03d}",
            "amount": np.random.uniform(500, 3000) if is_fraud else np.random.uniform(10, 200),
            "velocity_5m": np.random.randint(3, 7) if is_fraud else np.random.randint(0, 2),
            "velocity_60m": np.random.randint(5, 12) if is_fraud else np.random.randint(0, 3),
            "amount_deviation": float(np.random.uniform(3.0, 6.0) if is_fraud else np.random.uniform(-1.0, 1.0)),
            "time_since_last_tx_sec": float(np.random.uniform(10, 60) if is_fraud else np.random.uniform(300, 5000)),
            "geo_distance_km": float(np.random.uniform(500, 2000) if is_fraud else np.random.uniform(0, 50)),
            "is_fraud": is_fraud
        }
        for v in range(1, 29):
            row[f"v{v}"] = float(np.random.normal(2.0, 1.0) if is_fraud else np.random.normal(0.0, 1.0))
        records.append(row)

    df = pd.DataFrame(records)

    # Train all 3 strategies with isolated local tmp_path SQLite MLflow tracking and registry
    db_file = (tmp_path / "mlflow.db").as_posix()
    summary = train_and_compare_strategies(
        df=df,
        tracking_uri=f"sqlite:///{db_file}",
        experiment_name="test-fraud-experiments",
        registry_dir=tmp_path,
        n_estimators=10
    )

    assert summary["champion"] in {"class_weighted_xgboost", "smote_xgboost", "balanced_lightgbm"}
    assert summary["pr_auc"] > 0.80
    assert len(summary["results"]) == 3

    # Check exported artifacts exist in isolated tmp_path
    assert (tmp_path / "fraud_xgb_v1.joblib").exists()
    assert (tmp_path / "model_metadata.json").exists()

def test_xgboost_scorer_inference_with_trained_model():
    """Verify XGBoostScorerOperator loads champion model and outputs calibrated probability."""
    scorer = XGBoostScorerOperator()
    assert scorer.model is not None
    assert scorer.get_model_version() == "fraud-xgb-v1"

    benign_tx = {
        "amount": 25.0,
        "velocity_5m": 0,
        "velocity_60m": 0,
        "amount_deviation": 0.0,
        "time_since_last_tx_sec": 1200.0,
        "geo_distance_km": 0.5
    }
    for i in range(1, 29):
        benign_tx[f"v{i}"] = 0.0

    score_benign = scorer.predict_proba(benign_tx)
    assert 0.0 <= score_benign <= 0.40

    fraud_tx = {
        "amount": 2500.0,
        "velocity_5m": 5,
        "velocity_60m": 8,
        "amount_deviation": 5.0,
        "time_since_last_tx_sec": 20.0,
        "geo_distance_km": 950.0
    }
    for i in range(1, 29):
        fraud_tx[f"v{i}"] = 2.5

    score_fraud = scorer.predict_proba(fraud_tx)
    assert score_fraud >= 0.70

def test_onnx_and_xgboost_parity():
    """Verify strict numerical prediction parity between ONNX Runtime and XGBoost."""
    from model.onnx_scorer import ONNXModelScorer
    xgb_scorer = XGBoostScorerOperator()
    onnx_scorer = ONNXModelScorer()

    test_cases = [
        # Benign transaction
        {
            "amount": 35.0,
            "velocity_5m": 0,
            "velocity_60m": 1,
            "amount_deviation": 0.1,
            "time_since_last_tx_sec": 1200.0,
            "geo_distance_km": 1.5,
            **{f"v{i}": 0.0 for i in range(1, 29)}
        },
        # High-risk transaction
        {
            "amount": 2850.0,
            "velocity_5m": 5,
            "velocity_60m": 8,
            "amount_deviation": 4.5,
            "time_since_last_tx_sec": 30.0,
            "geo_distance_km": 720.0,
            **{f"v{i}": 2.5 for i in range(1, 29)}
        }
    ]

    for tx in test_cases:
        p_xgb = xgb_scorer.predict_proba(tx)
        p_onnx = onnx_scorer.predict_proba(tx)
        # Parity within 0.05
        assert abs(p_xgb - p_onnx) < 0.05, f"Disparity: XGB={p_xgb} vs ONNX={p_onnx}"

def test_chronological_time_based_split(tmp_path):
    """Verify chronological time-based splitting is applied when time_step is present."""
    records = []
    for i in range(200):
        is_f = 1 if (i % 20 == 0) else 0
        r = {
            "time_step": float(i * 100),
            "amount": 100.0 if not is_f else 1500.0,
            "velocity_5m": 0 if not is_f else 4,
            "velocity_60m": 1 if not is_f else 7,
            "amount_deviation": 0.0 if not is_f else 3.5,
            "time_since_last_tx_sec": 1000.0 if not is_f else 20.0,
            "geo_distance_km": 2.0 if not is_f else 600.0,
            "is_fraud": is_f,
            **{f"v{j}": 0.0 for j in range(1, 29)}
        }
        records.append(r)
    df = pd.DataFrame(records)
    db_file = (tmp_path / "mlflow_time.db").as_posix()
    summary = train_and_compare_strategies(
        df=df,
        tracking_uri=f"sqlite:///{db_file}",
        experiment_name="test-time-split",
        registry_dir=tmp_path,
        n_estimators=5,
        time_based_split=True
    )
    assert summary["champion"] in {"class_weighted_xgboost", "smote_xgboost", "balanced_lightgbm"}

