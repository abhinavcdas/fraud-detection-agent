"""Model Training Pipeline comparing 3 Imbalance Mitigation Strategies.

Imbalance Strategies:
1. Class-Weighted XGBoost (scale_pos_weight)
2. SMOTE Oversampling + XGBoost (imbalanced-learn)
3. Balanced/Focal Loss LightGBM

Evaluation:
- Primary metric: PR-AUC (Average Precision)
- Secondary metrics: ROC-AUC, Recall, Precision, F1, Confusion Matrix
- MLflow Experiment Tracking & Model Registry registration as 'fraud-xgb-v1'
"""

import os
import sys
import json
import joblib
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from dotenv import load_dotenv

from core.logger import get_logger
from consumer.feature_engineering import compute_rolling_features
from producer.stream_producer import generate_synthetic_metadata
from model.evaluate import calculate_metrics, find_optimal_threshold

load_dotenv()
logger = get_logger("model_train")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_CSV = DATA_DIR / "raw" / "creditcard.csv"
PROCESSED_DIR = DATA_DIR / "processed"
ENGINEERED_PARQUET = PROCESSED_DIR / "engineered_features.parquet"
REGISTRY_DIR = Path(__file__).resolve().parent / "registry"

FEATURE_COLUMNS = [
    *[f"v{i}" for i in range(1, 29)],
    "amount",
    "velocity_5m",
    "velocity_60m",
    "amount_deviation",
    "time_since_last_tx_sec",
    "geo_distance_km"
]

def generate_fallback_engineered_dataset(n_rows: int = 5000) -> pd.DataFrame:
    """Generate realistic deterministic synthetic dataset when creditcard.csv is not locally present."""
    rng = np.random.RandomState(42)
    records = []
    # 2% fraud rate
    is_fraud_arr = (rng.rand(n_rows) < 0.02).astype(int)
    if is_fraud_arr.sum() < 10:
        is_fraud_arr[:20] = 1

    for i in range(n_rows):
        is_f = int(is_fraud_arr[i])
        rec = {f"v{j}": float(rng.randn() + (2.5 if is_f and j in [14, 10, 12, 4] else 0.0)) for j in range(1, 29)}
        rec["amount"] = float(rng.exponential(scale=200 if is_f else 50) + 1.0)
        rec["velocity_5m"] = int(rng.poisson(lam=4 if is_f else 0.5))
        rec["velocity_60m"] = int(rng.poisson(lam=8 if is_f else 1.5))
        rec["amount_deviation"] = float(rng.uniform(2.0, 5.0) if is_f else rng.uniform(0.0, 1.2))
        rec["time_since_last_tx_sec"] = float(rng.uniform(5, 60) if is_f else rng.uniform(300, 7200))
        rec["geo_distance_km"] = float(rng.uniform(300, 1200) if is_f else rng.uniform(0, 15))
        rec["time_step"] = float(i * 10.0)
        rec["is_fraud"] = is_f
        rec["customer_id"] = f"CUST_{rng.randint(1, 200):04d}"
        rec["merchant_id"] = f"MERCH_{rng.randint(1, 50):03d}"
        records.append(rec)
    return pd.DataFrame(records)

def prepare_engineered_dataset(max_rows: int = None, force_recompute: bool = False) -> pd.DataFrame:
    """Load or precompute engineered features from creditcard.csv, or fallback to synthetic benchmark dataset."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if ENGINEERED_PARQUET.exists() and not force_recompute and max_rows is None:
        logger.info("Loading precomputed engineered features from: {path}", path=ENGINEERED_PARQUET)
        return pd.read_parquet(ENGINEERED_PARQUET)

    if not RAW_CSV.exists():
        logger.warning("Raw dataset not found at {path}. Using synthetic benchmark dataset.", path=RAW_CSV)
        return generate_fallback_engineered_dataset(n_rows=max_rows or 5000)

    logger.info("Generating engineered features from raw dataset ({path})...", path=RAW_CSV)
    raw_df = pd.read_csv(RAW_CSV)
    if max_rows:
        raw_df = raw_df.head(max_rows)

    # Sort strictly by time
    raw_df = raw_df.sort_values("Time").reset_index(drop=True)

    # Dictionary to maintain historical events per customer
    customer_histories: Dict[str, list] = {}
    engineered_records = []

    for idx, row in raw_df.iterrows():
        amount = float(row["Amount"])
        is_fraud = int(row["Class"])
        meta = generate_synthetic_metadata(idx, amount, is_fraud)
        cust_id = meta["customer_id"]

        current_tx = {
            "transaction_id": f"TX_{idx:07d}",
            "customer_id": cust_id,
            "merchant_id": meta["merchant_id"],
            "amount": amount,
            "time_step": float(row["Time"]),
            "lat": meta["lat"],
            "lon": meta["lon"],
            "is_fraud": is_fraud
        }
        for i in range(1, 29):
            current_tx[f"v{i}"] = float(row[f"V{i}"])

        history = customer_histories.get(cust_id, [])
        features = compute_rolling_features(current_tx, history)
        engineered_records.append(features)

        # Update customer in-memory history window (keep up to 50 recent)
        history.append(current_tx)
        if len(history) > 50:
            history.pop(0)
        customer_histories[cust_id] = history

        if idx > 0 and idx % 25000 == 0:
            logger.info("Feature engineering progress: {idx}/{total} records", idx=idx, total=len(raw_df))

    df_out = pd.DataFrame(engineered_records)
    
    if max_rows is None:
        df_out.to_parquet(ENGINEERED_PARQUET, index=False)
        logger.info("Saved complete engineered features dataset to: {path}", path=ENGINEERED_PARQUET)

    return df_out

def configure_mlflow(tracking_uri: str = None, experiment_name: str = "fraud-detection-experiments"):
    """Configure MLflow tracking with automatic fallback to local SQLite database if server is down."""
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    target_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

    if target_uri.startswith("http://") or target_uri.startswith("https://"):
        import urllib.request
        try:
            req = urllib.request.Request(target_uri)
            urllib.request.urlopen(req, timeout=1.0)
            logger.info("Connected to remote MLflow server at: {uri}", uri=target_uri)
        except Exception:
            logger.warning("Remote MLflow server at {uri} is offline. Falling back to local 'sqlite:///mlflow.db'.", uri=target_uri)
            target_uri = "sqlite:///mlflow.db"

    mlflow.set_tracking_uri(target_uri)
    try:
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        logger.warning("Could not set experiment {exp}: {err}", exp=experiment_name, err=str(e))

def train_and_compare_strategies(
    df: pd.DataFrame,
    tracking_uri: str = None,
    experiment_name: str = "fraud-detection-experiments",
    registry_dir: Path = None,
    n_estimators: int = 100
) -> Dict[str, Any]:
    """Train and compare 3 imbalance strategies with MLflow logging."""
    target_registry = Path(registry_dir) if registry_dir else REGISTRY_DIR
    target_registry.mkdir(parents=True, exist_ok=True)

    configure_mlflow(tracking_uri, experiment_name)

    X = df[FEATURE_COLUMNS]
    y = df["is_fraud"].astype(int)

    fraud_rate = y.mean() * 100
    total_fraud = y.sum()
    logger.info("Dataset shape: X={shape}, Fraud cases: {count} ({rate:.3f}%)",
                shape=X.shape, count=total_fraud, rate=fraud_rate)

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    logger.info("Stratified Split: Train={n_tr} ({tr_f} fraud) | Test={n_te} ({te_f} fraud)",
                n_tr=len(X_train), tr_f=int(y_train.sum()),
                n_te=len(X_test), te_f=int(y_test.sum()))

    pos_count = max(int(y_train.sum()), 1)
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / pos_count

    strategies_results = []

    # =========================================================================
    # Strategy 1: Class-Weighted XGBoost (scale_pos_weight)
    # =========================================================================
    logger.info("Training Strategy 1: Class-Weighted XGBoost (scale_pos_weight={spw:.1f})...", spw=scale_pos_weight)
    with mlflow.start_run(run_name="xgboost_class_weighted"):
        params_xgb1 = {
            "n_estimators": n_estimators,
            "max_depth": 5,
            "learning_rate": 0.1,
            "scale_pos_weight": scale_pos_weight,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "eval_metric": "aucpr"
        }
        mlflow.log_params(params_xgb1)
        mlflow.log_param("strategy", "class_weighted_xgboost")

        model_xgb1 = xgb.XGBClassifier(**params_xgb1)
        model_xgb1.fit(X_train, y_train)

        y_prob1 = model_xgb1.predict_proba(X_test)[:, 1]
        metrics1 = calculate_metrics(y_test, y_prob1, threshold=0.5)
        opt_th1, opt_m1 = find_optimal_threshold(y_test, y_prob1, target_recall=0.85)
        
        mlflow.log_metrics(metrics1)
        mlflow.log_metric("opt_threshold", opt_th1)
        mlflow.log_metric("opt_recall", opt_m1["recall"])
        mlflow.log_metric("opt_f1", opt_m1["f1"])
        try:
            mlflow.xgboost.log_model(model_xgb1, "model")
        except Exception as e:
            logger.warning("Could not log model artifact to MLflow: {err}", err=str(e))

        strategies_results.append({
            "name": "class_weighted_xgboost",
            "model": model_xgb1,
            "metrics": metrics1,
            "optimal_threshold": opt_th1,
            "opt_metrics": opt_m1,
            "pr_auc": metrics1["pr_auc"]
        })

    # =========================================================================
    # Strategy 2: SMOTE Oversampling + XGBoost
    # =========================================================================
    logger.info("Training Strategy 2: SMOTE Oversampling + XGBoost...")
    with mlflow.start_run(run_name="xgboost_smote"):
        # Resample minority class to 10% of majority to prevent overfitting
        smote = SMOTE(sampling_strategy=0.10, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        params_xgb2 = {
            "n_estimators": n_estimators,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "eval_metric": "aucpr"
        }
        mlflow.log_params(params_xgb2)
        mlflow.log_param("strategy", "smote_xgboost")
        mlflow.log_param("smote_sampling_ratio", 0.10)

        model_xgb2 = xgb.XGBClassifier(**params_xgb2)
        model_xgb2.fit(X_resampled, y_resampled)

        y_prob2 = model_xgb2.predict_proba(X_test)[:, 1]
        metrics2 = calculate_metrics(y_test, y_prob2, threshold=0.5)
        opt_th2, opt_m2 = find_optimal_threshold(y_test, y_prob2, target_recall=0.85)

        mlflow.log_metrics(metrics2)
        mlflow.log_metric("opt_threshold", opt_th2)
        mlflow.log_metric("opt_recall", opt_m2["recall"])
        mlflow.log_metric("opt_f1", opt_m2["f1"])
        try:
            mlflow.xgboost.log_model(model_xgb2, "model")
        except Exception as e:
            logger.warning("Could not log model artifact to MLflow: {err}", err=str(e))

        strategies_results.append({
            "name": "smote_xgboost",
            "model": model_xgb2,
            "metrics": metrics2,
            "optimal_threshold": opt_th2,
            "opt_metrics": opt_m2,
            "pr_auc": metrics2["pr_auc"]
        })

    # =========================================================================
    # Strategy 3: Balanced LightGBM
    # =========================================================================
    logger.info("Training Strategy 3: Balanced LightGBM...")
    with mlflow.start_run(run_name="lightgbm_balanced"):
        params_lgb = {
            "n_estimators": n_estimators,
            "max_depth": 5,
            "learning_rate": 0.1,
            "class_weight": "balanced",
            "random_state": 42,
            "verbose": -1
        }
        mlflow.log_params(params_lgb)
        mlflow.log_param("strategy", "balanced_lightgbm")

        model_lgb = lgb.LGBMClassifier(**params_lgb)
        model_lgb.fit(X_train, y_train)

        y_prob3 = model_lgb.predict_proba(X_test)[:, 1]
        metrics3 = calculate_metrics(y_test, y_prob3, threshold=0.5)
        opt_th3, opt_m3 = find_optimal_threshold(y_test, y_prob3, target_recall=0.85)

        mlflow.log_metrics(metrics3)
        mlflow.log_metric("opt_threshold", opt_th3)
        mlflow.log_metric("opt_recall", opt_m3["recall"])
        mlflow.log_metric("opt_f1", opt_m3["f1"])
        try:
            mlflow.lightgbm.log_model(model_lgb, "model")
        except Exception as e:
            logger.warning("Could not log model artifact to MLflow: {err}", err=str(e))

        strategies_results.append({
            "name": "balanced_lightgbm",
            "model": model_lgb,
            "metrics": metrics3,
            "optimal_threshold": opt_th3,
            "opt_metrics": opt_m3,
            "pr_auc": metrics3["pr_auc"]
        })

    # =========================================================================
    # Select Champion Model by PR-AUC and Export Artifact
    # =========================================================================
    champion = max(strategies_results, key=lambda x: x["pr_auc"])
    logger.info(
        "Champion Model Selected: '{name}' | PR-AUC={pr:.4f} | ROC-AUC={roc:.4f} | Recall={rec:.4f} | Optimal Thresh={th}",
        name=champion["name"],
        pr=champion["pr_auc"],
        roc=champion["metrics"]["roc_auc"],
        rec=champion["opt_metrics"]["recall"],
        th=champion["optimal_threshold"]
    )

    # Save local serialized model artifact
    artifact_path = target_registry / "fraud_xgb_v1.joblib"
    joblib.dump(champion["model"], artifact_path)
    logger.info("Champion model exported to: {path}", path=artifact_path)

    # Save model metadata
    metadata = {
        "model_version": "fraud-xgb-v1",
        "champion_strategy": champion["name"],
        "pr_auc": champion["pr_auc"],
        "roc_auc": champion["metrics"]["roc_auc"],
        "optimal_threshold": champion["optimal_threshold"],
        "default_metrics": champion["metrics"],
        "calibrated_metrics": champion["opt_metrics"],
        "features": FEATURE_COLUMNS,
        "trained_timestamp": pd.Timestamp.now().isoformat()
    }
    with open(target_registry / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "champion": champion["name"],
        "pr_auc": champion["pr_auc"],
        "results": strategies_results,
        "metadata": metadata
    }

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate fraud detection models with MLflow")
    parser.add_argument("--max-rows", type=int, default=None, help="Cap dataset size for rapid testing")
    parser.add_argument("--force-recompute", action="store_true", help="Force recalculation of engineered features")
    args = parser.parse_args()

    df = prepare_engineered_dataset(max_rows=args.max_rows, force_recompute=args.force_recompute)
    train_and_compare_strategies(df)

if __name__ == "__main__":
    main()
