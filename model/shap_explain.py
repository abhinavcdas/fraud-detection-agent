"""SHAP Explainability Module for Real-Time Fraud Detection.

Provides global model interpretability and local transaction-level explanations:
1. Global summary plot (SHAP beeswarm ranking features by impact).
2. Local waterfall plots (individual transaction feature push/pull forces).
3. Top contributing feature extraction for LLM agent dossiers and API responses.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless server execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import joblib

from core.logger import get_logger

logger = get_logger("shap_explain")

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "registry" / "fraud_xgb_v1.joblib"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

_EXPLAINER_CACHE: Dict[int, shap.TreeExplainer] = {}


def load_champion_model(model_path: Optional[Union[str, Path]] = None) -> Any:
    """Load trained champion model from registry.
    
    Args:
        model_path: Path to serialized joblib model. Defaults to fraud_xgb_v1.joblib.
        
    Returns:
        Trained model instance.
    """
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}. Train the model first.")
    logger.debug("Loading model artifact from: {path}", path=str(path))
    return joblib.load(path)


def get_tree_explainer(model: Any = None, model_path: Optional[Union[str, Path]] = None) -> shap.TreeExplainer:
    """Obtain or cache a SHAP TreeExplainer for the specified model.
    
    Args:
        model: Trained tree-based model instance. If None, loaded from model_path.
        model_path: Path to model artifact if model is not provided.
        
    Returns:
        shap.TreeExplainer instance.
    """
    if model is None:
        model = load_champion_model(model_path)

    model_id = id(model)
    if model_id not in _EXPLAINER_CACHE:
        logger.info("Initializing SHAP TreeExplainer for model: {cls}", cls=model.__class__.__name__)
        _EXPLAINER_CACHE[model_id] = shap.TreeExplainer(model)
    return _EXPLAINER_CACHE[model_id]


def compute_shap_values(model: Any, X_data: pd.DataFrame) -> shap.Explanation:
    """Compute SHAP explanation object for tabular feature inputs.
    
    Args:
        model: Trained model instance.
        X_data: DataFrame of features matching model training signature.
        
    Returns:
        shap.Explanation object containing values, base_values, and data.
    """
    explainer = get_tree_explainer(model)
    return explainer(X_data)


def generate_global_shap_summary(
    model: Any,
    X_sample: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
    max_display: int = 15
) -> str:
    """Generate and save global SHAP beeswarm summary plot.
    
    Args:
        model: Trained model instance.
        X_sample: Representative sample of transaction features.
        output_path: Path to write output PNG image.
        max_display: Number of top features to show on plot.
        
    Returns:
        Absolute string path to saved plot image.
    """
    out_file = Path(output_path) if output_path else REPORTS_DIR / "shap_summary.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Generating global SHAP summary plot across {n} sample records...", n=len(X_sample))
    explainer = get_tree_explainer(model)
    shap_explanation = explainer(X_sample)

    fig = plt.figure(figsize=(10, 7), dpi=150)
    shap.summary_plot(shap_explanation.values, X_sample, max_display=max_display, show=False)
    plt.title("Global Feature Importance (SHAP Summary)", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved global SHAP summary plot to: {path}", path=str(out_file))
    return str(out_file.resolve())


def generate_local_waterfall_plot(
    model: Any,
    tx_features: Union[pd.Series, pd.DataFrame, Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
    max_display: int = 10,
    title: Optional[str] = None
) -> str:
    """Generate and save local SHAP waterfall plot for an individual transaction.
    
    Args:
        model: Trained model instance.
        tx_features: Single transaction feature vector (dict, Series, or 1-row DataFrame).
        output_path: Path to write output PNG image.
        max_display: Maximum features to display in waterfall breakdown.
        title: Optional custom plot title.
        
    Returns:
        Absolute string path to saved plot image.
    """
    out_file = Path(output_path) if output_path else REPORTS_DIR / "shap_waterfall.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(tx_features, dict):
        df_row = pd.DataFrame([tx_features])
    elif isinstance(tx_features, pd.Series):
        df_row = pd.DataFrame([tx_features.to_dict()])
    elif isinstance(tx_features, pd.DataFrame):
        df_row = tx_features.iloc[[0]]
    else:
        raise TypeError(f"Unsupported tx_features type: {type(tx_features)}")

    logger.debug("Generating local SHAP waterfall plot for individual transaction...")
    explainer = get_tree_explainer(model)
    explanation = explainer(df_row)

    fig = plt.figure(figsize=(9, 6), dpi=150)
    shap.plots.waterfall(explanation[0], max_display=max_display, show=False)
    if title:
        plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved local SHAP waterfall plot to: {path}", path=str(out_file))
    return str(out_file.resolve())


def get_top_contributing_features(
    model: Any,
    tx_features: Union[pd.Series, pd.DataFrame, Dict[str, Any]],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Extract top-k risk-increasing and risk-decreasing features for a transaction.
    
    Args:
        model: Trained model instance.
        tx_features: Single transaction feature dictionary or row.
        top_k: Number of highest absolute impact features to extract.
        
    Returns:
        List of dicts: [{"feature": str, "value": float, "shap_value": float, "direction": str}]
    """
    if isinstance(tx_features, dict):
        df_row = pd.DataFrame([tx_features])
    elif isinstance(tx_features, pd.Series):
        df_row = pd.DataFrame([tx_features.to_dict()])
    elif isinstance(tx_features, pd.DataFrame):
        df_row = tx_features.iloc[[0]]
    else:
        raise TypeError(f"Unsupported tx_features type: {type(tx_features)}")

    explainer = get_tree_explainer(model)
    explanation = explainer(df_row)[0]

    feature_names = df_row.columns.tolist()
    shap_vals = explanation.values
    feat_vals = df_row.iloc[0].values

    contributors = []
    for name, s_val, f_val in zip(feature_names, shap_vals, feat_vals):
        direction = "INCREASES_RISK" if s_val > 0 else "DECREASES_RISK"
        contributors.append({
            "feature": name,
            "feature_value": round(float(f_val), 4) if pd.notna(f_val) else 0.0,
            "shap_value": round(float(s_val), 4),
            "abs_impact": abs(float(s_val)),
            "direction": direction
        })

    # Sort by absolute impact descending
    contributors.sort(key=lambda x: x["abs_impact"], reverse=True)
    return contributors[:top_k]
