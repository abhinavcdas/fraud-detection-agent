"""Model Evaluation & Metrics Calculation for Imbalanced Fraud Detection.

Prioritizes PR-AUC (Average Precision) and recall-biased operational thresholding
over deceptive ROC-AUC scores in extreme class imbalance (< 0.2% fraud prevalence).
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve
)

def calculate_metrics(y_true, y_pred_prob, threshold: float = 0.5) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics at a given decision threshold.
    
    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred_prob: Predicted probability of fraud in range [0.0, 1.0].
        threshold: Decision boundary for binary classification.
        
    Returns:
        Dictionary of metrics including PR-AUC, ROC-AUC, F1, Recall, Precision, and Confusion Matrix.
    """
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_pred_prob)
    y_pred = (y_prob_arr >= threshold).astype(int)

    # Calculate PR-AUC and ROC-AUC
    try:
        pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))
    except Exception:
        pr_auc = 0.0

    try:
        roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    except Exception:
        roc_auc = 0.5

    # Threshold-dependent metrics
    prec = float(precision_score(y_true_arr, y_pred, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))

    cm = confusion_matrix(y_true_arr, y_pred)
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])) if cm.shape == (2, 2) else (0, 0, 0, 0)

    return {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "threshold": round(threshold, 3),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }

def find_optimal_threshold(y_true, y_pred_prob, target_recall: float = 0.85) -> Tuple[float, Dict[str, Any]]:
    """Determine the optimal decision threshold that achieves target recall while maximizing precision.
    
    In banking transaction monitoring, false negatives (missed fraud) are costly,
    so thresholds are calibrated to guarantee minimum recall.
    """
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_pred_prob)

    best_thresh = 0.5
    best_metrics = calculate_metrics(y_true_arr, y_prob_arr, threshold=0.5)
    best_f1 = -1.0

    # Scan candidate thresholds
    for thresh in np.linspace(0.02, 0.90, 45):
        m = calculate_metrics(y_true_arr, y_prob_arr, threshold=thresh)
        if m["recall"] >= target_recall:
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                best_thresh = thresh
                best_metrics = m

    # Fallback to threshold that maximizes F1 if target recall cannot be reached
    if best_f1 < 0:
        for thresh in np.linspace(0.02, 0.90, 45):
            m = calculate_metrics(y_true_arr, y_prob_arr, threshold=thresh)
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                best_thresh = thresh
                best_metrics = m

    return best_thresh, best_metrics
