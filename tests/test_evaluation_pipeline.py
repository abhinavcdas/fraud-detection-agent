"""Unit and Integration Tests for Phase 4 Evaluation Pipeline & SHAP Explainability."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from model.shap_explain import (
    load_champion_model,
    get_tree_explainer,
    get_top_contributing_features,
    generate_global_shap_summary,
    generate_local_waterfall_plot
)
from eval.eval_model import compute_evaluation_metrics
from eval.eval_agent import run_agent_eval, align_recommendations

FEATURE_COLS = [f"v{i}" for i in range(1, 29)] + [
    "amount", "velocity_5m", "velocity_60m", "amount_deviation", "time_since_last_tx_sec", "geo_distance_km"
]


def test_shap_explainer_initialization():
    """Verify SHAP TreeExplainer initializes and caches properly."""
    model = load_champion_model()
    explainer1 = get_tree_explainer(model)
    explainer2 = get_tree_explainer(model)
    assert explainer1 is explainer2
    assert hasattr(explainer1, "shap_values") or hasattr(explainer1, "__call__")


def test_top_contributing_features():
    """Verify feature attribution returns structured, sorted impact metrics."""
    model = load_champion_model()
    dummy_tx = pd.Series(np.random.randn(len(FEATURE_COLS)), index=FEATURE_COLS)
    top_factors = get_top_contributing_features(model, dummy_tx, top_k=5)

    assert len(top_factors) == 5
    for item in top_factors:
        assert "feature" in item
        assert "feature_value" in item
        assert "shap_value" in item
        assert item["direction"] in ("INCREASES_RISK", "DECREASES_RISK")

    # Assert descending sort by absolute impact
    impacts = [item["abs_impact"] for item in top_factors]
    assert impacts == sorted(impacts, reverse=True)


def test_shap_plot_generation(tmp_path):
    """Verify SHAP summary and waterfall plots write valid image files."""
    model = load_champion_model()
    sample_df = pd.DataFrame(np.random.randn(15, len(FEATURE_COLS)), columns=FEATURE_COLS)

    summary_img = tmp_path / "test_shap_summary.png"
    waterfall_img = tmp_path / "test_shap_waterfall.png"

    path_sum = generate_global_shap_summary(model, sample_df, output_path=summary_img, max_display=5)
    path_wf = generate_local_waterfall_plot(model, sample_df.iloc[0], output_path=waterfall_img, max_display=5)

    assert Path(path_sum).exists()
    assert Path(path_sum).stat().st_size > 5000  # Non-trivial PNG size
    assert Path(path_wf).exists()
    assert Path(path_wf).stat().st_size > 5000


def test_compute_evaluation_metrics():
    """Verify standard evaluation metric calculations on controlled synthetic arrays."""
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])

    metrics = compute_evaluation_metrics(y_true, y_proba, threshold=0.5)

    assert metrics["pr_auc"] > 0.9
    assert metrics["roc_auc"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["true_positives"] == 2
    assert metrics["true_negatives"] == 4
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0


def test_agent_recommendation_alignment_logic():
    """Verify fuzzy concordance logic between expected and predicted actions."""
    assert align_recommendations("DECLINE", "DECLINE") is True
    assert align_recommendations("APPROVE", "APPROVE") is True
    assert align_recommendations("ESCALATE", "ESCALATE") is True
    assert align_recommendations("MANUAL_REVIEW", "ESCALATE") is True
    assert align_recommendations("MANUAL_REVIEW", "MONITOR") is True
    assert align_recommendations("APPROVE", "DECLINE") is False
    assert align_recommendations("DECLINE", "APPROVE") is False


def test_run_agent_eval_on_benchmark():
    """Verify agent evaluation harness over benchmark cases."""
    summary = run_agent_eval(output_scorecard=False)

    assert summary["total_evaluated"] >= 30
    assert summary["average_faithfulness"] >= 0.95
    assert summary["perfect_faithfulness_rate"] >= 0.90
    assert summary["recommendation_concordance_rate"] >= 0.75
    assert summary["latency_mean_ms"] > 0.0
    assert len(summary["detailed_results"]) == summary["total_evaluated"]
