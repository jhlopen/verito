"""Test metrics calculation logic."""

import pytest
import numpy as np
from verito.metrics import calculate_metrics, print_evaluation_report


def test_calculate_metrics_perfect_predictions():
    """Test metrics with perfect predictions."""
    y_true = np.array([0, 0, 0, 1, 1, 1])  # 3 fraud, 3 authentic
    y_pred = np.array([0, 0, 0, 1, 1, 1])  # All correct

    metrics = calculate_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0  # Fraud precision
    assert metrics["recall"] == 1.0  # Fraud recall
    assert metrics["f1_score"] == 1.0
    assert metrics["true_fraud_detected"] == 3
    assert metrics["false_positive"] == 0
    assert metrics["false_negative"] == 0
    assert metrics["true_authentic"] == 3


def test_calculate_metrics_realistic_scenario():
    """Test with realistic imbalanced predictions."""
    # Simulate: 3 fraud, 10 authentic (like 25:1 ratio scaled down)
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    # Simulate: caught 2/3 fraud, classified 2 authentic as fraud
    y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

    metrics = calculate_metrics(y_true, y_pred)

    # Fraud recall: 2/3 = 0.67
    assert metrics["recall"] == pytest.approx(0.67, abs=0.01)

    # Fraud precision: 2/4 = 0.5 (2 true positives, 2 false positives)
    assert metrics["precision"] == pytest.approx(0.5, abs=0.01)

    # Check confusion matrix values
    assert metrics["true_fraud_detected"] == 2
    assert metrics["false_positive"] == 2
    assert metrics["false_negative"] == 1
    assert metrics["true_authentic"] == 8


def test_calculate_metrics_all_wrong():
    """Test metrics when everything is predicted wrong."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 0, 0, 0])  # All flipped

    metrics = calculate_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 0.0
    assert metrics["true_fraud_detected"] == 0
    assert metrics["false_negative"] == 3
    assert metrics["false_positive"] == 3
    assert metrics["true_authentic"] == 0


def test_print_evaluation_report_runs():
    """Test that Rich table formatting doesn't crash."""
    metrics = {
        "accuracy": 0.85,
        "precision": 0.75,
        "recall": 0.70,
        "f1_score": 0.72,
        "confusion_matrix": np.array([[70, 30], [8, 92]]),
        "true_fraud_detected": 70,
        "false_positive": 8,
        "false_negative": 30,
        "true_authentic": 92,
    }

    # Just verify it runs without errors
    try:
        print_evaluation_report(metrics, total_forged=100, total_authentic=100)
        assert True
    except Exception as e:
        pytest.fail(f"print_evaluation_report raised {e}")


def test_threshold_logic():
    """Test the 0.5 threshold logic."""
    raw_predictions = np.array([0.1, 0.49, 0.5, 0.51, 0.9])
    threshold = 0.5

    # Logic: < threshold = fraud (0), >= threshold = authentic (1)
    predictions = (raw_predictions >= threshold).astype(int)

    expected = np.array([0, 0, 1, 1, 1])
    np.testing.assert_array_equal(predictions, expected)
