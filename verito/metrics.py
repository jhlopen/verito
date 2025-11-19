"""Evaluation metrics and reporting."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate classification metrics.

    Args:
        y_true: True labels (0 = Fraud, 1 = Non-Fraud)
        y_pred: Predicted labels (0 = Fraud, 1 = Non-Fraud)

    Returns:
        Dictionary of metrics including:
        - accuracy: Overall accuracy
        - precision: Precision for fraud class
        - recall: Recall for fraud class
        - f1_score: F1 score for fraud class
        - confusion_matrix: 2x2 confusion matrix
        - true_fraud_detected: Count of correctly detected fraud
        - false_positive: Count of authentic marked as fraud
        - false_negative: Count of fraud marked as authentic
        - true_authentic: Count of correctly identified authentic
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="binary", pos_label=0)
    recall = recall_score(y_true, y_pred, average="binary", pos_label=0)
    f1 = f1_score(y_true, y_pred, average="binary", pos_label=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Extract confusion matrix values
    # Note: sklearn confusion matrix format is [[TN, FP], [FN, TP]]
    # But we want fraud (class 0) as positive class
    if cm.shape == (2, 2):
        true_fraud_detected = cm[0, 0]  # True positives (fraud correctly identified)
        false_positive = cm[1, 0]  # False positives (authentic marked as fraud)
        false_negative = cm[0, 1]  # False negatives (fraud marked as authentic)
        true_authentic = cm[1, 1]  # True negatives (authentic correctly identified)
    else:
        true_fraud_detected = 0
        false_positive = 0
        false_negative = 0
        true_authentic = 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "true_fraud_detected": true_fraud_detected,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_authentic": true_authentic,
    }


def print_evaluation_report(metrics: dict, total_forged: int, total_authentic: int):
    """
    Print a detailed evaluation report.

    Args:
        metrics: Dictionary of calculated metrics from calculate_metrics()
        total_forged: Total number of forged images
        total_authentic: Total number of authentic images
    """
    print("\n" + "=" * 70)
    print("               🔍 Forged Image Detection Results")
    print("=" * 70 + "\n")

    # Dataset summary
    print("📊 Dataset Summary:")
    print(f"   Total images: {total_forged + total_authentic}")
    print(f"   Forged: {total_forged}")
    print(f"   Authentic: {total_authentic}\n")

    # Performance metrics
    print("Performance Metrics:")
    print(f"   Accuracy:              {metrics['accuracy']:.2%}")
    print(f"   Precision:             {metrics['precision']:.2%}")
    print(f"   Recall (Fraud):        {metrics['recall']:.2%}")
    print(f"   F1 Score:              {metrics['f1_score']:.2%}\n")

    # Detailed results
    print("📈 Detailed Results:")
    print(
        f"   ✅ Forged images detected: {metrics['true_fraud_detected']}/{total_forged} "
        f"({metrics['true_fraud_detected'] / total_forged * 100:.1f}%)"
    )
    print(
        f"   ❌ Forged images missed (False Negatives): {metrics['false_negative']}/{total_forged} "
        f"({metrics['false_negative'] / total_forged * 100:.1f}%)"
    )
    print(
        f"   ⚠️  Authentic images incorrectly flagged (False Positives): {metrics['false_positive']}/{total_authentic} "
        f"({metrics['false_positive'] / total_authentic * 100:.1f}%)"
    )
    print(
        f"   ✅ Authentic images correctly identified: {metrics['true_authentic']}/{total_authentic} "
        f"({metrics['true_authentic'] / total_authentic * 100:.1f}%)\n"
    )

    # Confusion matrix
    print("🔢 Confusion Matrix:")
    print("                 Predicted")
    print("              Fraud  Non-Fraud")
    print(
        f"    Fraud      {metrics['confusion_matrix'][0, 0]:4d}     {metrics['confusion_matrix'][0, 1]:4d}"
    )
    print(
        f"    Non-Fraud  {metrics['confusion_matrix'][1, 0]:4d}     {metrics['confusion_matrix'][1, 1]:4d}\n"
    )

    print("=" * 70 + "\n")
