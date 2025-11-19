"""Find optimal classification threshold using validation data."""

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc

from verito.model import load_detector
from verito.metrics import calculate_metrics


def find_optimal_threshold(
    model_path: Path, fraud_dir: Path, authentic_dir: Path, save_plot: bool = True
):
    """
    Find optimal threshold by analyzing validation data.

    Args:
        model_path: Path to trained model
        fraud_dir: Directory with fraud validation images
        authentic_dir: Directory with authentic validation images
        save_plot: Whether to save threshold analysis plot

    Returns:
        dict: Optimal thresholds for different objectives
    """
    print("\n🔍 Finding Optimal Classification Threshold\n")

    # Load model
    print("Loading model...")
    model = load_detector(model_path)

    # Get raw probabilities (don't apply threshold yet)
    print("Processing validation images...")

    # We need raw probabilities, so we'll get them directly
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    def get_probabilities(image_dir):
        """Get raw probabilities for images in directory."""
        probabilities = []
        for img_path in sorted(image_dir.glob("*.*")):
            if img_path.suffix.lower() in [
                ".jpg",
                ".jpeg",
                ".png",
                ".tiff",
                ".tif",
                ".bmp",
            ]:
                img = load_img(img_path, target_size=(480, 480))
                img_array = img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                prob = model.predict(img_array, verbose=0)[0][0].item()
                probabilities.append(prob)
        return np.array(probabilities)

    # Get probabilities
    fraud_probs = get_probabilities(fraud_dir)
    authentic_probs = get_probabilities(authentic_dir)

    # Create true labels
    y_true = np.concatenate(
        [
            np.zeros(len(fraud_probs)),  # Fraud = 0
            np.ones(len(authentic_probs)),  # Authentic = 1
        ]
    )
    y_prob = np.concatenate([fraud_probs, authentic_probs])

    print(f"   Fraud images: {len(fraud_probs)}")
    print(f"   Authentic images: {len(authentic_probs)}")
    print(f"   Total: {len(y_true)}\n")

    # Test different thresholds
    thresholds_to_test = np.arange(0.1, 0.9, 0.05)
    results = []

    print("Testing thresholds...")
    print("-" * 80)
    print(
        f"{'Threshold':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}"
    )
    print("-" * 80)

    for threshold in thresholds_to_test:
        y_pred = (y_prob >= threshold).astype(int)
        metrics = calculate_metrics(y_true, y_pred)

        results.append(
            {
                "threshold": threshold,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
            }
        )

        print(
            f"{threshold:<12.2f} {metrics['accuracy']:<12.1%} "
            f"{metrics['precision']:<12.1%} {metrics['recall']:<12.1%} "
            f"{metrics['f1_score']:<12.1%}"
        )

    print("-" * 80 + "\n")

    # Find optimal thresholds for different objectives
    results_df = results

    # Best F1 (balanced)
    best_f1_idx = max(range(len(results_df)), key=lambda i: results_df[i]["f1_score"])
    best_f1 = results_df[best_f1_idx]

    # Best recall (catch more fraud)
    best_recall_idx = max(range(len(results_df)), key=lambda i: results_df[i]["recall"])
    best_recall = results_df[best_recall_idx]

    # Best precision (reduce false positives)
    best_precision_idx = max(
        range(len(results_df)), key=lambda i: results_df[i]["precision"]
    )
    best_precision = results_df[best_precision_idx]

    # Calculate Youden's J statistic (optimal ROC point)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_roc_threshold = roc_thresholds[optimal_idx]

    # Print recommendations
    print("=" * 80)
    print("📊 Optimal Thresholds for Different Objectives")
    print("=" * 80)

    print(f"\n1. Balanced (Best F1 Score): {best_f1['threshold']:.2f}")
    print(
        f"   Accuracy: {best_f1['accuracy']:.1%}, Precision: {best_f1['precision']:.1%}, "
        f"Recall: {best_f1['recall']:.1%}, F1: {best_f1['f1_score']:.1%}"
    )
    print("   → Use when you want overall balanced performance")

    print(f"\n2. ROC-Optimized (Youden's J): {optimal_roc_threshold:.2f}")
    print("   → Use for theoretical optimal based on ROC curve")

    print(f"\n3. High Recall (Catch More Fraud): {best_recall['threshold']:.2f}")
    print(
        f"   Accuracy: {best_recall['accuracy']:.1%}, Precision: {best_recall['precision']:.1%}, "
        f"Recall: {best_recall['recall']:.1%}, F1: {best_recall['f1_score']:.1%}"
    )
    print("   → Use when missing fraud is very costly")

    print(
        f"\n4. High Precision (Fewer False Alarms): {best_precision['threshold']:.2f}"
    )
    print(
        f"   Accuracy: {best_precision['accuracy']:.1%}, Precision: {best_precision['precision']:.1%}, "
        f"Recall: {best_precision['recall']:.1%}, F1: {best_precision['f1_score']:.1%}"
    )
    print("   → Use when false positives are costly")

    print("\n" + "=" * 80 + "\n")

    # Create visualization
    if save_plot:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Threshold Analysis", fontsize=16, fontweight="bold")

        # Extract values
        thresholds = [r["threshold"] for r in results_df]
        accuracies = [r["accuracy"] for r in results_df]
        precisions = [r["precision"] for r in results_df]
        recalls = [r["recall"] for r in results_df]
        f1_scores = [r["f1_score"] for r in results_df]

        # Plot 1: All metrics vs threshold
        axes[0, 0].plot(thresholds, accuracies, "o-", label="Accuracy", linewidth=2)
        axes[0, 0].plot(thresholds, precisions, "s-", label="Precision", linewidth=2)
        axes[0, 0].plot(thresholds, recalls, "^-", label="Recall", linewidth=2)
        axes[0, 0].plot(thresholds, f1_scores, "d-", label="F1 Score", linewidth=2)
        axes[0, 0].axvline(
            best_f1["threshold"],
            color="red",
            linestyle="--",
            alpha=0.5,
            label=f"Best F1 ({best_f1['threshold']:.2f})",
        )
        axes[0, 0].set_xlabel("Threshold", fontsize=12)
        axes[0, 0].set_ylabel("Score", fontsize=12)
        axes[0, 0].set_title("Metrics vs Threshold", fontsize=13, fontweight="bold")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 1])

        # Plot 2: Precision-Recall Curve
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
            y_true, y_prob
        )
        axes[0, 1].plot(recall_curve, precision_curve, linewidth=2)
        axes[0, 1].set_xlabel("Recall (Fraud Caught)", fontsize=12)
        axes[0, 1].set_ylabel("Precision", fontsize=12)
        axes[0, 1].set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xlim([0, 1])
        axes[0, 1].set_ylim([0, 1])

        # Plot 3: ROC Curve
        roc_auc = auc(fpr, tpr)
        axes[1, 0].plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {roc_auc:.3f})")
        axes[1, 0].plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
        axes[1, 0].scatter(
            fpr[optimal_idx],
            tpr[optimal_idx],
            c="red",
            s=100,
            label=f"Optimal ({optimal_roc_threshold:.2f})",
            zorder=5,
        )
        axes[1, 0].set_xlabel("False Positive Rate", fontsize=12)
        axes[1, 0].set_ylabel("True Positive Rate (Recall)", fontsize=12)
        axes[1, 0].set_title("ROC Curve", fontsize=13, fontweight="bold")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: F1 Score detail
        axes[1, 1].plot(thresholds, f1_scores, "o-", linewidth=2, markersize=6)
        axes[1, 1].axvline(
            best_f1["threshold"], color="red", linestyle="--", alpha=0.7, linewidth=2
        )
        axes[1, 1].axhline(best_f1["f1_score"], color="red", linestyle="--", alpha=0.3)
        axes[1, 1].scatter(
            [best_f1["threshold"]],
            [best_f1["f1_score"]],
            c="red",
            s=150,
            zorder=5,
            label=f"Best F1: {best_f1['threshold']:.2f}",
        )
        axes[1, 1].set_xlabel("Threshold", fontsize=12)
        axes[1, 1].set_ylabel("F1 Score", fontsize=12)
        axes[1, 1].set_title("F1 Score vs Threshold", fontsize=13, fontweight="bold")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 1])

        plt.tight_layout()

        # Save plot
        project_root = Path(__file__).parent.parent
        output_path = project_root / "models" / "threshold_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"📈 Threshold analysis plot saved to: {output_path}\n")

    return {
        "balanced": best_f1["threshold"],
        "high_recall": best_recall["threshold"],
        "high_precision": best_precision["threshold"],
        "roc_optimal": optimal_roc_threshold,
    }


def main():
    """Main function."""
    project_root = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(
        description="Find optimal classification threshold using validation data"
    )
    parser.add_argument(
        "--fraud-dir",
        type=Path,
        default=project_root / "datasets" / "test" / "Fraud",
        help="Directory with fraud validation images",
    )
    parser.add_argument(
        "--authentic-dir",
        type=Path,
        default=project_root / "datasets" / "test" / "Non-Fraud",
        help="Directory with authentic validation images",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=project_root / "models" / "verito.keras",
        help="Path to trained model",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Don't save visualization plot"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.model_path.exists():
        print(f"❌ Model not found: {args.model_path}")
        print("   Please train the model first: uv run scripts/train.py")
        sys.exit(1)

    if not args.fraud_dir.exists():
        print(f"❌ Fraud directory not found: {args.fraud_dir}")
        sys.exit(1)

    if not args.authentic_dir.exists():
        print(f"❌ Authentic directory not found: {args.authentic_dir}")
        sys.exit(1)

    # Find optimal threshold
    optimal_thresholds = find_optimal_threshold(
        model_path=args.model_path,
        fraud_dir=args.fraud_dir,
        authentic_dir=args.authentic_dir,
        save_plot=not args.no_plot,
    )

    print("💡 Recommendation:")
    print(
        f"   Use threshold {optimal_thresholds['balanced']:.2f} for balanced performance"
    )
    print("   or adjust based on your specific cost-benefit analysis.\n")


if __name__ == "__main__":
    main()
