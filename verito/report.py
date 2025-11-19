"""Training report generation and history tracking."""

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend


def save_training_history(history: Dict[str, list], output_dir: Path) -> Path:
    """
    Save training history to JSON file.

    Args:
        history: Training history dictionary from model.fit()
        output_dir: Directory to save history

    Returns:
        Path to saved history file
    """
    history_path = output_dir / "history.json"

    # Convert numpy arrays to lists for JSON serialization
    history_serializable: dict[str, list[float]] = {}
    for key, value in history.items():
        if isinstance(value, np.ndarray):
            history_serializable[key] = value.tolist()
        elif isinstance(value, list):
            history_serializable[key] = value
        else:
            history_serializable[key] = [float(v) for v in value]

    with open(history_path, "w") as f:
        json.dump(history_serializable, f, indent=2)

    return history_path


def generate_training_plots(
    history: Dict[str, list], output_dir: Path
) -> Dict[str, Path]:
    """
    Generate training visualization plots.

    Args:
        history: Training history dictionary
        output_dir: Directory to save plots

    Returns:
        Dictionary mapping plot names to paths
    """
    plots = {}
    epochs = range(1, len(history["loss"]) + 1)
    best_epoch = int(np.argmin(history["val_loss"]) + 1)
    best_val_loss = min(history["val_loss"])

    # Plot 1: Loss curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        epochs, history["loss"], "o-", label="Training Loss", linewidth=2, markersize=5
    )
    ax.plot(
        epochs,
        history["val_loss"],
        "s-",
        label="Validation Loss",
        linewidth=2,
        markersize=5,
    )
    ax.axvline(
        best_epoch,
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"Best Epoch: {best_epoch}",
    )
    ax.scatter([best_epoch], [best_val_loss], c="red", s=150, zorder=5)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = output_dir / "loss_curves.png"
    plt.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close()
    plots["loss"] = loss_path

    # Plot 2: Accuracy curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        epochs,
        history["accuracy"],
        "o-",
        label="Training Accuracy",
        linewidth=2,
        markersize=5,
    )
    ax.plot(
        epochs,
        history["val_accuracy"],
        "s-",
        label="Validation Accuracy",
        linewidth=2,
        markersize=5,
    )
    ax.axvline(
        best_epoch,
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"Best Epoch: {best_epoch}",
    )
    ax.axhline(0.85, color="green", linestyle=":", alpha=0.5, label="Target: 85%")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim((0, 1))
    plt.tight_layout()
    acc_path = output_dir / "accuracy_curves.png"
    plt.savefig(acc_path, dpi=150, bbox_inches="tight")
    plt.close()
    plots["accuracy"] = acc_path

    # Plot 3: Precision curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        epochs,
        history["precision"],
        "o-",
        label="Training Precision",
        linewidth=2,
        markersize=5,
    )
    ax.plot(
        epochs,
        history["val_precision"],
        "s-",
        label="Validation Precision",
        linewidth=2,
        markersize=5,
    )
    ax.axvline(
        best_epoch,
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"Best Epoch: {best_epoch}",
    )
    ax.axhline(0.75, color="green", linestyle=":", alpha=0.5, label="Target: 75%")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision (Fraud Detection)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim((0, 1))
    plt.tight_layout()
    prec_path = output_dir / "precision_curves.png"
    plt.savefig(prec_path, dpi=150, bbox_inches="tight")
    plt.close()
    plots["precision"] = prec_path

    # Plot 4: Recall curves (most important for fraud detection)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        epochs,
        history["recall"],
        "o-",
        label="Training Recall",
        linewidth=2,
        markersize=5,
    )
    ax.plot(
        epochs,
        history["val_recall"],
        "s-",
        label="Validation Recall",
        linewidth=2,
        markersize=5,
    )
    ax.axvline(
        best_epoch,
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"Best Epoch: {best_epoch}",
    )
    ax.axhline(0.7, color="green", linestyle=":", alpha=0.5, label="Target: 70%")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Recall (Fraud Detection Rate)", fontsize=12)
    ax.set_title("Recall", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim((0, 1))
    plt.tight_layout()
    recall_path = output_dir / "recall_curves.png"
    plt.savefig(recall_path, dpi=150, bbox_inches="tight")
    plt.close()
    plots["recall"] = recall_path

    return plots


def generate_report_md(
    history: Dict[str, list],
    config: Dict[str, Any],
    training_time: float,
    output_dir: Path,
    model_path: Path,
) -> Path:
    """
    Generate markdown training report.

    Args:
        history: Training history dictionary
        config: Training configuration
        training_time: Total training time in seconds
        output_dir: Directory to save report
        model_path: Path to the trained model

    Returns:
        Path to report file
    """
    report_path = output_dir / "REPORT.md"

    # Use relative path for report
    model_path_rel = Path("models") / model_path.name

    # Calculate summary statistics
    final_epoch = len(history["loss"])
    best_epoch = int(np.argmin(history["val_loss"]) + 1)
    best_val_loss = min(history["val_loss"])
    final_val_loss = history["val_loss"][-1]
    final_val_acc = history["val_accuracy"][-1]
    final_val_precision = history["val_precision"][-1]
    final_val_recall = history["val_recall"][-1]

    # Training timestamp
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Check if training was interrupted
    interrupted = config.get("interrupted", False)
    interrupted_notice = ""
    if interrupted:
        interrupted_notice = "\n\n⚠️ **TRAINING INTERRUPTED**: This training session was stopped prematurely. The report shows results up to the point of interruption.\n"

    # Generate report content
    report = f"""# Training Report

**Date**: {timestamp}  
**Total Training Time**: {training_time / 60:.1f} minutes ({training_time:.0f} seconds)  
**Total Epochs**: {final_epoch}  
**Best Epoch**: {best_epoch} (based on validation loss){interrupted_notice}

---

## Configuration

### Model Architecture
- Base Model: EfficientNetV2L
- Input Size: {config.get("target_size", (480, 480))}
- Total Parameters: ~120M
- Trainable (Phase 1): ~364K
- Trainable (Phase 2): ~120M

### Training Parameters
- Batch Size: {config.get("batch_size", 16)}
- Phase 1 Epochs: {config.get("epochs_phase1", 15)}
- Phase 2 Epochs: {config.get("epochs_phase2", 25)}
- Early Stopping Patience: {config.get("patience", 8)}
- Learning Rate (Phase 1): {config.get("lr_phase1", 0.001)}
- Learning Rate (Phase 2): {config.get("lr_phase2", 0.00001)}

### Data
- Training Samples: {config.get("train_samples", "N/A")}
- Validation Samples: {config.get("val_samples", "N/A")}
- Class Weight Boost: {config.get("boost_factor", 3.0):.1f}x
- Validation Split: {config.get("validation_split", 0.2):.0%}

---

## Final Results

### Validation Metrics (Final Epoch)

| Metric | Value |
|--------|-------|
| Loss | {final_val_loss:.4f} |
| Accuracy | {final_val_acc:.2%} |
| Precision | {final_val_precision:.2%} |
| Recall (Fraud Detection) | {final_val_recall:.2%} |

### Best Performance (Epoch {best_epoch})

| Metric | Value |
|--------|-------|
| Validation Loss | {best_val_loss:.4f} |
| Validation Accuracy | {history["val_accuracy"][best_epoch - 1]:.2%} |
| Validation Precision | {history["val_precision"][best_epoch - 1]:.2%} |
| Validation Recall | {history["val_recall"][best_epoch - 1]:.2%} |

---

## Training Progress

### 1. Loss Curves

![Loss Curves](loss_curves.png)

**What to look for:**
- Both training and validation loss should **decrease steadily**
- If validation loss increases while training loss decreases → **overfitting**
- If both plateau early → **underfitting** (train longer or increase capacity)
- Best model was saved at **Epoch {best_epoch}** (marked with red line)

**Current status:** Best validation loss = `{best_val_loss:.4f}` at epoch {best_epoch}

---

### 2. Accuracy Curves

![Accuracy Curves](accuracy_curves.png)

**What it means:**
- **Accuracy** = (Correct predictions) / (Total predictions)
- Target: **85-90%** for production use
- Final validation accuracy: **{final_val_acc:.2%}**

**Note:** Accuracy can be misleading with class imbalance (25:1 ratio). Precision and recall are more informative for fraud detection.

---

### 3. Precision Curves

![Precision Curves](precision_curves.png)

**What it means:**
- **Precision** = (True fraud detected) / (All flagged as fraud)
- Measures **false positive rate** - how many false alarms
- Target: **70-85%** (balance between catching fraud and review workload)
- Final validation precision: **{final_val_precision:.2%}**

**Impact:** Lower precision = more manual reviews of false positives

---

### 4. Recall Curves

![Recall Curves](recall_curves.png)

**What it means:**
- **Recall** = (True fraud detected) / (All actual fraud)
- Measures **fraud detection rate** - how many fraud cases we catch
- Target: **60-80%** (higher is better, but harder with subtle forgeries)
- Final validation recall: **{final_val_recall:.2%}**

**Impact:** This is the most critical metric for fraud detection. Low recall means we're missing fraud cases!

**Why recall matters most:** Missing a fraud case (false negative) costs much more than a false alarm (false positive).

---

## Epoch-by-Epoch History

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Val Precision | Val Recall |
|-------|------------|----------|-----------|---------|---------------|------------|
"""

    # Add epoch-by-epoch data
    for i in range(final_epoch):
        report += (
            f"| {i + 1:2d} | {history['loss'][i]:.4f} | {history['val_loss'][i]:.4f} | "
        )
        report += f"{history['accuracy'][i]:.3f} | {history['val_accuracy'][i]:.3f} | "
        report += (
            f"{history['val_precision'][i]:.3f} | {history['val_recall'][i]:.3f} |"
        )

        # Mark best epoch
        if i == best_epoch - 1:
            report += " ⭐ Best"

        report += "\n"

    report += """
---

## Analysis

### Training Stability

"""

    # Check for overfitting
    train_val_loss_diff = history["loss"][-1] - history["val_loss"][-1]
    if train_val_loss_diff < -0.1:
        report += "⚠️ **Potential Overfitting**: Validation loss significantly lower than training loss. Consider:\n"
        report += "- Reducing regularization (dropout)\n"
        report += "- Increasing data augmentation\n\n"
    elif abs(train_val_loss_diff) < 0.05:
        report += "✅ **Good Balance**: Training and validation losses are close, indicating good generalization.\n\n"
    else:
        report += "⚠️ **Potential Underfitting**: Training loss much lower than validation loss. Consider:\n"
        report += "- Training for more epochs\n"
        report += "- Reducing regularization\n"
        report += "- Increasing model capacity\n\n"

    # Check recall (most important for fraud detection)
    if final_val_recall < 0.60:
        report += f"⚠️ **Low Recall ({final_val_recall:.1%})**: Model is missing too many fraud cases. Consider:\n"
        report += f"- Increasing class weight boost (currently {config.get('boost_factor', 3.0):.1f}x)\n"
        report += "- Lowering inference threshold\n"
        report += "- Training for more epochs\n\n"
    elif final_val_recall >= 0.70:
        report += f"✅ **Good Recall ({final_val_recall:.1%})**: Model is detecting most fraud cases.\n\n"
    else:
        report += f"ℹ️ **Moderate Recall ({final_val_recall:.1%})**: Acceptable but could be improved.\n\n"

    # Check precision
    if final_val_precision < 0.70:
        report += f"⚠️ **Low Precision ({final_val_precision:.1%})**: Too many false positives. Consider:\n"
        report += "- Decreasing class weight boost\n"
        report += "- Raising inference threshold\n"
        report += "- More training data for non-fraud class\n\n"
    elif final_val_precision >= 0.75:
        report += f"✅ **Good Precision ({final_val_precision:.1%})**: False positive rate is acceptable.\n\n"
    else:
        report += f"ℹ️ **Moderate Precision ({final_val_precision:.1%})**: Acceptable but could be improved.\n\n"

    report += """
### Recommendations

Based on the training results:

"""

    # Special note for interrupted training
    if interrupted:
        report += """⚠️ **Training was interrupted**:
   - The best model checkpoint has been saved automatically (via ModelCheckpoint callback)
   - You can either:
     1. Resume training by modifying the script to load the checkpoint
     2. Use the current best checkpoint for detection if results look promising
     3. Restart training from scratch

"""

    report += "1. **Model Performance**: "

    # Overall assessment
    if (
        final_val_acc >= 0.85
        and final_val_recall >= 0.65
        and final_val_precision >= 0.70
    ):
        report += "Model meets target metrics. Ready for deployment.\n"
    else:
        report += "Model could benefit from further tuning. See specific recommendations above.\n"

    report += f"""
2. **Next Steps**:
   - Run detection on test set: `uv run scripts/detect.py --forged-dir datasets/test/Fraud --authentic-dir datasets/test/Non-Fraud`
   - Find optimal threshold: `uv run scripts/tune.py`
   - If needed, adjust hyperparameters and retrain

---

## Files in This Run

- `REPORT.md` - This report
- `history.json` - Raw training history data
- `loss_curves.png` - Loss visualization
- `accuracy_curves.png` - Accuracy visualization
- `precision_curves.png` - Precision visualization
- `recall_curves.png` - Recall visualization
- `../verito.keras` - Trained model file (at models/ root, {model_path.stat().st_size / (1024 * 1024):.1f} MB)

---

## How to Use This Model

```sh
# Run detection
uv run scripts/detect.py \\
  --forged-dir datasets/test/Fraud \\
  --authentic-dir datasets/test/Non-Fraud \\
  --model-path {model_path_rel}

# Find optimal threshold
uv run scripts/tune.py \\
  --model-path {model_path_rel}
```

---

*Report generated automatically by training script*
"""

    # Write report
    with open(report_path, "w") as f:
        f.write(report)

    return report_path


def create_training_run_directory() -> Path:
    """
    Create timestamped directory for training run.

    Uses ISO 8601 basic format: YYYYMMDDTHHMMSS

    Returns:
        Path to created directory
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = Path("models") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_training_run(
    history: Dict[str, list],
    config: Dict[str, Any],
    training_time: float,
    model_path: Path,
) -> Path:
    """
    Save complete training run with report, history, and plots.

    Args:
        history: Training history from model.fit()
        config: Training configuration
        training_time: Total training time in seconds
        model_path: Path where model was saved

    Returns:
        Path to run directory
    """
    # Create run directory
    run_dir = create_training_run_directory()

    print(f"\n📊 Saving training run to: {run_dir}")

    # Save history
    history_path = save_training_history(history, run_dir)
    print(f"   History: {history_path.name}")

    # Generate plots
    plots = generate_training_plots(history, run_dir)
    print(f"   Plots: {', '.join(p.name for p in plots.values())}")

    # Generate report
    report_path = generate_report_md(
        history, config, training_time, run_dir, model_path
    )
    print(f"   Report: {report_path.name}")

    print("\n✅ Training run saved successfully!")
    print(f"   View report: {run_dir / 'REPORT.md'}")

    return run_dir
