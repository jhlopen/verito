"""
Train insurance fraud detection model using EfficientNetV2L.
"""

import sys
import time
from pathlib import Path
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from verito.model import build_model
from verito.data import create_data_generators, calculate_class_weights
from verito.report import save_training_run


def create_callbacks(model_save_path: Path, patience: int = 15):
    """Create training callbacks."""
    return [
        ModelCheckpoint(
            str(model_save_path), monitor="val_loss", save_best_only=True, verbose=1
        ),
        EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
        ),
    ]


def check_and_download_dataset(train_dir: Path, test_dir: Path) -> bool:
    """
    Check if dataset exists, and download if missing.

    Args:
        train_dir: Path to training data directory
        test_dir: Path to test data directory

    Returns:
        bool: True if dataset is available, False otherwise
    """
    # Check if both directories exist and have data
    train_fraud = train_dir / "Fraud"
    train_non_fraud = train_dir / "Non-Fraud"
    test_fraud = test_dir / "Fraud"
    test_non_fraud = test_dir / "Non-Fraud"

    dataset_exists = all(
        [
            train_fraud.exists(),
            train_non_fraud.exists(),
            test_fraud.exists(),
            test_non_fraud.exists(),
        ]
    )

    if dataset_exists:
        # Check if directories have actual images
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.tiff", "*.tif", "*.bmp"]
        has_images = (
            any(list(train_fraud.glob(ext)) for ext in image_extensions)
            and any(list(train_non_fraud.glob(ext)) for ext in image_extensions)
            and any(list(test_fraud.glob(ext)) for ext in image_extensions)
            and any(list(test_non_fraud.glob(ext)) for ext in image_extensions)
        )

        if has_images:
            return True

    # Dataset not found or incomplete
    print("⚠️  Dataset not found or incomplete\n")
    response = (
        input("Would you like to download the dataset from Kaggle now? [Y/n]: ")
        .strip()
        .lower()
    )

    if response in ["", "y", "yes"]:
        try:
            # Import and run download script
            from scripts.download_dataset import main as download_main

            download_main()
            return True
        except Exception as e:
            print(f"\n❌ Error downloading dataset: {e}")
            print(
                "   Please download it manually: uv run scripts/download_dataset.py\n"
            )
            return False
    else:
        print("\n❌ Dataset is required for training.")
        print("   Please download it manually: uv run scripts/download_dataset.py\n")
        return False


def main():
    """Main training function."""
    print("\n🚀 Starting Insurance Fraud Detection Model Training\n")

    # Paths
    project_root = Path(__file__).parent.parent
    train_dir = project_root / "datasets" / "train"
    test_dir = project_root / "datasets" / "test"
    model_dir = project_root / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "verito.keras"

    # Check if dataset exists, download if needed
    if not check_and_download_dataset(train_dir, test_dir):
        sys.exit(1)

    # Hyperparameters
    TARGET_SIZE = (480, 480)
    BATCH_SIZE = 16
    EPOCHS_PHASE1 = 15  # Train classification head
    EPOCHS_PHASE2 = 25  # Fine-tuning phase
    LR_PHASE1 = 0.001  # Initial learning rate
    LR_PHASE2 = 0.00001  # Fine-tuning learning rate
    PATIENCE = 8  # Early stopping patience
    VALIDATION_SPLIT = 0.2  # Validation split ratio
    BOOST_FACTOR = 3.0  # Class weight boost for fraud class

    print("📁 Configuration:")
    print("   Model: EfficientNetV2L")
    print(f"   Resolution: {TARGET_SIZE[0]}×{TARGET_SIZE[1]}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Train directory: {train_dir}")
    print(f"   Test directory: {test_dir}")
    print(f"   Model save path: {model_path}\n")

    # Load data
    print("📊 Loading data...")
    train_gen, val_gen, test_gen = create_data_generators(
        train_dir=train_dir,
        test_dir=test_dir,
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
    )

    print(f"   Training samples: {train_gen.samples}")
    print(f"   Validation samples: {val_gen.samples}")
    print(f"   Test samples: {test_gen.samples}\n")

    print("⚖️  Calculating class weights...")
    class_weights = calculate_class_weights(train_gen, boost_factor=BOOST_FACTOR)
    print(f"   Class 0 (Fraud): {class_weights[0]:.2f}")
    print(f"   Class 1 (Non-Fraud): {class_weights[1]:.2f}")
    print(f"   Effective ratio: {class_weights[0] / class_weights[1]:.1f}:1\n")

    # Calculate class imbalance ratio for model initialization
    class_counts = np.bincount(train_gen.classes)
    imbalance_ratio = class_counts[1] / class_counts[0]  # Non-Fraud / Fraud

    # Build model
    print("🏗️  Building EfficientNetV2L model...")
    model = build_model(
        input_size=(TARGET_SIZE[0], TARGET_SIZE[1], 3),
        dropout_rates=(0.5, 0.25),
        class_imbalance_ratio=imbalance_ratio,
    )

    # Freeze base model for Phase 1
    model.layers[0].trainable = False

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=LR_PHASE1),
        loss="binary_crossentropy",
        metrics=["accuracy", "precision", "recall"],
    )

    # Print model info
    total_params = model.count_params()
    trainable_params = sum([np.prod(v.shape) for v in model.trainable_weights])
    print(f"   Total parameters: {total_params:,}")
    print(
        f"   Trainable parameters (Phase 1): {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)\n"
    )

    # Create callbacks
    callbacks = create_callbacks(model_path, patience=PATIENCE)

    # Start timing
    training_start_time = time.time()

    # Variables to track training progress
    history1 = None
    history2 = None
    training_interrupted = False

    try:
        # ============================================================================
        # Phase 1: Train classification head with frozen base
        # ============================================================================
        print("=" * 70)
        print("🎯 Phase 1: Training Classification Head (Frozen Base)")
        print("=" * 70)
        print(f"   Epochs: {EPOCHS_PHASE1}")
        print(f"   Learning rate: {LR_PHASE1}")
        print(f"   Early stopping patience: {PATIENCE} epochs\n")

        history1 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS_PHASE1,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )

        # ============================================================================
        # Phase 2: Fine-tune entire model
        # ============================================================================
        print("\n" + "=" * 70)
        print("🔥 Phase 2: Fine-Tuning Entire Model (Unfrozen Base)")
        print("=" * 70)
        print(f"   Epochs: {EPOCHS_PHASE2}")
        print(f"   Learning rate: {LR_PHASE2} (100x lower)\n")

        # Unfreeze base model
        model.layers[0].trainable = True

        # Recompile with lower learning rate
        model.compile(
            optimizer=Adam(learning_rate=LR_PHASE2),
            loss="binary_crossentropy",
            metrics=["accuracy", "precision", "recall"],
        )

        # Print updated trainable params
        trainable_params = sum([np.prod(v.shape) for v in model.trainable_weights])
        print(
            f"   Trainable parameters (Phase 2): {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)\n"
        )

        history2 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS_PHASE2,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )

        # ============================================================================
        # Final Evaluation
        # ============================================================================
        print("\n" + "=" * 70)
        print("📈 Final Evaluation on Test Set")
        print("=" * 70)

        test_loss, test_acc, test_precision, test_recall = model.evaluate(
            test_gen, verbose=0
        )

        # Calculate F1 score
        f1_score = (
            2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-7)
        )

        # Calculate total training time
        total_seconds = int(time.time() - training_start_time)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        if hours > 0:
            time_str = f"{hours}h {minutes}m {seconds}s"
        else:
            time_str = f"{minutes}m {seconds}s"

        print("\n✅ Training Complete!")
        print("\n📊 Test Set Performance:")
        print(f"   Accuracy:  {test_acc:.2%}")
        print(f"   Precision: {test_precision:.2%}")
        print(f"   Recall:    {test_recall:.2%}")
        print(f"   F1 Score:  {f1_score:.2%}")
        print(f"\n💾 Model saved to: {model_path}")
        print(f"   Model size: {model_path.stat().st_size / (1024 * 1024):.1f} MB\n")

        # Training summary
        print("📝 Training Summary:")
        print(f"   Phase 1 epochs: {len(history1.history['loss'])}")
        print(f"   Phase 2 epochs: {len(history2.history['loss'])}")
        print(f"   Total training time: {time_str}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user (Ctrl+C)")
        training_interrupted = True

    # ============================================================================
    # Generate Report (even if interrupted)
    # ============================================================================

    # Check if we have any training history to save
    if history1 is None and history2 is None:
        print("❌ No training history available to generate report.")
        print("   Training was interrupted before completing any epochs.\n")
        return None

    # Combine histories from completed phases
    combined_history = {}
    if history1 is not None:
        for key in history1.history.keys():
            combined_history[key] = history1.history[key]

    if history2 is not None:
        for key in history2.history.keys():
            if key in combined_history:
                combined_history[key] = combined_history[key] + history2.history[key]
            else:
                combined_history[key] = history2.history[key]

    # Build configuration dict for report
    training_config = {
        "target_size": TARGET_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs_phase1": EPOCHS_PHASE1,
        "epochs_phase2": EPOCHS_PHASE2,
        "patience": PATIENCE,
        "lr_phase1": LR_PHASE1,
        "lr_phase2": LR_PHASE2,
        "train_samples": train_gen.samples,
        "val_samples": val_gen.samples,
        "boost_factor": BOOST_FACTOR,
        "validation_split": VALIDATION_SPLIT,
        "interrupted": training_interrupted,
    }

    # Calculate actual training time
    training_time = time.time() - training_start_time

    if training_interrupted:
        print("\n📊 Generating partial training report...")
        if history1 and history2:
            print(f"   Phase 1: {len(history1.history['loss'])} epochs completed")
            print(f"   Phase 2: {len(history2.history['loss'])} epochs interrupted")
        elif history1:
            print(f"   Phase 1: {len(history1.history['loss'])} epochs interrupted")

        # Calculate time summary
        total_seconds = int(training_time)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        if hours > 0:
            time_str = f"{hours}h {minutes}m {seconds}s"
        else:
            time_str = f"{minutes}m {seconds}s"
        print(f"   Training time before interruption: {time_str}")

    # Save training run with report, history, and plots
    save_training_run(
        history=combined_history,
        config=training_config,
        training_time=training_time,
        model_path=model_path,
    )

    if not training_interrupted:
        print("\n🎉 Ready for detection!")
        print(
            "   Run: uv run scripts/detect.py --forged-dir datasets/test/Fraud --authentic-dir datasets/test/Non-Fraud\n"
        )
    else:
        print("\n⚠️  Training was interrupted, but report has been saved.")
        print(f"   Best model checkpoint saved to: {model_path}")
        print("   You can resume training or use the best checkpoint for detection.\n")

    return model_path


if __name__ == "__main__":
    main()
