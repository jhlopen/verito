"""Detection script for forged image detection."""

import sys
import argparse
from pathlib import Path
import numpy as np

from verito.model import load_detector
from verito.data import load_and_predict_images
from verito.metrics import calculate_metrics, print_evaluation_report
from scripts.train import main as train_main


def main():
    """Run forgery detection on test images."""
    project_root = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(
        description="Detect forged images and evaluate performance"
    )
    parser.add_argument(
        "--forged-dir",
        type=Path,
        required=True,
        help="Directory containing forged images",
    )
    parser.add_argument(
        "--authentic-dir",
        type=Path,
        required=True,
        help="Directory containing authentic images",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to trained model (default: models/verito.keras)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (default: 0.5)",
    )

    args = parser.parse_args()

    print("\n🔍 Forged Image Detection\n")

    # Validate directories
    if not args.forged_dir.exists():
        print(f"❌ Error: Forged directory not found: {args.forged_dir}")
        sys.exit(1)

    if not args.authentic_dir.exists():
        print(f"❌ Error: Authentic directory not found: {args.authentic_dir}")
        sys.exit(1)

    # Set default model path
    if args.model_path is None:
        args.model_path = project_root / "models" / "verito.keras"

    if not args.model_path.exists():
        print(f"⚠️  Model not found at {args.model_path}\n")

        # Prompt user to load from the Hugging Face Hub
        response = (
            input(
                "Would you like to load the pre-trained model from Hugging Face? [Y/n]: "
            )
            .strip()
            .lower()
        )

        if response in ["", "y", "yes"]:
            print("📥 Using model from Hugging Face Hub (jhlopen/verito)...\n")
            args.model_path = "hf://jhlopen/verito"
        else:
            # Prompt user to train
            response = (
                input("Would you like to train the model now? [Y/n]: ").strip().lower()
            )

            if response in ["", "y", "yes"]:
                print("\n🚀 Starting training...\n")
                try:
                    train_main()
                    print("\n✅ Training complete! Now running detection...\n")
                except Exception as e:
                    print(f"❌ Training failed: {e}")
                    sys.exit(1)
            else:
                print("Training cancelled. Please train the model first:")
                print("  uv run scripts/train.py")
                sys.exit(1)

    print("📁 Configuration:")
    print(f"   Forged images: {args.forged_dir}")
    print(f"   Authentic images: {args.authentic_dir}")
    print(f"   Model: {args.model_path}")
    print(f"   Threshold: {args.threshold}\n")

    # Load model
    print("🤖 Loading model...")
    model = load_detector(args.model_path)
    print("   Model loaded successfully!\n")

    # Process images
    print("📊 Processing images...")

    # Predict on forged images (label = 0)
    forged_preds, forged_raw = load_and_predict_images(
        model, args.forged_dir, threshold=args.threshold
    )

    # Predict on authentic images (label = 1)
    authentic_preds, authentic_raw = load_and_predict_images(
        model, args.authentic_dir, threshold=args.threshold
    )

    if len(forged_preds) == 0 or len(authentic_preds) == 0:
        print("❌ Error: No images found to process")
        sys.exit(1)

    # Create true labels
    y_true = np.concatenate(
        [
            np.zeros(len(forged_preds)),  # Forged = 0
            np.ones(len(authentic_preds)),  # Authentic = 1
        ]
    )

    # Combine predictions
    y_pred = np.concatenate([forged_preds, authentic_preds])

    # Calculate metrics
    print("   Done!\n")
    metrics = calculate_metrics(y_true, y_pred)

    # Print report
    print_evaluation_report(
        metrics=metrics,
        total_forged=len(forged_preds),
        total_authentic=len(authentic_preds),
    )


if __name__ == "__main__":
    main()
