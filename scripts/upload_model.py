"""
Upload trained model to Hugging Face Hub.
"""

import argparse
from pathlib import Path
import sys
from verito.model import load_detector


def upload_to_huggingface(
    repo_id: str,
    model_path: Path,
):
    """
    Upload model to Hugging Face Hub.

    Args:
        repo_id: Hugging Face repo ID (e.g., "jhlopen/verito")
        model_path: Path to the model file
    """
    # Load model
    print("📦 Loading model from {model_path}...")
    model = load_detector(model_path)

    # Upload model
    print(f"📤 Uploading model to Hugging Face Hub: {repo_id}...")
    model.save(f"hf://{repo_id}")

    print("\n✅ Upload complete!\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Upload trained model to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run scripts/upload_model.py --repo-id jhlopen/verito
  uv run scripts/upload_model.py --repo-id jhlopen/verito --model-path models/verito.keras

Environment Variables:
  HF_TOKEN: Hugging Face API token
""",
    )

    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face repository ID (e.g., 'jhlopen/verito')",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to model file (default: models/verito.keras)",
    )

    args = parser.parse_args()

    # Get model path
    if args.model_path:
        model_path = args.model_path
    else:
        project_root = Path(__file__).parent.parent
        model_path = project_root / "models" / "verito.keras"

    # Check if model exists
    if not model_path.exists():
        print(f"❌ Error: Model not found at {model_path}")
        print("\nTrain a model first:")
        print("   uv run scripts/train.py")
        sys.exit(1)

    # Upload
    upload_to_huggingface(
        repo_id=args.repo_id,
        model_path=model_path,
    )


if __name__ == "__main__":
    main()
