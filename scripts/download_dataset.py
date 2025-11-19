"""Download and set up car insurance fraud detection dataset from Kaggle."""

import sys
import shutil
from pathlib import Path
import kagglehub


def main():
    """
    Download car insurance fraud detection dataset from Kaggle and organize it.

    This script downloads the dataset from Kaggle using `kagglehub` (which
    caches the download) and organizes it into the project's `datasets/`
    directory with the following structure:
        datasets/
        ├── train/
        │   ├── Fraud/
        │   └── Non-Fraud/
        └── test/
            ├── Fraud/
            └── Non-Fraud/

    The script handles the nested Kaggle dataset structure automatically and
    counts the images in each category to verify the download.
    """
    print("\n📥 Downloading Car Insurance Fraud Detection Dataset\n")

    project_root = Path(__file__).parent.parent
    target_path = project_root / "datasets"

    try:
        # Download dataset from Kaggle
        print("🔄 Downloading from Kaggle (cached if previously downloaded)...")
        path = kagglehub.dataset_download("pacificrm/car-insurance-fraud-detection")
        print(f"   Cached at: {path}\n")

        # Navigate to the nested structure
        source_path = (
            Path(path) / "Insurance-Fraud-Detection" / "Insurance-Fraud-Detection"
        )

        if not source_path.exists():
            print(f"❌ Error: Expected nested path not found: {source_path}")
            print("   The dataset structure may have changed.")
            sys.exit(1)

        # Create datasets directory if it doesn't exist
        target_path.mkdir(exist_ok=True)

        # Copy train and test folders
        print("📁 Organizing dataset into project structure...")
        folders_copied = 0

        for folder in ["train", "test"]:
            source_folder = source_path / folder
            target_folder = target_path / folder

            if not source_folder.exists():
                print(f"⚠️  Warning: {folder} folder not found at {source_folder}")
                continue

            # Remove existing target folder if it exists
            if target_folder.exists():
                print(f"   Removing existing {folder}/ folder...")
                shutil.rmtree(target_folder)

            # Copy the folder
            print(f"   Copying {folder}/ folder...")
            shutil.copytree(source_folder, target_folder)

            # Count images in each subfolder
            image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.tiff", "*.tif", "*.bmp"]
            fraud_count = sum(
                len(list((target_folder / "Fraud").glob(ext)))
                for ext in image_extensions
            )
            non_fraud_count = sum(
                len(list((target_folder / "Non-Fraud").glob(ext)))
                for ext in image_extensions
            )

            print(f"   ✓ {folder}/ copied successfully")
            print(f"      - Fraud: {fraud_count} images")
            print(f"      - Non-Fraud: {non_fraud_count} images")
            folders_copied += 1

        if folders_copied == 0:
            print("\n❌ Error: No folders were copied successfully")
            sys.exit(1)

        print("\n✅ Dataset setup complete!")
        print(f"   Location: {target_path}")
        print("\n🚀 Next steps:")
        print("   1. Train model: uv run scripts/train.py")
        print(
            "   2. Run detection: uv run scripts/detect.py --forged-dir datasets/test/Fraud --authentic-dir datasets/test/Non-Fraud\n"
        )

    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("   Please check your network connection.")
        sys.exit(1)


if __name__ == "__main__":
    main()
