"""Shared test fixtures."""

import pytest
from pathlib import Path
from PIL import Image
import tempfile


@pytest.fixture
def temp_image():
    """Create a temporary test image."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img = Image.new("RGB", (480, 480), color="red")
        img.save(f.name)
        yield Path(f.name)
        Path(f.name).unlink()  # Cleanup


@pytest.fixture
def temp_dataset_dir():
    """Create a minimal test dataset with realistic structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create Fraud and Non-Fraud directories
        fraud_dir = tmpdir / "Fraud"
        non_fraud_dir = tmpdir / "Non-Fraud"
        fraud_dir.mkdir()
        non_fraud_dir.mkdir()

        # Create 2 fraud, 10 non-fraud (5:1 ratio)
        for i in range(2):
            img = Image.new("RGB", (480, 480), color="red")
            img.save(fraud_dir / f"fraud_{i}.jpg")

        for i in range(10):
            img = Image.new("RGB", (480, 480), color="blue")
            img.save(non_fraud_dir / f"authentic_{i}.jpg")

        yield tmpdir


@pytest.fixture
def simple_model():
    """Create a minimal model for testing image loading logic."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten, Input

    model = Sequential(
        [Input(shape=(480, 480, 3)), Flatten(), Dense(1, activation="sigmoid")]
    )
    return model
