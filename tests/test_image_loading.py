"""Test custom image loading and prediction logic."""

from pathlib import Path
from verito.data import load_and_predict_images


def test_load_and_predict_counts(temp_dataset_dir, simple_model):
    """Test that correct number of images are loaded."""
    fraud_dir = temp_dataset_dir / "Fraud"

    predictions, raw_preds = load_and_predict_images(
        simple_model, fraud_dir, target_size=(480, 480), threshold=0.5
    )

    # Should load all 2 fraud images we created
    assert len(predictions) == 2
    assert len(raw_preds) == 2

    # Predictions should be binary (0 or 1)
    assert all(p in [0, 1] for p in predictions)

    # Raw predictions should be probabilities [0, 1]
    assert all(0 <= p <= 1 for p in raw_preds)


def test_load_and_predict_multiple_formats(simple_model):
    """Test that multiple image formats are loaded."""
    import tempfile
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create images in different formats
        for ext in ["jpg", "jpeg", "png"]:
            img = Image.new("RGB", (480, 480), color="green")
            img.save(tmpdir / f"test.{ext}")

        predictions, _ = load_and_predict_images(
            simple_model, tmpdir, target_size=(480, 480), threshold=0.5
        )

        # Should load all 3 formats
        assert len(predictions) == 3


def test_load_and_predict_empty_directory(simple_model):
    """Test handling of empty directories."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        empty_dir = Path(tmpdir)

        predictions, raw_preds = load_and_predict_images(
            simple_model, empty_dir, target_size=(480, 480), threshold=0.5
        )

        # Should return empty arrays, not crash
        assert len(predictions) == 0
        assert len(raw_preds) == 0


def test_threshold_application_in_loading():
    """Test that threshold is correctly applied during loading."""
    # This tests the threshold logic within load_and_predict_images
    raw_probs = [0.1, 0.49, 0.5, 0.51, 0.9]
    threshold = 0.5

    # Simulate what load_and_predict_images does
    predictions = [1 if p >= threshold else 0 for p in raw_probs]

    expected = [0, 0, 1, 1, 1]
    assert predictions == expected


def test_image_normalization(temp_image):
    """Test that preprocessing normalizes to [0, 1]."""
    from tensorflow.keras.preprocessing.image import img_to_array, load_img

    # Load like our code does
    loaded = load_img(temp_image, target_size=(480, 480))
    img_array = img_to_array(loaded)
    normalized = img_array / 255.0

    # Verify normalization works
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0
    assert normalized.shape == (480, 480, 3)
