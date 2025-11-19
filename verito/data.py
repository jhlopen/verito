"""Data loading and preprocessing utilities."""

import numpy as np
from pathlib import Path
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array,
)


def create_data_generators(
    train_dir: Path,
    test_dir: Path,
    target_size: tuple[int, int] = (480, 480),
    batch_size: int = 16,
    validation_split: float = 0.2,
) -> tuple:
    """
    Create training, validation, and test data generators with augmentation.

    Args:
        train_dir: Path to training data directory
        test_dir: Path to test data directory
        target_size: Image size to resize to (height, width)
        batch_size: Batch size for training
        validation_split: Fraction of training data to use for validation

    Returns:
        Tuple of (train_generator, val_generator, test_generator)
    """
    # Heavy augmentation for training to handle class imbalance
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=validation_split,
    )

    # Only rescaling for validation and test
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Training generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="training",
        shuffle=True,
    )

    # Validation generator
    val_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="validation",
        shuffle=False,
    )

    # Test generator
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )

    return train_generator, val_generator, test_generator


def calculate_class_weights(train_generator, boost_factor: float = 3.0) -> dict:
    """
    Calculate class weights to handle imbalanced dataset.

    Uses manual calculation with boost factor for fraud class to handle
    severe class imbalance (e.g., 25:1 ratio).

    Args:
        train_generator: Keras ImageDataGenerator with class information
        boost_factor: Multiplier for fraud class weight (default: 3.0)

    Returns:
        Dictionary mapping class indices to weights
        {0: fraud_weight, 1: non_fraud_weight}
    """
    # Get class counts
    class_counts = np.bincount(train_generator.classes)

    # Calculate weights with stronger emphasis on minority class
    # Formula: total / (n_classes * class_count) * boost_factor
    total = len(train_generator.classes)

    weight_fraud = (total / (2 * class_counts[0])) * boost_factor
    weight_non_fraud = total / (2 * class_counts[1])

    weights = np.array([weight_fraud, weight_non_fraud])

    class_weights_dict = {i: weight for i, weight in enumerate(weights)}

    print("\n📊 Class Distribution:")
    print(f"   Class 0 (Fraud): {class_counts[0]} images (weight: {weights[0]:.2f})")
    print(
        f"   Class 1 (Non-Fraud): {class_counts[1]} images (weight: {weights[1]:.2f})"
    )
    print(f"   Imbalance ratio: 1:{class_counts[1] / class_counts[0]:.1f}")
    print(f"   Weight ratio: {weights[0] / weights[1]:.1f}:1\n")

    return class_weights_dict


def load_and_predict_images(
    model,
    image_dir: Path,
    target_size: tuple[int, int] = (480, 480),
    threshold: float = 0.5,
) -> tuple[np.ndarray, list]:
    """
    Load images from directory and make predictions.

    Args:
        model: Trained Keras model
        image_dir: Directory containing images
        target_size: Image size for model input (height, width)
        threshold: Classification threshold (< threshold = fraud)

    Returns:
        Tuple of (predictions array, raw prediction values)
        - predictions: Binary predictions (0=fraud, 1=authentic)
        - raw_preds: Raw probability values from model
    """
    predictions = []
    raw_preds = []

    # Support multiple insurance image formats
    image_files = sorted(
        list(image_dir.glob("*.jpg"))
        + list(image_dir.glob("*.jpeg"))
        + list(image_dir.glob("*.png"))
        + list(image_dir.glob("*.tiff"))
        + list(image_dir.glob("*.tif"))
        + list(image_dir.glob("*.bmp"))
    )

    if not image_files:
        print(f"⚠️  No images found in {image_dir}")
        return np.array([]), []

    print(f"📁 Loading {len(image_files)} images from {image_dir.name}...")

    for img_path in image_files:
        # Load and preprocess image
        img = load_img(img_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        pred = model.predict(img_array, verbose=0)[0][0].item()
        raw_preds.append(pred)

        # Apply threshold (< threshold = fraud = 0, >= threshold = authentic = 1)
        predictions.append(1 if pred >= threshold else 0)

    return np.array(predictions), raw_preds
