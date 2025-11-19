"""Test custom class weight calculation with boost factor."""

import pytest
from verito.data import calculate_class_weights


def test_class_weights_with_3x_boost(temp_dataset_dir):
    """Test the 3x boost factor logic."""
    # Create a mock generator with class information
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(rescale=1.0 / 255)
    generator = datagen.flow_from_directory(
        temp_dataset_dir,
        target_size=(480, 480),
        batch_size=4,
        class_mode="binary",
        shuffle=False,
    )

    # Calculate with 3x boost
    weights = calculate_class_weights(generator, boost_factor=3.0)

    assert isinstance(weights, dict)
    assert 0 in weights  # Fraud class
    assert 1 in weights  # Non-fraud class

    # Fraud weight should be significantly higher due to 3x boost
    assert weights[0] > weights[1]

    # Check that boost factor actually applies
    weights_1x = calculate_class_weights(generator, boost_factor=1.0)
    weights_3x = calculate_class_weights(generator, boost_factor=3.0)

    # 3x boost should give approximately 3x the weight
    assert weights_3x[0] == pytest.approx(weights_1x[0] * 3.0, rel=0.01)
    assert weights_3x[1] == pytest.approx(weights_1x[1], rel=0.01)


def test_class_weights_formula(temp_dataset_dir):
    """Test the weight calculation formula."""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(rescale=1.0 / 255)
    generator = datagen.flow_from_directory(
        temp_dataset_dir,
        target_size=(480, 480),
        batch_size=4,
        class_mode="binary",
        shuffle=False,
    )

    # With 2 fraud, 10 non-fraud, boost=3.0:
    total = 12
    fraud_count = 2
    non_fraud_count = 10
    boost = 3.0

    # Expected: fraud_weight = (12 / (2 * 2)) * 3.0 = 9.0
    # Expected: non_fraud_weight = (12 / (2 * 10)) = 0.6
    expected_fraud = (total / (2 * fraud_count)) * boost
    expected_non_fraud = total / (2 * non_fraud_count)

    weights = calculate_class_weights(generator, boost_factor=boost)

    assert weights[0] == pytest.approx(expected_fraud, rel=0.01)
    assert weights[1] == pytest.approx(expected_non_fraud, rel=0.01)
