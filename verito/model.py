"""Model architecture and loading."""

import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.applications import EfficientNetV2L
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.initializers import Constant


def build_model(
    input_size=(480, 480, 3),
    dropout_rates=(0.5, 0.25),
    class_imbalance_ratio=None,
):
    """
    Build EfficientNetV2L model for forgery detection.

    Args:
        input_size: Input image dimensions (height, width, channels)
        dropout_rates: Tuple of (dropout1, dropout2) for regularization
        class_imbalance_ratio: Ratio of negative/positive samples (e.g., 25.0 for 25:1)
                              Used to initialize output bias for faster convergence

    Returns:
        Keras Sequential model
    """
    base_model = EfficientNetV2L(
        include_top=False, weights="imagenet", input_shape=input_size
    )

    output_bias = None
    if class_imbalance_ratio is not None:
        # Calculate initial bias for output layer to handle class imbalance
        initial_bias = np.log(1.0 / class_imbalance_ratio)
        output_bias = Constant(initial_bias)
        print(
            f"   Setting output bias to {initial_bias:.4f} (for {class_imbalance_ratio:.1f}:1 imbalance)"
        )

    model = Sequential(
        [
            base_model,
            GlobalAveragePooling2D(),
            BatchNormalization(),
            Dense(256, activation="relu"),
            Dropout(dropout_rates[0]),
            Dense(128, activation="relu"),
            Dropout(dropout_rates[1]),
            Dense(1, activation="sigmoid", bias_initializer=output_bias),
        ]
    )

    return model


def load_detector(model_path: str | Path):
    """
    Load trained model from disk or Hugging Face Hub.

    Args:
        model_path: Path to saved .keras model file or Hugging Face Hub path
                   (e.g., "hf://username/repo")

    Returns:
        Loaded Keras model
    """
    return load_model(model_path)
