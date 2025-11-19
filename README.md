# Forged Image Detection

A deep learning solution for detecting forged (digitally altered) images using transfer learning with EfficientNetV2L.

## 🎯 Use Case: Insurance Claim Validation

This system detects common insurance fraud forgeries:

- **Damage added to vehicles** (scratches, dents, broken lamps)
- **Weather manipulation** (rainy conditions made to appear sunny)
- **License plate alterations** (numbers changed)

**Supported formats**: JPEG, PNG, TIFF, BMP

## Assumptions

This solution makes the following assumptions:

### Data Assumptions

1. **Forgery extent**: Modifications affect approximately 10% of image area (car damage, weather changes, license plate alterations)
2. **Image quality**: Standard insurance claim quality (e.g. from phone cameras with basic JPEG compression)
3. **Legitimate edits**: Cropping, resizing, and basic color correction are NOT considered forgeries
4. **File formats**: JPEG, PNG, TIFF, BMP

### Operational Assumptions

1. **Evaluation context**: Forged and authentic images are provided in separate directories with known ground truth
2. **Computational resources**: GPU is available for training (CPU-only training is possible but slower)
3. **Dataset representativeness**: Training data represents the types of forgeries expected in production
4. **Human review**: High-stakes decisions should include manual review, especially for predictions near the threshold

### Forgery Type Focus

Based on the challenge requirements, this model focuses on:

- ✅ **Added damage**: Scratches, dents, broken parts added to vehicles
- ✅ **Weather manipulation**: Rainy conditions made to appear sunny or vice versa
- ⚠️ **License plate alterations**: Character changes (challenging due to small size - see Limitations)
- ❌ **Complete image replacement**: Out of scope (requires metadata analysis)
- ❌ **Metadata-only manipulation**: EXIF tampering not detected (pixel analysis only)

## Approach

This section explains the technical approach and design rationale.

### Architecture Decision

This solution uses **EfficientNetV2L** (~120M parameters) fine-tuned for binary classification. This model was chosen because:

- **High capacity**: Necessary for detecting subtle forgeries like small damage, weather changes, and license plate alterations
- **Proven performance**: State-of-the-art accuracy on ImageNet with excellent transfer learning characteristics
- **Resolution support**: Efficiently handles 480×480 images, balancing detail capture and computational efficiency
- **Feature richness**: Deep architecture captures both low-level artifacts (compression, noise) and high-level semantic inconsistencies

**Alternatives considered**:

- **ResNet50**: Too shallow (26M params) - missed subtle forgeries in testing
- **EfficientNetV2M**: Too unstable in training
- **EfficientNetB7**: Comparable accuracy but significantly slower
- **Custom CNN**: Insufficient dataset for training

### Training Strategy

**Two-phase fine-tuning approach**:

1. **Phase 1 (15 epochs)**: Train classification head only with frozen base model

   - Fast adaptation to fraud detection task (~364K trainable parameters)
   - Prevents catastrophic forgetting of ImageNet features
   - Learning rate: 0.001

2. **Phase 2 (25 epochs)**: Fine-tune entire model
   - Refines deep features for fraud-specific patterns
   - Lower learning rate (0.00001) prevents overfitting
   - All ~120M parameters trainable

**Class imbalance handling** (25:1 Non-Fraud:Fraud ratio):

The training dataset has severe imbalance (~5,000 authentic vs ~200 forged). To address this:

- **3x boosted class weights**: Makes fraud samples 3x more important during training
- **Extensive data augmentation**: Rotation, shift, zoom, flip (effectively increases fraud dataset from 200 to ~2000 variations)
- **Validation split (20%)**: Monitors performance on unseen data to detect overfitting
- **Early stopping**: Halts training if validation loss doesn't improve for 8 epochs

### Why This Works

**Transfer learning** from ImageNet provides robust low-level feature detectors (edges, textures, gradients) that transfer well to forgery detection. These features detect:

- Compression artifacts from photo editing software
- Inconsistent lighting and shadows
- Edge discontinuities from splicing
- Texture inconsistencies from cloning/patching

**Fine-tuning** adapts these generic features to insurance fraud patterns, learning to recognize:

- Typical damage patterns (realistic vs. added)
- Weather-related lighting characteristics
- Natural vs. manipulated sky textures
- Authentic vs. altered vehicle components

## Limitations

This solution has the following known limitations:

### 1. Small Region Forgeries (License Plates)

- **Issue**: License plate number changes are difficult to detect due to small size (typically < 2% of image)
- **Impact**: May miss fraud if only small text/numbers are altered
- **Mitigation**: Enforce image standards or combine with OCR-based verification for critical text fields

### 2. Professional/Sophisticated Forgeries

- **Issue**: Highly sophisticated edits by professional photo manipulators may evade detection
- **Impact**: Model trained on typical insurance fraud patterns; novel techniques may not be detected
- **Mitigation**: Regularly retrain with new fraud examples from production

### 3. Heavily Compressed or Low-Quality Images

- **Issue**: Extreme JPEG compression can mask forgery artifacts or create false positives
- **Impact**: Reduced accuracy on heavily compressed images (< 70% quality)
- **Mitigation**: Enforce minimum quality standards for claim image submissions

### 4. Dataset Representativeness

- **Issue**: Model performance depends on similarity between training data and production data
- **Impact**: Different fraud patterns or new editing tools may not be detected
- **Mitigation**: Continuous model monitoring and retraining with production feedback

### 5. No Forgery Localization

- **Issue**: Model only predicts forged/authentic, doesn't indicate where the forgery is located
- **Impact**: Cannot show insurance adjusters which specific part of the image is suspicious
- **Mitigation**: Consider adding explainability features (GradCAM, attention maps) in future versions

### 6. Single Image Analysis Only

- **Issue**: Doesn't compare multiple images from the same incident for consistency
- **Impact**: Misses cross-image inconsistencies (e.g., different weather in photos from same time)
- **Mitigation**: Implement cross-image consistency checks in production pipeline

### 7. Metadata Ignored

- **Issue**: Only analyzes pixel data, ignores EXIF metadata (timestamps, GPS, camera model)
- **Impact**: Misses metadata-based fraud indicators (e.g., image edited in Photoshop but claimed as original)
- **Mitigation**: Combine with metadata analysis tools for comprehensive fraud detection

### 8. Legitimate Heavy Editing May Be Flagged

- **Issue**: Authentic images with heavy legitimate editing (Instagram filters, color adjustments) may be flagged
- **Impact**: False positives on genuinely edited but not forged images
- **Mitigation**: Establish clear guidelines on acceptable image editing for insurance claims

## Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) - Python package and project manager
- See `pyproject.toml` for full dependencies

## Installation

Clone repository and install dependencies:

```sh
git clone https://github.com/jhlopen/verito.git
cd verito
uv sync
```

## Usage

### Detection

Detect forged images and evaluate performance:

```sh
uv run scripts/detect.py \
  --forged-dir PATH \
  --authentic-dir PATH \
  [--model-path PATH] \
  [--threshold FLOAT]
```

**Arguments:**

- `--forged-dir`: Directory containing known forged images (required)
- `--authentic-dir`: Directory containing known authentic images (required)
- `--model-path`: Path to trained model (default: `models/verito.keras`)
- `--threshold`: Classification threshold (default: 0.5)

If the model doesn't exist locally, the script will prompt you to download the pre-trained model from the Hugging Face Hub.

**Example:**

```sh
uv run scripts/detect.py \
  --forged-dir datasets/test/Fraud \
  --authentic-dir datasets/test/Non-Fraud
```

### Dataset Setup

Download the car insurance fraud detection dataset from Kaggle:

```sh
uv run scripts/download_dataset.py
```

This script:

- Downloads the [Car Insurance Fraud Detection dataset](https://www.kaggle.com/datasets/pacificrm/car-insurance-fraud-detection) from Kaggle (cached for subsequent runs)
- Organizes it into `datasets/train/` and `datasets/test/` directories
- Verifies the download by counting images in each category
- Creates the proper directory structure for training

**Note:** The training script will automatically prompt to download the dataset if it's not found, so this step is optional.

### Training

Train model:

```sh
uv run scripts/train.py
```

Each training run automatically generates:

- `REPORT.md` - Training report with analysis
- `history.json` - Raw training data
- `loss_curves.png` - Loss visualization with best epoch
- `accuracy_curves.png` - Accuracy trends
- `precision_curves.png` - Precision (false positive rate)
- `recall_curves.png` - Recall (fraud detection rate)
- Saved to `models/YYYYMMDDTHHMMSS/` (ISO 8601 timestamp)
- Model saved to `models/verito.keras`

**Example:** View a training report at `models/20251119T133742/REPORT.md`

### Tuning

Find the optimal classification threshold using validation data:

```sh
uv run scripts/tune.py \
  --fraud-dir datasets/test/Fraud \
  --authentic-dir datasets/test/Non-Fraud
```

This script:

- Tests multiple thresholds (0.1 to 0.9)
- Shows metrics for each threshold
- Recommends optimal thresholds for different objectives
- Creates visualization plot

### Uploading to Hugging Face

Upload your trained model to Hugging Face Hub:

```sh
export HF_TOKEN=hf_xxxxx
uv run scripts/upload_model.py --repo-id jhlopen/verito
```

Get your Hugging Face token from: https://huggingface.co/settings/tokens

**Arguments:**

- `--repo-id`: Hugging Face repository ID (required, e.g., "jhlopen/verito")
- `--model-path`: Path to model file (default: `models/verito.keras`)

## Understanding Results

### Classification Threshold

The model outputs a probability between 0 and 1:

- **< threshold**: Classified as fraud (0)
- **≥ threshold**: Classified as authentic (1)

Default: **0.5** (standard for binary classification)

**When to adjust:**

| Threshold | Use Case            | Trade-off            |
| --------- | ------------------- | -------------------- |
| 0.2-0.4   | Reduce false alarms | May miss some fraud  |
| 0.5       | Balanced (default)  | Standard             |
| 0.6-0.7   | Catch more fraud    | More false positives |

### Metrics Explained

- **Accuracy**: Overall correctness
- **Precision**: Of images flagged as fraud, how many are actually fraud
- **Recall**: Of actual fraud images, how many did we detect
- **F1 Score**: Harmonic mean of precision and recall

**For fraud detection, recall is most important** - we want to catch fraud even if it means some false positives.

## Design Trade-offs

This section documents key trade-offs and design decisions made during development.

### 1. Model Size vs. Accuracy

**Decision**: Use EfficientNetV2L (~120M params, ~450MB)

**Trade-off**:

- ✅ **Pro**: Higher accuracy for subtle forgeries
- ❌ **Con**: Large model size, slower inference
- 🔄 **Alternative**: EfficientNetV2S (22M params, ~90MB) - faster but lower accuracy

**Justification**: Insurance fraud detection prioritizes accuracy over speed. Even a few minutes per image is acceptable for claim processing, but missing fraud is very costly.

### 2. Resolution: 480×480 vs. Higher/Lower

**Decision**: 480×480 resolution

**Trade-off**:

- ✅ **Pro**: Good detail preservation for damage/weather detection
- ✅ **Pro**: Fits in GPU memory with batch size 16
- ❌ **Con**: May miss very small details (license plate numbers)
- 🔄 **Alternative**: 224×224 (faster, less accurate) or 640×640 (better detail, slower)

**Justification**: License plate fraud detection is inherently difficult (acknowledged limitation). 480×480 provides good balance for car damage and weather manipulation.

### 3. False Positive vs. False Negative Balance

**Decision**: Default threshold 0.5 (balanced), with tuning tool provided

We provide the threshold tuning tool (`tune.py`) because the optimal balance between fraud detected and false alarm rate depends on:

- Cost of missing fraud vs. cost of investigating false alarms
- Human review capacity
- Risk tolerance

**Recommendation**: Run `tune.py` on validation data to find your optimal threshold.

### 4. Training Time vs. Accuracy

**Decision**: Two-phase training (Phase 1: 15 epochs, Phase 2: 25 epochs)

**Trade-off**:

- ✅ **Pro**: Better convergence, prevents catastrophic forgetting
- ❌ **Con**: Longer training time
- 🔄 **Alternative**: Single-phase training (faster but lower accuracy)

**Justification**: Training is one-time cost (or infrequent retraining). The 2-phase approach provides measurably better results and is worth the extra time.

### 5. Class Imbalance: Oversampling vs. Class Weights

**Decision**: Use 3x boosted class weights + data augmentation

**Trade-off**:

- ✅ **Pro**: Simple, no dataset modification needed
- ✅ **Pro**: Fast training (no duplicate images)
- ❌ **Con**: Less effective than SMOTE or heavy oversampling
- 🔄 **Alternative**: SMOTE (synthetic samples) - more complex, slower

**Justification**: With 25:1 imbalance, class weights and data augmentation provide good results without overcomplicating the pipeline.

## Common Issues

### Dataset Not Found

```
⚠️  Dataset not found or incomplete

Would you like to download the dataset from Kaggle now? [Y/n]:
```

**Solution:**

- Type `Y` to automatically download the dataset from Kaggle
- Or download manually with `uv run scripts/download_dataset.py`

### Model Not Found

```
⚠️  Model not found at models/verito.keras

Would you like to load the pre-trained model from Hugging Face? [Y/n]:
```

**Solution:**

- Type `Y` to download the pre-trained model from Hugging Face Hub (recommended)
- Or type `n` and then choose to train the model locally with `uv run scripts/train.py`

### No Images Found

```
❌ Error: No images found in /path/to/dir
```

**Solution:**

- Check directory path is correct
- Ensure images are in supported formats (JPEG, PNG, TIFF, BMP)
- Images should be directly in the directory (not in subdirectories)

### Import Errors

```
ModuleNotFoundError: No module named 'verito'
```

**Solution:** Ensure you're in the project root and dependencies are installed

```sh
cd /path/to/verito
uv sync
```

## Acknowledgements

### Datasets

This project uses the [Car Insurance Fraud Detection dataset](https://www.kaggle.com/datasets/pacificrm/car-insurance-fraud-detection), available under the [CC0 1.0 Universal (Public Domain)](https://creativecommons.org/publicdomain/zero/1.0/) license.

### AI Tools

This project was developed with assistance from AI-powered tools:

- **[Cursor](https://cursor.com/)** - AI-powered IDE used throughout development for:

  - Code completion and intelligent suggestions
  - Code generation and refactoring
  - Documentation writing and improvements
  - Debugging and problem-solving assistance

AI tools were used as collaborative assistants to enhance productivity and code quality while maintaining human oversight for all design decisions and implementation choices.

## Additional Documentation

- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Developer guide (architecture, training, tuning)
- **[STYLE.md](STYLE.md)** - Style guide for messaging and documentation
