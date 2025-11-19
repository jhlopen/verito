# Style Guide

This guide defines conventions for user-facing messaging, output formatting, and documentation. These guidelines ensure consistency for both human users and AI agents interacting with the codebase.

## 🎯 Guiding Principles

1. **Clarity**: Messages should be immediately understandable
2. **Consistency**: Similar actions produce similar output
3. **Actionability**: Errors include clear next steps
4. **Professionalism**: Appropriate for production use
5. **Accessibility**: Works for humans and AI agents

## 💬 User-Facing Messaging

### Emoji Usage

Emojis provide visual cues and make output more scannable. Use consistently:

| Emoji | Meaning            | Usage                       |
| ----- | ------------------ | --------------------------- |
| 🚀    | Starting/Launch    | "🚀 Starting training..."   |
| ✅    | Success            | "✅ Training complete!"     |
| ❌    | Error/Failure      | "❌ Model not found"        |
| ⚠️    | Warning            | "⚠️ Model not found at..."  |
| 📊    | Data/Statistics    | "📊 Dataset Summary:"       |
| 📈    | Results/Metrics    | "📈 Detailed Results:"      |
| 🔍    | Detection/Search   | "🔍 Forged Image Detection" |
| 🏗️    | Building           | "🏗️ Building model..."      |
| 🤖    | Model/AI           | "🤖 Loading model..."       |
| 📁    | Files/Directories  | "📁 Configuration:"         |
| 🔢    | Numbers/Matrix     | "🔢 Confusion Matrix:"      |
| 💡    | Tip/Recommendation | "💡 Recommendation:"        |
| 🎯    | Goal/Target        | "🎯 Phase 1: Training..."   |
| 🎉    | Celebration        | "🎉 Ready for detection!"   |

**Guidelines:**

- Use ONE emoji per message type
- Place at the start of the line
- Don't overuse - only for major sections/events
- Be consistent across scripts

### Message Formatting

#### Progress Messages

```python
# Good: Clear, informative
print("\n🚀 Starting Insurance Fraud Detection Model Training\n")
print("🏗️  Building EfficientNetV2L model...")
print("   Model loaded successfully!\n")

# Bad: Too verbose or unclear
print("Now we will begin the training process for the model")
print("Building...")
```

**Pattern:**

- Start with emoji + action verb
- Indent sub-items with 3 spaces
- Blank lines before/after major sections

#### Success Messages

```python
# Good: Clear outcome + next steps
print("\n🎉 Ready for detection!")
print(f"   Run: uv run scripts/detect.py --forged-dir datasets/test/Fraud --authentic-dir datasets/test/Non-Fraud\n")

# Bad: Vague
print("Done!")
```

**Pattern:**

- State what succeeded
- Provide next action (if applicable)
- Include example command

#### Error Messages

```python
# Good: Clear error + solution
print(f"❌ Error: Forged directory not found: {args.forged_dir}")
print("   Solution: Check the path and try again")
sys.exit(1)

# Bad: Unclear or no solution
print("Error: Invalid input")
sys.exit(1)
```

**Pattern:**

- State the problem clearly
- Explain why it's a problem (if not obvious)
- Provide solution or next steps
- Use appropriate exit code

#### Warning Messages

```python
# Good: Clear warning + options
print(f"⚠️  Model not found at {model_path}\n")
response = input("Would you like to train the model now? [Y/n]: ").strip().lower()

# Bad: Unclear or no action
print("Warning: Model missing")
```

**Pattern:**

- State what's missing/wrong
- Offer action or choice
- Default to safe option

### Prompts and User Input

```python
# Good: Clear default, simple input
response = input("Would you like to train the model now? [Y/n]: ").strip().lower()
if response in ['', 'y', 'yes']:
    # proceed

# Good: Clear options
response = input("Choose threshold (0.3=sensitive, 0.5=balanced, 0.7=conservative): ")
```

**Guidelines:**

- Show default in brackets: `[Y/n]` means Y is default
- Handle multiple valid inputs: `['', 'y', 'yes']`
- Strip and lowercase input
- Be forgiving with input variants

---

## 📊 Output Formatting

### Report Headers

Use separators for major sections:

```python
print("\n" + "="*70)
print("               🔍 Forged Image Detection Results")
print("="*70 + "\n")
```

**Guidelines:**

- Width: 70 characters (fits most terminals)
- Title centered or left-aligned with indent
- Blank lines before and after

### Metrics Tables

```python
# Performance metrics
print("Performance Metrics:")
print(f"   Accuracy:              {metrics['accuracy']:.2%}")
print(f"   Precision:             {metrics['precision']:.2%}")
print(f"   Recall (Fraud):        {metrics['recall']:.2%}")
print(f"   F1 Score:              {metrics['f1_score']:.2%}\n")
```

**Guidelines:**

- Left-align labels, right-align values
- Use 3-space indent
- Consistent spacing (can use f-string formatting)
- Percentages with 2 decimal places: `:.2%`
- Floats with 2-3 decimal places: `:.2f` or `:.3f`

### Lists and Summaries

```python
# Good: Clear structure
print("📁 Configuration:")
print(f"   Forged images: {args.forged_dir}")
print(f"   Authentic images: {args.authentic_dir}")
print(f"   Model: {args.model_path}")
print(f"   Threshold: {args.threshold}\n")

# Detailed results with context
print("📈 Detailed Results:")
print(f"   ✅ Forged images detected: {true_positive}/{total_forged} ({percent:.1f}%)")
print(f"   ❌ Forged images missed: {false_negative}/{total_forged} ({percent:.1f}%)")
```

**Guidelines:**

- Use emojis for visual separation (✅, ❌, ⚠️)
- Show absolute numbers AND percentages: `64/93 (68.8%)`
- Indent consistently (3 spaces)
- Add context: "Forged images detected" not just "Detected"

### Confusion Matrix

```python
print("🔢 Confusion Matrix:")
print("                 Predicted")
print("              Fraud  Non-Fraud")
print(f"    Fraud      {cm[0,0]:4d}     {cm[0,1]:4d}")
print(f"    Non-Fraud  {cm[1,0]:4d}     {cm[1,1]:4d}\n")
```

**Guidelines:**

- Right-align numbers with fixed width: `:4d`
- Label axes clearly
- Keep spacing consistent

---

## 📝 Documentation Style

### Markdown Headers

```markdown
# Main Title (H1) - Only one per document

## Major Section (H2)

### Subsection (H3)

#### Detail (H4) - Use sparingly
```

**Guidelines:**

- H1: Document title only
- H2: Major sections
- H3: Subsections
- H4+: Avoid if possible (restructure instead)

### Code Blocks

````markdown
```sh
# Shell commands with comments
uv run scripts/train.py
```

```python
# Python code with meaningful variable names
model = load_detector("models/verito.keras")
predictions, _ = load_and_predict_images(model, image_dir)
```
````

### Docstrings

```python
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate classification metrics for binary predictions.

    Args:
        y_true: Ground truth labels (0 = fraud, 1 = authentic)
        y_pred: Predicted labels (0 = fraud, 1 = authentic)

    Returns:
        dict: Metrics including accuracy, precision, recall, F1, confusion matrix

    Example:
        >>> metrics = calculate_metrics(y_true, y_pred)
        >>> print(f"Accuracy: {metrics['accuracy']:.2%}")
    """
```

**Guidelines:**

- One-line summary
- Blank line
- Args section with type hints in signature
- Returns section describing structure
- Example if helpful

## 📖 Summary

**Key Takeaways:**

1. **Be consistent** - Similar actions, similar output
2. **Be clear** - No ambiguous messages
3. **Be helpful** - Errors include solutions
4. **Be professional** - Suitable for production
5. **Think about AI** - Structured, parseable output

**When in doubt:**

- Look at existing code for patterns
- Prioritize clarity over cleverness
- Test with both human and AI users

## 🤝 Feedback

If you notice inconsistencies or have suggestions for this style guide, please open an issue or submit a pull request.
