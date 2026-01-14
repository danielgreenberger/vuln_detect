# Quick Start Guide

## Installation

```bash
# Install required packages
pip install torch transformers datasets evaluate accelerate scikit-learn pandas numpy
```

## Step 1: Train the Model

Run this command to train with sample data:

```bash
python3 fine_tune_vulnerability_detector.py
```

This will:
- Automatically create 15 sample posts (5 per risk level)
- Fine-tune the twitter-roberta-base model
- Save the trained model to `vulnerability_detector_model/`
- Take ~2-5 minutes on CPU, faster on GPU

Expected output:
```
CUDA available: True/False
======================================================================
STEP 1: Loading Data
======================================================================
No data file found. Creating sample data...
Label distribution:
  Neutral (0): 5 samples
  Moderate (1): 5 samples
  Severe (2): 5 samples
...
Training complete!
Model saved to: vulnerability_detector_model
```

## Step 2: Test the Model

After training is complete, test it:

```bash
# Demo mode with sample texts
python3 evaluate_and_predict.py --demo

# Or predict a single text
python3 evaluate_and_predict.py --text "I've been feeling down lately"
```

## Expected Demo Output

```
======================================================================
VULNERABILITY DETECTOR - DEMO
======================================================================
Loading model from: vulnerability_detector_model
Using device: cpu
Model loaded successfully!
Labels: ['Neutral', 'Moderate', 'Severe']

======================================================================
PREDICTIONS
======================================================================

--- Example 1 ---
Text: Just finished a great workout at the gym! Feeling energized.

Predicted Risk Level: Neutral
Confidence: 0.9234

Probabilities:
  Neutral   : 0.9234
  Moderate  : 0.0651
  Severe    : 0.0115

--- Example 2 ---
Text: I've been feeling really down lately, struggling to get out of bed.

Predicted Risk Level: Moderate
Confidence: 0.8567

Probabilities:
  Neutral   : 0.0893
  Moderate  : 0.8567
  Severe    : 0.0540

--- Example 3 ---
Text: I don't think I can keep going. Everything feels hopeless.

Predicted Risk Level: Severe
Confidence: 0.9123

Probabilities:
  Neutral   : 0.0123
  Moderate  : 0.0754
  Severe    : 0.9123
```

## Common Issues

### Issue: "Unrecognized model in vulnerability_detector_model"
**Solution**: You need to train the model first!
```bash
python3 fine_tune_vulnerability_detector.py
```

### Issue: Training is slow
**Solution**: The sample data is very small, so training should be fast. If you have a larger dataset:
- Reduce batch size: `--batch-size 8`
- Reduce epochs: `--epochs 2`

### Issue: Low accuracy
**Solution**: The sample data has only 15 examples. For real use:
- Collect more data (100+ examples per class minimum)
- Use the data_utils.py to create larger datasets

## Next Steps

### Use Your Own Data

1. Create a CSV file with columns: `text`, `label`
2. Train with your data:
```bash
python3 fine_tune_vulnerability_detector.py --data your_data.csv
```

### Create More Sample Data

```bash
python3 data_utils.py --create-sample --samples 300 --output sample_data.csv
python3 fine_tune_vulnerability_detector.py --data sample_data.csv
```

### Programmatic Usage

```python
from evaluate_and_predict import VulnerabilityDetector

detector = VulnerabilityDetector("vulnerability_detector_model")
result = detector.predict_single("Your text here", return_probabilities=True)
print(f"Risk Level: {result['predicted_label']}")