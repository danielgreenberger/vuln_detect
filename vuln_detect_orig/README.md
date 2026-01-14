# Mental Health Vulnerability Detection

A transformer-based deep learning system for detecting mental health vulnerability levels in social media posts using fine-tuned RoBERTa.

---

## Quick Start Guide

### Prerequisites
```bash
pip install -r requirements.txt
pip install kaggle  # For dataset download
```

### Step 1: Get the Data (One-Time Setup)

```bash
cd project/vuln_detect

# Download Kaggle dataset (~232K Reddit posts)
kaggle datasets download -d nikhileswarkomati/suicide-watch
unzip suicide-watch.zip -d ./data/

# Process to 3-class system (neutral/moderate/severe)
python data_utils.py \
    --mode kaggle \
    --kaggle-input ./data/Suicide_Detection.csv \
    --output ./data/mental_health_processed.csv
```

**Output:** `./data/mental_health_processed.csv` with ~220K processed posts

### Step 2: Train Your Model

```bash
# Train with 5K samples on CPU (~30-45 min)
python fine_tune_vulnerability_detector.py \
    --data ./data/mental_health_processed.csv \
    --max-samples 5000 \
    --device cpu
```

**Output:** `vulnerability_detector_model/` with trained model

**For M2 Pro Mac:** CPU mode is recommended to avoid memory issues. For full dataset (220K), use Google Colab or Kaggle Notebooks (free GPU).

### Step 3: Test Your Model

```bash
# Evaluate on 15% held-out test set
python evaluate_and_predict.py \
    --eval-data ./data/mental_health_processed.csv \
    --test-split
```

**Output:** Accuracy, F1, precision, recall, confusion matrix, and **safety check** (severe recall ≥ 90%)

### Step 4: Use Your Model

**Single prediction:**
```bash
python evaluate_and_predict.py \
    --text "I can't take this anymore, I want to end it all"
```

**Demo mode:**
```bash
python evaluate_and_predict.py --demo
```

---

## Detailed Documentation

For more information, see:
- [`DATA_SETUP.md`](DATA_SETUP.md) - Complete dataset setup guide
- [`QUICKSTART.md`](QUICKSTART.md) - Additional usage examples
- [`PROJECT_IMPROVEMENTS.md`](PROJECT_IMPROVEMENTS.md) - Complete changelog

---

## Risk Levels

The model classifies posts into three risk categories:
- **0: Neutral** - No depression detected
- **1: Moderate** - Person would benefit from help, no immediate danger  
- **2: Severe** - Person is in danger

## Project Structure

```
vuln_detect/
├── fine_tune_vulnerability_detector.py  # Training script
├── evaluate_and_predict.py             # Evaluation and inference
├── data_utils.py                       # Data loading utilities
├── requirements.txt                    # Python dependencies
├── README.md                          # This file
└── hugging_face_tutorial.py           # Original tutorial reference
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install torch transformers datasets evaluate accelerate scikit-learn pandas numpy
```

### 2. Prepare Your Data

Your data should be in CSV or JSON format with the following columns:
- `text`: The social media post content
- `label`: Risk level (0=Neutral, 1=Moderate, 2=Severe)
- `subreddit` (optional): Source subreddit

Example CSV format:
```csv
text,label,subreddit
"Just watched an amazing movie!",0,movies
"I've been feeling really down lately",1,depression
"I don't think I can go on",2,SuicideWatch
```

## Data Setup

### Quick Start: Sample Data
```bash
python data_utils.py --mode sample --output sample_data.csv
```

### Production: Kaggle Dataset
See [`DATA_SETUP.md`](DATA_SETUP.md) for complete instructions on downloading and processing the Kaggle dataset.

Quick summary:
```bash
# 1. Setup Kaggle CLI and credentials
pip install kaggle

# 2. Download dataset
kaggle datasets download -d nikhileswarkomati/suicide-watch
unzip suicide-watch.zip -d ./data/

# 3. Process to 3-class system
python data_utils.py --mode kaggle --kaggle-input ./data/Suicide_Detection.csv
```

This provides ~220K real Reddit posts for training.

## Usage

### Step 1: Train the Model

**IMPORTANT**: You must train the model first before using evaluation!

#### Option A: Train with your own data
```bash
python3 fine_tune_vulnerability_detector.py --data path/to/your_data.csv
```

#### Option B: Train with sample data (for testing)
```bash
python3 fine_tune_vulnerability_detector.py
```

This will:
- Create sample data automatically
- Fine-tune the twitter-roberta-base model
- Save the trained model to `vulnerability_detector_model/`
- Display training progress and metrics

**Training arguments:**
```bash
python3 fine_tune_vulnerability_detector.py \
    --data data/reddit_posts.csv \
    --model cardiffnlp/twitter-roberta-base \
    --epochs 3 \
    --batch-size 16
```

### Step 2: Evaluate and Make Predictions

After training is complete, you can:

#### Demo Mode (test with sample texts)
```bash
python3 evaluate_and_predict.py --demo
```

#### Single Text Prediction
```bash
python3 evaluate_and_predict.py --text "I've been feeling really down lately"
```

#### Evaluate on Test Data
```bash
python3 evaluate_and_predict.py --eval-data path/to/test_data.csv
```

#### Use Custom Model Path
```bash
python3 evaluate_and_predict.py --model path/to/model --demo
```

## Configuration

Edit the `Config` class in [`fine_tune_vulnerability_detector.py`](fine_tune_vulnerability_detector.py:19) to customize:

```python
class Config:
    # Model configuration
    model_name = "cardiffnlp/twitter-roberta-base"
    num_labels = 3
    
    # Training configuration
    learning_rate = 2e-5
    per_device_train_batch_size = 16
    num_train_epochs = 3
    
    # Data configuration
    max_length = 512
    train_test_split = 0.2
```

## Data Collection

To collect Reddit data for training:

### Option 1: Use the Data Utilities
```python
from data_utils import RedditDataCollector

collector = RedditDataCollector()

# Collect posts from specific subreddits
data = collector.collect_posts(
    subreddits={
        'movies': 0,      # Neutral
        'depression': 1,  # Moderate
        'SuicideWatch': 2 # Severe
    },
    posts_per_subreddit=1000
)

# Save to CSV
data.to_csv('reddit_data.csv', index=False)
```

### Option 2: Manual Data Preparation
Create a CSV file with your labeled data in the format described above.

## Model Performance

The model will output metrics including:
- **Accuracy**: Overall correctness
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: How many predicted positives are correct
- **Recall**: How many actual positives were found
- **Confusion Matrix**: Detailed breakdown by class

Example output:
```
EVALUATION RESULTS
======================================================================
Total Samples: 300

Overall Metrics:
  Accuracy:  0.8567
  F1 Score:  0.8523
  Precision: 0.8601
  Recall:    0.8567

Per-Class Metrics:
  Neutral:
    Precision: 0.9200
    Recall:    0.9020
    F1-Score:  0.9109
    Support:   100

  Moderate:
    Precision: 0.8300
    Recall:    0.8500
    F1-Score:  0.8399
    Support:   100

  Severe:
    Precision: 0.8303
    Recall:    0.8180
    F1-Score:  0.8241
    Support:   100
```

## Programmatic Usage

```python
from evaluate_and_predict import VulnerabilityDetector

# Load the trained model
detector = VulnerabilityDetector("vulnerability_detector_model")

# Single prediction
result = detector.predict_single(
    "I've been feeling really down lately",
    return_probabilities=True
)

print(f"Risk Level: {result['predicted_label']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: {result['probabilities']}")

# Batch prediction
texts = [
    "Great day at the park!",
    "Everything feels hopeless",
    "I don't want to be here anymore"
]
results = detector.predict_batch(texts)

for r in results:
    print(f"{r['text'][:50]}: {r['predicted_label']} ({r['confidence']:.2%})")
```

## Troubleshooting

### Error: "Unrecognized model in vulnerability_detector_model"
**Solution**: You need to train the model first! Run:
```bash
python3 fine_tune_vulnerability_detector.py
```

### CUDA Out of Memory
**Solution**: Reduce batch size:
```bash
python3 fine_tune_vulnerability_detector.py --batch-size 8
```

### Low Accuracy
**Solutions**:
- Collect more training data
- Increase training epochs
- Try a different base model (e.g., `roberta-base`, `bert-base-uncased`)
- Balance your dataset (equal samples per class)

## Model Selection

Available pre-trained models:
- `cardiffnlp/twitter-roberta-base` (Recommended for social media)
- `roberta-base` (General purpose)
- `distilbert-base-uncased` (Faster, smaller)
- `bert-base-uncased` (Classic choice)

## Citation

Based on the project outline for AI for Social Good course, implementing vulnerability detection using fine-tuned transformers for mental health support.

## License

This project is for educational purposes.