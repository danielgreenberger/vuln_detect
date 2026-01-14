# Dataset Setup Guide

This guide explains how to download and prepare the Kaggle mental health dataset for training.

## Option 1: Use Sample Data (Quick Start)

For testing purposes, use the built-in sample dataset generator:

```bash
cd project/vuln_detect
python data_utils.py --mode sample --output sample_data.csv
```

This creates 300 synthetic examples (100 per class) for testing the pipeline.

## Option 2: Use Kaggle Dataset (Production Training)

### Step 1: Install Kaggle CLI

```bash
pip install kaggle
```

### Step 2: Setup Kaggle Credentials

1. Go to https://www.kaggle.com/settings
2. Click "Create New API Token"
3. Download `kaggle.json`
4. Move it to the correct location:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Step 3: Download Dataset

```bash
# Create data directory
mkdir -p ./data

# Download Suicide Detection dataset (~232K examples)
kaggle datasets download -d nikhileswarkomati/suicide-watch

# Extract
unzip suicide-watch.zip -d ./data/
```

This downloads `Suicide_Detection.csv` with binary labels (suicide/non-suicide).

### Step 4: Process Dataset

Convert binary labels to 3-class system (neutral/moderate/severe):

```bash
python data_utils.py \
    --mode kaggle \
    --kaggle-input ./data/Suicide_Detection.csv \
    --output ./data/mental_health_processed.csv
```

**Output:**
- Processes ~220K examples after cleaning
- Maps labels: suicide → 2 (severe), depression keywords → 1 (moderate), other → 0 (neutral)
- Removes duplicates and short posts
- Applies text cleaning (URLs, formatting, etc.)

### Step 5: Optional - Balance Dataset

If classes are imbalanced, balance them:

```bash
# Undersample to minority class size
python data_utils.py \
    --mode kaggle \
    --kaggle-input ./data/Suicide_Detection.csv \
    --output ./data/mental_health_processed.csv \
    --balance undersample

# Or oversample to majority class size
python data_utils.py \
    --mode kaggle \
    --kaggle-input ./data/Suicide_Detection.csv \
    --output ./data/mental_health_processed.csv \
    --balance oversample
```

## Dataset Details

### Source
- **Kaggle:** https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch
- **Original size:** ~232,000 Reddit posts
- **After processing:** ~220,000 examples

### Label Mapping
Our system uses 3 classes:

- **Class 0 (Neutral):** Posts without mental health indicators
  - Source: r/teenagers, r/AskReddit posts without depression keywords
  
- **Class 1 (Moderate):** Posts indicating mental health concerns
  - Identified by keywords: depression, anxiety, stress, therapy, etc.
  - Suggests person would benefit from help, no immediate danger
  
- **Class 2 (Severe):** Posts indicating suicide risk
  - Source: r/SuicideWatch posts
  - Indicates person may be in danger, requires immediate attention

### Expected Distribution
After processing:
- Class 0: ~120,000 examples (55%)
- Class 1: ~60,000 examples (27%)
- Class 2: ~40,000 examples (18%)

*Note: Exact numbers vary based on keyword matching*

## Training with Dataset

Once processed, train the model:

```bash
python fine_tune_vulnerability_detector.py \
    --data ./data/mental_health_processed.csv \
    --output_dir ./models/mental_health_detector
```

See [`QUICKSTART.md`](QUICKSTART.md) for full training instructions.

## Ethical Considerations

⚠️ **Important:** This dataset contains sensitive mental health content.

- Data is anonymized (no usernames)
- Use only for research/detection purposes
- Include crisis resources in any deployment (e.g., 988 Suicide & Crisis Lifeline)
- Do not share raw posts publicly
- Implement human oversight for high-risk predictions
- Be aware of demographic biases in Reddit data

## Troubleshooting

**"Dataset file not found"**
- Verify file path: `ls ./data/Suicide_Detection.csv`
- Re-download if corrupted

**"Kaggle API credentials not found"**
- Check: `ls ~/.kaggle/kaggle.json`
- Ensure correct permissions: `chmod 600 ~/.kaggle/kaggle.json`

**"Out of memory during processing"**
- Process in chunks or use a machine with more RAM
- Consider downsampling for initial experiments

**"Class imbalance issues"**
- Use `--balance` option
- Or adjust `class_weight` parameter in training

For more help, see [`README.md`](README.md) or open an issue.