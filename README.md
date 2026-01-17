# Mental Health Vulnerability Detector

Fine-tunes RoBERTa to detect depression and suicide risk in text.

## Setup

```bash
# Install Git LFS
brew install git-lfs              # macOS
sudo apt-get install git-lfs      # Linux
# Windows: download from https://git-lfs.github.com

# Clone repository and pull model files
git clone git@github.com:danielgreenberger/vuln_detect.git
cd vuln_detect
git lfs pull

# Create virtual environment
python3 -m venv venv
source venv/bin/activate          # macOS/Linux
venv\Scripts\activate             # Windows

# Install dependencies
pip install -r requirements.txt
```

## Run the App

```bash
python app.py
```

Open http://127.0.0.1:8050 - select model, enter text, see predictions with risk scores.

## Training (Optional)

### Data Preparation
```bash
chmod +x run_data_prep.sh
./run_data_prep.sh
python create_incremental_batches.py
```

### Incremental Training
```bash
python train.py --config configs/incremental_batch.yaml --batch-number 1
python train.py --config configs/incremental_batch.yaml --batch-number 2
# Continue for more batches...
```

Models saved to `outputs/batch_N_run/final_model/`.

## Classes

| Label | Description |
|-------|-------------|
| Safe | No risk indicators |
| Depression | Depression indicators |
| Suicide | Suicide risk indicators |
