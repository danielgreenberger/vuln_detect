# Mental Health Vulnerability Detector

Fine-tunes RoBERTa to detect depression and suicide risk in text. Includes data prep, incremental training, evaluation, and model backup/restore.

## Quick Start

```bash
pip install -r requirements.txt
```

## Download Pre-trained Model

To use the pre-trained model without training from scratch:

```bash
python model_restore.py --repo daniel7643/detector_batch_1
```

This downloads the model to `outputs/detector_batch_1/final_model/`. Then run the app:

```bash
python app.py
```

## 1. Data Preparation

```bash
chmod +x run_data_prep.sh
./run_data_prep.sh
python create_incremental_batches.py
```

Creates `data/incremental/test_set.csv` and `batch_1.csv` through `batch_30.csv`.

## 2. Training

Incremental training - run batches in sequence:

```bash
python train.py --config configs/incremental_batch.yaml --batch-number 1
python train.py --config configs/incremental_batch.yaml --batch-number 2
# ... continue for more batches
```

Models saved to `outputs/batch_N_run/final_model/`.

## 3. Evaluation

Run the interactive web app:

```bash
python app.py
```

Open http://127.0.0.1:8050 - select model, enter text, see predictions with risk scores.

## 4. Command-Line Prediction

```python
from predict import Predictor
p = Predictor("outputs/batch_1_run/final_model")
result = p.predict("I'm feeling really down today")
print(result)  # {'label': 'Depression', 'confidence': 0.85, 'probabilities': {...}}
```

## 5. Model Backup & Restore

Models are excluded from git (too large). Use Hugging Face Hub for storage:

### Backup
```bash
python model_backup.py --login                    # First time only
python model_backup.py --list                     # Show local models
python model_backup.py --model outputs/batch_1_run/final_model --repo USERNAME/model-name
```

### Restore
```bash
python model_restore.py --list-remote             # Show your HF models
python model_restore.py --repo USERNAME/model-name
```

## Project Structure

```
├── train.py              # Training script
├── predict.py            # Prediction API
├── evaluate.py           # Evaluation script
├── app.py                # Dash web UI
├── model_backup.py       # Upload to Hugging Face
├── model_restore.py      # Download from Hugging Face
├── configs/              # Training configs
├── data/incremental/     # Training batches
└── outputs/              # Trained models (gitignored)
```

## Classes

| Label | ID | Description |
|-------|-----|-------------|
| Safe | 0 | No risk indicators |
| Depression | 1 | Depression indicators |
| Suicide | 2 | Suicide risk indicators |