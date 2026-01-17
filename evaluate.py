#!/usr/bin/env python3
"""
A simplified, direct evaluation script for vulnerability detection models.

This script evaluates a trained model on a test dataset, providing key metrics
such as accuracy, F1-score, precision, and recall, along with detailed per-class
metrics and confusion matrix.

Usage:
    python evaluate.py --model-path "outputs/batch_1_run/final_model" --data-path "data/incremental/test_set.csv"
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.vuln_detect.data.loaders import load_data
from src.vuln_detect.data.preprocess import prepare_dataframe
from src.vuln_detect.metrics import compute_metrics

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate a vulnerability detection model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model directory.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the evaluation data file.",
    )
    parser.add_argument(
        "--text-column", type=str, default="text", help="Name of the text column."
    )
    parser.add_argument(
        "--label-column", type=str, default="label", help="Name of the label column."
    )
    parser.add_argument(
        "--max-length", type=int, default=512, help="Max sequence length for tokenization."
    )
    args = parser.parse_args()

    # --- 1. Device Selection ---
    device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # --- 2. Load Model and Tokenizer ---
    logger.info(f"Loading model from: {args.model_path}")
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)

    # --- 3. Load and Prepare Data ---
    logger.info(f"Loading data from: {args.data_path}")
    df = load_data(args.data_path, text_column=args.text_column, label_column=args.label_column)
    
    # Use prepare_dataframe to clean text and convert labels to integers
    df = prepare_dataframe(df)
    
    eval_dataset = Dataset.from_pandas(df)

    # --- 4. Tokenization ---
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )

    tokenized_dataset = eval_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # --- 5. Evaluation ---
    # We use the Trainer here as a convenient way to run evaluation.
    # No training is performed.
    training_args = TrainingArguments(
        output_dir="./tmp_eval",  # Temporary directory, not used for saving
        per_device_eval_batch_size=16,
        do_train=False,
        do_eval=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Running evaluation...")
    results = trainer.evaluate()

    # --- 6. Get Predictions for Detailed Analysis ---
    logger.info("Generating predictions for detailed metrics...")
    predictions_output = trainer.predict(tokenized_dataset)
    predictions = np.argmax(predictions_output.predictions, axis=1)
    true_labels = predictions_output.label_ids

    # Define label names (matching your dataset)
    label_names = ["Neutral", "Suicide", "Depression"]

    # --- 7. Display Summary Results ---
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION RESULTS (Weighted Average)")
    logger.info("=" * 70)
    for key, value in results.items():
        if isinstance(value, float):
            logger.info(f"  {key.replace('eval_', '')}: {value:.4f}")
        else:
            logger.info(f"  {key.replace('eval_', '')}: {value}")
    logger.info("=" * 70)

    # --- 8. Display Per-Class Metrics ---
    logger.info("\n" + "=" * 70)
    logger.info("PER-CLASS METRICS (Classification Report)")
    logger.info("=" * 70)
    report = classification_report(
        true_labels,
        predictions,
        target_names=label_names,
        digits=4
    )
    # Print each line of the report
    for line in report.split('\n'):
        logger.info(line)
    logger.info("=" * 70)

    # --- 9. Display Confusion Matrix ---
    logger.info("\n" + "=" * 70)
    logger.info("CONFUSION MATRIX")
    logger.info("=" * 70)
    cm = confusion_matrix(true_labels, predictions)
    
    # Create a formatted confusion matrix display
    logger.info(f"{'':>12} " + " ".join(f"{name:>10}" for name in label_names) + "  (Predicted)")
    logger.info("-" * (14 + 11 * len(label_names)))
    for i, row in enumerate(cm):
        row_str = " ".join(f"{val:>10}" for val in row)
        logger.info(f"{label_names[i]:>12} {row_str}")
    logger.info("(Actual)")
    logger.info("=" * 70)

    # --- 10. Display Key Insights ---
    logger.info("\n" + "=" * 70)
    logger.info("KEY INSIGHTS")
    logger.info("=" * 70)
    
    # Calculate per-class accuracy
    for i, name in enumerate(label_names):
        class_total = cm[i].sum()
        class_correct = cm[i][i]
        class_accuracy = class_correct / class_total if class_total > 0 else 0
        class_missed = class_total - class_correct
        logger.info(f"  {name}: {class_correct}/{class_total} correct ({class_accuracy:.2%}), {class_missed} misclassified")
    
    # Calculate overall accuracy
    total_correct = np.trace(cm)
    total_samples = cm.sum()
    logger.info(f"\n  Overall Accuracy: {total_correct}/{total_samples} ({total_correct/total_samples:.2%})")
    logger.info("=" * 70)

    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()