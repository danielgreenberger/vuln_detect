#!/usr/bin/env python3
"""
A simplified, direct evaluation script for vulnerability detection models.

This script evaluates a trained model on a test dataset, providing key metrics
such as accuracy, F1-score, precision, and recall.

Usage:
    python evaluate.py --model-path "outputs/batch_1_run/final_model" --data-path "data/incremental/test_set.csv"
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset, DatasetDict
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

    # --- 6. Display Results ---
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 70)
    for key, value in results.items():
        logger.info(f"  {key.replace('eval_', '')}: {value:.4f}")
    logger.info("=" * 70)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()