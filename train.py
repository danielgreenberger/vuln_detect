#!/usr/bin/env python3
"""
A streamlined, direct training script for vulnerability detection models.

This script is responsible for the core training loop. All data preparation,
including cleaning, balancing, and splitting, is handled by the `preprocess.py`
script, which can be run automatically via configuration.

Key Features:
- Centralized data preprocessing controlled by `config.run_preprocessing`.
- Simplified data loading that consumes pre-split train, validation, and test sets.
- A clean, focused training pipeline using the Hugging Face Trainer.

Usage:
    # Preprocess data and then train
    python train.py --config configs/train_default.yaml

    # Run incremental training for a specific batch
    python train.py --config configs/incremental_batch.yaml --batch-number 2
"""

import argparse
import logging
import os
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.vuln_detect.config import Config
from src.vuln_detect.data.loaders import load_data
from src.vuln_detect.data.preprocess import process_and_split_data, prepare_dataframe
from src.vuln_detect.metrics import compute_metrics

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train a vulnerability detection model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--batch-number",
        type=int,
        default=None,
        help="For incremental training, specify the batch number (e.g., 1, 2, 3).",
    )
    args = parser.parse_args()

    # --- 1. Load Configuration ---
    config = Config.from_yaml(args.config)

    # --- NEW: Validate arguments for incremental training ---
    if "incremental" in args.config and not args.batch_number:
        raise ValueError(
            "An incremental config was provided, but --batch-number is missing. "
            "Please specify which batch to run, e.g., --batch-number 1"
        )

    # --- 2. Run Preprocessing if configured ---
    if config.run_preprocessing:
        logger.info("Preprocessing step is enabled. Running the data processing script...")
        process_and_split_data(
            input_path=config.raw_data_path,
            output_dir=config.processed_data_dir,
            balance_strategy=config.balance_strategy,
            train_split=config.train_split,
            val_split=config.val_split,
            random_seed=config.random_seed,
        )
        logger.info("Preprocessing finished. Training will use the processed data.")
    else:
        logger.info("Skipping preprocessing step as per configuration.")

    # --- 3. Device Selection ---
    # Forcing CPU for this task to ensure no MPS memory errors.
    device = torch.device("cpu")
    logger.info(f"Forcing CPU-only training. Using device: {device}")

    # --- 4. Data Loading ---
    datasets = DatasetDict()
    if args.batch_number:
        # Incremental training: train on the new batch, evaluate on the standard validation set.
        logger.info(f"Setting up for incremental training batch #{args.batch_number}")
        config.data_path = f"data/incremental/batch_{args.batch_number}.csv"
        config.output_dir = f"outputs/batch_{args.batch_number}_run"

        logger.info(f"Loading and preparing incremental training data from: {config.data_path}")
        train_df = load_data(config.data_path)
        prepared_train_df = prepare_dataframe(train_df)
        datasets["train"] = Dataset.from_pandas(prepared_train_df)

        logger.info(f"Loading and preparing validation data from: {config.val_data_path}")
        val_df = load_data(config.val_data_path)
        prepared_val_df = prepare_dataframe(val_df)

        # --- Validation Set Sampling ---
        if config.val_sample_size and config.val_sample_size < len(prepared_val_df):
            logger.info(f"Sampling validation set down to {config.val_sample_size} samples.")
            # Use stratified sampling to maintain label distribution
            prepared_val_df = prepared_val_df.groupby('label', group_keys=False).apply(
                lambda x: x.sample(
                    n=min(len(x), int(config.val_sample_size * len(x) / len(prepared_val_df))),
                    random_state=config.random_seed
                )
            ).reset_index(drop=True)
            logger.info(f"Sampled validation set size: {len(prepared_val_df)}")

        datasets["validation"] = Dataset.from_pandas(prepared_val_df)

        if args.batch_number > 1:
            previous_batch_dir = f"outputs/batch_{args.batch_number - 1}_run"
            checkpoint_path = f"{previous_batch_dir}/final_model"
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found for previous batch: {checkpoint_path}")
            config.model_name = checkpoint_path
            config.learning_rate /= 4
            config.num_epochs = 2
            logger.info(f"Loading model from: {config.model_name}")
            logger.info(f"Reduced learning rate to: {config.learning_rate}")
    else:
        # Standard training: load pre-split datasets.
        logger.info(f"Loading pre-split data from: {config.processed_data_dir}")
        datasets = DatasetDict.from_csv({
            "train": config.train_data_path,
            "validation": config.val_data_path,
            "test": config.test_data_path,
        })

    logger.info(f"Datasets loaded: {datasets}")

    # --- 5. Model and Tokenizer Initialization ---
    logger.info(f"Loading model and tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name, num_labels=config.num_labels
    )
    model.to(device)

    # --- 6. Tokenization ---
    def tokenize_function(examples):
        return tokenizer(
            examples[config.text_column],
            truncation=True,
            max_length=config.max_length,
            padding=False,
        )

    tokenized_datasets = datasets.map(
        tokenize_function, batched=True, remove_columns=[config.text_column]
    )
    tokenized_datasets = tokenized_datasets.rename_column(config.label_column, "labels")
    tokenized_datasets.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # --- 7. Training Arguments and Trainer Setup ---
    training_args_dict = {
        "output_dir": config.output_dir,
        "learning_rate": config.learning_rate,
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": config.batch_size,
        "num_train_epochs": config.num_epochs,
        "weight_decay": config.weight_decay,
        "logging_steps": config.logging_steps,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "eval_strategy": config.eval_strategy,
        "save_strategy": config.save_strategy,
        "load_best_model_at_end": config.load_best_model_at_end,
        "metric_for_best_model": config.metric_for_best_model,
        "greater_is_better": False if "loss" in config.metric_for_best_model else True,
        "fp16": config.fp16 and device.type == "cuda",
        "report_to": config.report_to,
        "save_total_limit": config.save_total_limit,
        "dataloader_pin_memory": False,
        "no_cuda": True,
        "use_cpu": True,
    }
    if config.use_gradient_checkpointing:
        training_args_dict["gradient_checkpointing"] = True

    logger.info("Hardcoding `no_cuda=True` and `use_cpu=True` in TrainingArguments to guarantee CPU usage.")
    training_args = TrainingArguments(**training_args_dict)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # --- 8. Training ---
    logger.info("Starting training...")
    trainer.train()

    # --- 9. Evaluation ---
    if "test" in tokenized_datasets:
        logger.info("Evaluating on the test set...")
        test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])

        logger.info("\n" + "=" * 70)
        logger.info("TEST SET EVALUATION RESULTS")
        logger.info("=" * 70)
        for key, value in test_results.items():
            logger.info(f"  {key}: {value:.4f}")
        logger.info("=" * 70)
    else:
        logger.info("No test set found, skipping final evaluation.")

    # --- 10. Save Final Model ---
    final_model_path = Path(config.output_dir) / "final_model"
    logger.info(f"Saving final model to: {final_model_path}")
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))

    logger.info("Training complete!")


if __name__ == "__main__":
    main()