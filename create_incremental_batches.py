#!/usr/bin/env python3
"""
A script to split a consolidated dataset into a test set and a series of
sequential training batches for incremental learning.

This script takes a single input file (e.g., 'final_dataset.csv'), shuffles it,
carves out a test set, and then divides the remaining data into a specified
number of smaller batch files.

Usage:
    python create_incremental_batches.py \
        --input-csv "data/processed/final_dataset.csv" \
        --output-dir "data/incremental" \
        --test-split-size 0.2 \
        --num-batches 10
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_batches(
    input_csv: str,
    output_dir: str,
    test_split_size: float,
    num_batches: int,
    random_seed: int = 42,
):
    """
    Splits the input CSV into a test set and a number of training batches.
    """
    input_path = Path(input_csv)
    output_path = Path(output_dir)

    # --- 1. Load and Validate Input ---
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}. Aborting.")
        return

    try:
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} rows from {input_path}")
    except Exception as e:
        logger.error(f"Failed to read CSV {input_path}: {e}", exc_info=True)
        return

    if df.empty:
        logger.error("Input DataFrame is empty. Cannot create batches.")
        return

    # --- 2. Shuffle and Split Test Set ---
    # Shuffle the dataset to ensure random distribution
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Carve out the test set
    test_size = int(len(df) * test_split_size)
    test_df = df.iloc[:test_size]
    train_df = df.iloc[test_size:]

    logger.info(f"Separated {len(test_df)} rows for the test set.")
    logger.info(f"{len(train_df)} rows remaining for training batches.")

    # --- 3. Create and Save Batches ---
    # Clean the output directory
    output_path.mkdir(parents=True, exist_ok=True)
    for f in output_path.glob("*.csv"):
        f.unlink()
        logger.info(f"Removed old file: {f}")

    # Save the test set
    test_set_path = output_path / "test_set.csv"
    test_df.to_csv(test_set_path, index=False)
    logger.info(f"Saved test set to: {test_set_path}")

    # Split the remaining data into N batches
    if train_df.empty or num_batches == 0:
        logger.warning("No training data available or num_batches is 0. Skipping batch creation.")
        return
        
    # Use np.array_split to handle cases where the data isn't perfectly divisible
    batch_dfs = np.array_split(train_df, num_batches)

    for i, batch_df in enumerate(batch_dfs):
        batch_num = i + 1
        batch_path = output_path / f"batch_{batch_num}.csv"
        batch_df.to_csv(batch_path, index=False)
        logger.info(f"Saved batch #{batch_num} ({len(batch_df)} rows) to: {batch_path}")

    logger.info("\nBatch creation complete.")


def main():
    """Command-line interface for the batch creation script."""
    parser = argparse.ArgumentParser(
        description="Split a dataset into a test set and sequential training batches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="data/processed/final_dataset.csv",
        help="Path to the consolidated input CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/incremental",
        help="Directory to save the test set and batch files.",
    )
    parser.add_argument(
        "--test-split-size",
        type=float,
        default=0.2,
        help="Fraction of the dataset to reserve for the final test set (e.g., 0.2 for 20%).",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=10,
        help="The number of sequential batches to create from the remaining data.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for shuffling to ensure reproducibility.",
    )
    args = parser.parse_args()

    create_batches(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        test_split_size=args.test_split_size,
        num_batches=args.num_batches,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()