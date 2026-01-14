#!/usr/bin/env python3
"""
A simple, argument-driven script to extract, filter, and standardize data
from a source CSV and append it to a target CSV.

This script is designed to be called multiple times to build a consolidated
dataset from various sources.

Example Usage:
    python prepare_data_from_source.py \
        --input-csv "data/source/kaggle_1.csv" \
        --text-columns "clean_text" \
        --filter-column "is_depression" \
        --filter-value 1 \
        --output-csv "data/processed/final_dataset.csv" \
        --output-label "Depression"
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def prepare_and_append(
    input_csv: str,
    text_columns: list,
    filter_column: str,
    filter_value,
    output_csv: str,
    output_label: str,
):
    """
    Processes a single source CSV and appends the standardized data to an output file.
    """
    input_path = Path(input_csv)
    output_path = Path(output_csv)

    # --- 1. Load Input Data ---
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}. Aborting.")
        return

    try:
        df = pd.read_csv(input_path)
        logger.info(f"Successfully loaded {len(df)} rows from {input_path}")
    except Exception as e:
        logger.error(f"Failed to read CSV {input_path}: {e}", exc_info=True)
        return

    # --- 2. Filter Data ---
    if filter_column and filter_value is not None:
        if filter_column not in df.columns:
            logger.error(f"Filter column '{filter_column}' not found in {input_path}. Aborting.")
            return
        
        # Attempt to convert filter_value to the same type as the column for safe comparison
        try:
            col_type = df[filter_column].dtype
            if pd.api.types.is_numeric_dtype(col_type):
                filter_value = type(df[filter_column].iloc[0])(filter_value)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert filter_value '{filter_value}' to match column type. Proceeding with original type.")

        initial_rows = len(df)
        df = df[df[filter_column] == filter_value].copy()
        logger.info(f"Filtered {initial_rows} rows down to {len(df)} where '{filter_column}' == '{filter_value}'")

    if df.empty:
        logger.warning("No data remaining after filtering. Nothing to append.")
        return

    # --- 3. Standardize Data ---
    # Combine multiple text columns if specified
    for col in text_columns:
        if col not in df.columns:
            logger.error(f"Text column '{col}' not found in {input_path}. Aborting.")
            return
    
    df["text"] = df[text_columns].astype(str).agg(" ".join, axis=1)
    
    # Create the final standardized DataFrame
    standardized_df = pd.DataFrame({
        "text": df["text"],
        "label": output_label,
    })

    # --- 4. Append to Output ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # If the file doesn't exist, write with a header. Otherwise, append without a header.
    write_header = not output_path.exists()
    
    standardized_df.to_csv(
        output_path, mode="a", header=write_header, index=False
    )
    logger.info(f"Appended {len(standardized_df)} rows with label '{output_label}' to {output_path}")


def main():
    """Command-line interface for the data preparation script."""
    parser = argparse.ArgumentParser(
        description="Prepare and append data from a source CSV to a target CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-csv", type=str, required=True, help="Path to the source CSV file.")
    parser.add_argument("--text-columns", nargs="+", required=True, help="One or more columns to be used as the input text.")
    parser.add_argument("--filter-column", type=str, help="The column to filter rows by.")
    parser.add_argument("--filter-value", type=str, help="The value to keep in the filter_column.")
    parser.add_argument("--output-csv", type=str, required=True, help="Path to the output CSV file to append results to.")
    parser.add_argument("--output-label", type=str, required=True, help="The standard label to assign to the processed data.")
    
    args = parser.parse_args()

    prepare_and_append(
        input_csv=args.input_csv,
        text_columns=args.text_columns,
        filter_column=args.filter_column,
        filter_value=args.filter_value,
        output_csv=args.output_csv,
        output_label=args.output_label,
    )


if __name__ == "__main__":
    main()