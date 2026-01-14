"""
Data Preprocessing and Splitting Utilities for Vulnerability Detection.
"""

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean a single text string while preserving emotional content.

    Handles:
    - URLs
    - Reddit-specific markdown
    - Multiple newlines
    - Special Reddit formatting (Edit:, Update:, etc.)
    - Excessive punctuation (preserves emotion but reduces noise)
    """
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove Reddit markdown links [text](url) - keep just the text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Remove special Reddit formatting prefixes
    text = re.sub(r'\b(Edit|Update|EDIT|UPDATE):\s*', '', text)
    # Replace multiple newlines with single space
    text = re.sub(r'\n\n+', ' ', text)
    # Replace excessive punctuation (keep emotion but reduce noise)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    # Remove extra whitespace (including single newlines)
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    return text.strip()


def prepare_dataframe(df: pd.DataFrame, min_word_count: int = 5) -> pd.DataFrame:
    """
    Prepares a raw DataFrame by transforming the 'class' column to a numeric
    'label' and cleaning the 'text' column. This is the unified function
    for all raw data.
    """
    logger.info(f"Preparing DataFrame with {len(df)} rows...")

    # --- Transformation Step ---
    def map_to_numeric_label(label_val):
        """Maps a string label or class to a numeric label."""
        label_str = str(label_val).lower()
        if 'suicide' in label_str:
            return 2  # Severe
        elif 'depression' in label_str:
            return 1  # Moderate
        else:
            return 0  # Neutral

    if 'class' in df.columns:
        logger.info("Transforming 'class' column to numeric 'label'.")
        df['label'] = df['class'].apply(map_to_numeric_label)
    elif 'label' in df.columns and pd.api.types.is_string_dtype(df['label']):
        logger.info("Found string 'label' column. Converting to numeric labels.")
        df['label'] = df['label'].apply(map_to_numeric_label)
    elif 'label' not in df.columns:
        raise ValueError("DataFrame must contain either a 'label' or a 'class' column.")

    # --- Cleaning Step ---
    df['text'] = df['text'].apply(clean_text)
    df = df.dropna(subset=['text', 'label']).reset_index(drop=True)
    df['word_count'] = df['text'].str.split().str.len()
    df = df[df['word_count'] >= min_word_count]
    df = df[df['text'].str.len() > 0].reset_index(drop=True)
    
    logger.info(f"Finished preparation. DataFrame now has {len(df)} rows.")
    return df[['text', 'label']].copy()


def balance_dataset(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Balances the dataset using the specified strategy."""
    logger.info(f"Balancing dataset using {strategy} strategy...")
    class_counts = df['label'].value_counts()

    if strategy == 'undersample':
        target_size = class_counts.min()
        sampler = lambda group: group.sample(n=target_size, random_state=42)
    else:
        target_size = class_counts.max()
        sampler = lambda group: group.sample(n=target_size, replace=True, random_state=42)

    balanced_df = df.groupby('label', group_keys=False).apply(sampler)
    return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)


def process_and_split_data(
    input_path: str,
    output_dir: str,
    balance_strategy: str = None,
    train_split: float = 0.7,
    val_split: float = 0.15,
    random_seed: int = 42,
):
    """
    Full pipeline for the initial raw dataset: loads, prepares (transforms and
    cleans), balances, and splits the data.
    """
    logger.info(f"Starting full data processing for: {input_path}")
    df = pd.read_csv(Path(input_path))
    
    # Use the unified preparation function
    df = prepare_dataframe(df)

    if balance_strategy:
        df = balance_dataset(df, balance_strategy)

    test_split = 1.0 - train_split - val_split
    train_df = df.sample(frac=train_split, random_state=random_seed)
    remaining_df = df.drop(train_df.index)
    val_test_frac = val_split / (val_split + test_split)
    val_df = remaining_df.sample(frac=val_test_frac, random_state=random_seed)
    test_df = remaining_df.drop(val_df.index)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_path / "train.csv", index=False)
    val_df.to_csv(output_path / "validation.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)
    logger.info(f"Preprocessing and splitting complete. Data saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and split raw data.")
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--balance", type=str, choices=["undersample", "oversample"])
    parser.add_argument("--train-split", type=float, default=0.7)
    parser.add_argument("--val-split", type=float, default=0.15)
    args = parser.parse_args()
    process_and_split_data(**vars(args))