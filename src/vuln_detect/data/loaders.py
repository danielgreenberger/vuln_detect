"""Data loading functions for vulnerability detection."""

import json
from pathlib import Path

import pandas as pd


def load_data(
    file_path: str,
    text_column: str = "text",
    label_column: str = "label"
) -> pd.DataFrame:
    """Load data from CSV or JSON file.
    
    Automatically detects file format based on extension and loads data
    into a standardized DataFrame format.
    
    Args:
        file_path: Path to the data file (.csv or .json).
        text_column: Name of the column/field containing text data.
        label_column: Name of the column/field containing labels.
        
    Returns:
        DataFrame with 'text' and 'label' columns.
        
    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file format is not supported or required columns missing.
        
    Example:
        >>> df = load_data("data/mental_health.csv")
        >>> print(df.columns)
        Index(['text', 'label'], dtype='object')
    """
    file_path_obj = Path(file_path)
    
    if not file_path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    suffix = file_path_obj.suffix.lower()
    
    if suffix == '.csv':
        df = pd.read_csv(file_path_obj)
    elif suffix == '.json':
        with open(file_path_obj, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            raise ValueError(
                f"Unsupported JSON format. Expected list or dict, got {type(data)}"
            )
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            f"Supported formats: .csv, .json"
        )
    
    # Verify required columns exist
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in file")
    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' not found in file")
    
    # Standardize column names
    df = df.rename(columns={
        text_column: 'text',
        label_column: 'label'
    })
    
    # Keep only required columns
    df = df[['text', 'label']].copy()
    
    return df