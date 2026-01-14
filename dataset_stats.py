#!/usr/bin/env python3
"""
Dataset Label Statistics Script

Prints the count and percentage of each label in a CSV dataset file.

Usage:
    python dataset_stats.py <csv_file> [csv_file2] ...

Examples:
    python dataset_stats.py data/incremental/batch_1.csv
    python dataset_stats.py data/incremental/test_set.csv
    python dataset_stats.py data/incremental/batch_*.csv
"""

import sys
import pandas as pd
from pathlib import Path


def get_label_stats(csv_path: str) -> None:
    """Print label statistics for a CSV dataset."""
    path = Path(csv_path)
    
    if not path.exists():
        print(f"Error: File '{csv_path}' not found.")
        return
    
    # Load the CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading '{csv_path}': {e}")
        return
    
    if 'label' not in df.columns:
        print(f"Error: No 'label' column found in {csv_path}")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Calculate statistics
    total = len(df)
    label_counts = df['label'].value_counts()
    label_percentages = df['label'].value_counts(normalize=True) * 100
    
    # Print results
    print(f"\n{'='*50}")
    print(f"Dataset: {path.name}")
    print(f"{'='*50}")
    print(f"Total examples: {total:,}\n")
    
    print(f"{'Label':<15} {'Count':>10} {'Percentage':>12}")
    print(f"{'-'*37}")
    
    for label in label_counts.index:
        count = label_counts[label]
        pct = label_percentages[label]
        print(f"{label:<15} {count:>10,} {pct:>11.2f}%")
    
    print(f"{'='*50}\n")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    for csv_file in sys.argv[1:]:
        get_label_stats(csv_file)


if __name__ == "__main__":
    main()