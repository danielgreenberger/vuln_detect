"""
Data Utilities for Vulnerability Detection
Tools for collecting and preparing Reddit data for training
"""

import pandas as pd
import json
import re
from typing import Dict, List, Optional


# Label mapping for vulnerability levels
LABEL_MAPPING = {
    0: "Neutral",
    1: "Moderate",
    2: "Severe"
}


def load_from_json(json_path: str) -> pd.DataFrame:
    """
    Load Reddit data from JSON file
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        DataFrame with posts
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return pd.DataFrame(data)


def load_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Load Reddit data from CSV file
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with posts
    """
    return pd.read_csv(csv_path)


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate that data has required columns and proper format
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    # Check required columns
    required_cols = ['text', 'label']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for null values
    if df['text'].isnull().any():
        raise ValueError("Found null values in 'text' column")
    
    if df['label'].isnull().any():
        raise ValueError("Found null values in 'label' column")
    
    # Check label values
    valid_labels = {0, 1, 2}
    invalid_labels = set(df['label'].unique()) - valid_labels
    
    if invalid_labels:
        raise ValueError(f"Found invalid labels: {invalid_labels}. Must be 0, 1, or 2")
    
    print("✓ Data validation passed")
    return True


def print_statistics(df: pd.DataFrame):
    """Print statistics about the dataset"""
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    
    print(f"\nTotal samples: {len(df)}")
    
    print(f"\nLabel distribution:")
    for label in sorted(df['label'].unique()):
        count = (df['label'] == label).sum()
        pct = count / len(df) * 100
        label_name = LABEL_MAPPING.get(label, "Unknown")
        print(f"  {label_name} ({label}): {count} samples ({pct:.1f}%)")
    
    if 'subreddit' in df.columns:
        print(f"\nSubreddit distribution:")
        for subreddit, count in df['subreddit'].value_counts().items():
            pct = count / len(df) * 100
            print(f"  r/{subreddit}: {count} samples ({pct:.1f}%)")
    
    print(f"\nText length statistics:")
    text_lengths = df['text'].str.len()
    print(f"  Mean: {text_lengths.mean():.0f} characters")
    print(f"  Median: {text_lengths.median():.0f} characters")
    print(f"  Min: {text_lengths.min()} characters")
    print(f"  Max: {text_lengths.max()} characters")


def balance_dataset(df: pd.DataFrame, method: str = 'undersample') -> pd.DataFrame:
    """
    Balance the dataset to have equal samples per class
    
    Args:
        df: Input DataFrame
        method: 'undersample' (reduce to min class) or 'oversample' (increase to max class)
        
    Returns:
        Balanced DataFrame
    """
    print(f"\nBalancing dataset using {method}...")
    
    if method == 'undersample':
        # Find the minority class size
        min_size = df['label'].value_counts().min()
        
        # Sample that many from each class
        balanced_df = pd.concat([
            df[df['label'] == label].sample(n=min_size, random_state=42)
            for label in df['label'].unique()
        ])
    
    elif method == 'oversample':
        # Find the majority class size
        max_size = df['label'].value_counts().max()
        
        # Sample with replacement to reach that size
        balanced_df = pd.concat([
            df[df['label'] == label].sample(n=max_size, replace=True, random_state=42)
            for label in df['label'].unique()
        ])
    
    else:
        raise ValueError("method must be 'undersample' or 'oversample'")
    
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Original size: {len(df)}")
    print(f"Balanced size: {len(balanced_df)}")
    
    return balanced_df


def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean text data while preserving emotional content
    
    Handles:
    - URLs
    - Reddit-specific markdown
    - Multiple newlines
    - Special Reddit formatting (Edit:, Update:, etc.)
    - Excessive punctuation (preserves emotion but reduces noise)
    - Keeps emojis (important emotional signals)
    
    Args:
        df: DataFrame with 'text' column
        
    Returns:
        DataFrame with cleaned text
    """
    def clean_single_text(text):
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove Reddit markdown links [text](url) - keep just the text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # Remove special Reddit formatting prefixes
        text = re.sub(r'\b(Edit|Update|EDIT|UPDATE):\s*', '', text)
        
        # Replace multiple newlines with single space
        text = re.sub(r'\n\n+', ' ', text)
        
        # Replace excessive punctuation (keep emotion but reduce noise)
        text = re.sub(r'!{2,}', '!', text)  # Multiple ! to single !
        text = re.sub(r'\?{2,}', '?', text)  # Multiple ? to single ?
        
        # Remove extra whitespace (including single newlines)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    df['text'] = df['text'].apply(clean_single_text)
    
    # Remove empty texts
    df = df[df['text'].str.len() > 0].reset_index(drop=True)
    
    print(f"Cleaned {len(df)} texts")
    
    return df


def clean_single_text(text: str) -> str:
    """
    Clean a single text string while preserving emotional content
    
    Args:
        text: Text string to clean
        
    Returns:
        Cleaned text string
    """
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove Reddit markdown links [text](url) - keep just the text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Remove special Reddit formatting prefixes
    text = re.sub(r'\b(Edit|Update|EDIT|UPDATE):\s*', '', text)
    
    # Replace multiple newlines with single space
    text = re.sub(r'\n\n+', ' ', text)
    
    # Replace excessive punctuation (keep emotion but reduce noise)
    text = re.sub(r'!{2,}', '!', text)  # Multiple ! to single !
    text = re.sub(r'\?{2,}', '?', text)  # Multiple ? to single ?
    
    # Remove extra whitespace (including single newlines)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def create_sample_dataset(output_path: str = "sample_reddit_data.csv", num_samples: int = 300):
    """
    Create a sample dataset for testing
    
    Args:
        output_path: Where to save the CSV
        num_samples: Total number of samples (will be split equally across classes)
    """
    samples_per_class = num_samples // 3
    
    sample_data = {
        'text': [
            # Neutral examples (0)
            *["Just watched an amazing movie last night! Highly recommend it.",
              "Does anyone know a good recipe for chocolate cake?",
              "The weather is beautiful today, perfect for a walk in the park.",
              "I love this new album, the music is incredible!",
              "Planning my vacation for next summer, any recommendations?",
              "Finished reading a great book today.",
              "My favorite team won the game!",
              "Had a wonderful dinner with friends.",
              "Excited for the weekend plans!",
              "Learning a new skill, it's challenging but fun."] * (samples_per_class // 10 + 1),
            
            # Moderate examples (1)
            *["I've been feeling really down lately, nothing seems to help.",
              "Every day feels harder than the last. I don't know what to do.",
              "I feel so alone, even when I'm surrounded by people.",
              "I can't remember the last time I felt happy about anything.",
              "Everything just feels empty and meaningless lately.",
              "I'm struggling to find motivation for anything.",
              "Sleep has become my escape from reality.",
              "I feel disconnected from everyone around me.",
              "Nothing brings me joy anymore.",
              "I'm exhausted all the time, mentally and physically."] * (samples_per_class // 10 + 1),
            
            # Severe examples (2)
            *["I don't think I can go on like this anymore.",
              "Nobody would even notice if I disappeared.",
              "I've been having thoughts that scare me. I don't want to be here.",
              "There's no point in trying anymore, nothing will get better.",
              "I feel like I'm a burden to everyone around me.",
              "I can't see a future for myself anymore.",
              "Every day I wake up wishing I hadn't.",
              "I feel trapped with no way out.",
              "I don't deserve to be here.",
              "The pain is too much to bear."] * (samples_per_class // 10 + 1),
        ],
        'label': [0] * samples_per_class + [1] * samples_per_class + [2] * samples_per_class,
        'subreddit': ['movies'] * samples_per_class + ['depression'] * samples_per_class + ['SuicideWatch'] * samples_per_class
    }
    
    df = pd.DataFrame(sample_data)
    df = df.iloc[:num_samples]  # Trim to exact number
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    df.to_csv(output_path, index=False)
    print(f"Created sample dataset with {len(df)} samples")
    print(f"Saved to: {output_path}")

def load_kaggle_mental_health_dataset(filepath: str, output_csv: str = None) -> pd.DataFrame:
    """
    Load and process the Kaggle Suicide Detection dataset.
    Converts binary labels (suicide/non-suicide) to 3-class system:
    - 0 (Neutral): Posts without mental health indicators
    - 1 (Moderate): Posts with depression/anxiety keywords
    - 2 (Severe): Posts from r/SuicideWatch
    
    Args:
        filepath: Path to Suicide_Detection.csv from Kaggle
        output_csv: Optional path to save processed dataset
        
    Returns:
        DataFrame with columns: text, label, subreddit (if available)
    """
    print(f"Loading Kaggle dataset from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Check for expected columns
    if 'text' not in df.columns or 'class' not in df.columns:
        raise ValueError("Expected columns 'text' and 'class' not found in dataset")
    
    # Define moderate-risk keywords for classification
    moderate_keywords = [
        'depress', 'anxiety', 'stress', 'sad', 'lonely', 
        'hopeless', 'worthless', 'struggling', 'therapy',
        'medication', 'mental health', 'help me', 'can\'t cope',
        'breakdown', 'overwhelmed', 'exhausted', 'insomnia'
    ]
    
    def map_to_three_classes(row):
        """Convert binary labels to 3-class system"""
        if row['class'] == 'suicide':
            return 2  # Severe - suicide risk
        
        # Check for moderate mental health indicators
        text_lower = str(row['text']).lower()
        if any(keyword in text_lower for keyword in moderate_keywords):
            return 1  # Moderate - mental health concerns
        
        return 0  # Neutral - no significant mental health indicators
    
    # Apply label mapping
    df['label'] = df.apply(map_to_three_classes, axis=1)
    
    # Clean text using existing function
    df['text'] = df['text'].apply(clean_single_text)
    
    # Remove duplicates and very short posts (less than 10 words)
    df = df.drop_duplicates(subset=['text'])
    df['word_count'] = df['text'].str.split().str.len()
    df = df[df['word_count'] >= 10]
    df = df[df['text'].str.len() > 0]  # Remove empty after cleaning
    
    # Keep only necessary columns
    df = df[['text', 'label']].copy()
    
    # Print statistics
    print(f"\n=== Kaggle Dataset Statistics ===")
    print(f"Total examples: {len(df)}")
    print(f"Class distribution:")
    print(f"  Class 0 (Neutral): {(df['label'] == 0).sum()} ({(df['label'] == 0).sum() / len(df) * 100:.1f}%)")
    print(f"  Class 1 (Moderate): {(df['label'] == 1).sum()} ({(df['label'] == 1).sum() / len(df) * 100:.1f}%)")
    print(f"  Class 2 (Severe): {(df['label'] == 2).sum()} ({(df['label'] == 2).sum() / len(df) * 100:.1f}%)")
    print(f"Average text length: {df['text'].str.len().mean():.0f} characters")
    # print(f"Average word count: {df['word_count'].mean():.0f} words")
    
    # Save processed dataset if requested
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\nProcessed dataset saved to: {output_csv}")
    
    return df

    
    print_statistics(df)
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Mental Health Data Utilities')
    parser.add_argument('--mode', type=str, default='sample',
                       choices=['sample', 'kaggle'],
                       help='Data generation mode: sample or kaggle')
    parser.add_argument('--kaggle-input', type=str,
                       help='Path to Kaggle Suicide_Detection.csv')
    parser.add_argument('--output', type=str, default='mental_health_data.csv',
                       help='Output CSV file path')
    parser.add_argument('--balance', type=str, choices=['undersample', 'oversample'],
                       help='Balance dataset classes')
    
    args = parser.parse_args()
    
    if args.mode == 'kaggle':
        if not args.kaggle_input:
            print("Error: --kaggle-input required for kaggle mode")
            print("Example: python data_utils.py --mode kaggle --kaggle-input ./data/Suicide_Detection.csv")
            exit(1)
        
        # Load Kaggle dataset
        df = load_kaggle_mental_health_dataset(args.kaggle_input, args.output)
        
    else:  # sample mode
        # Create sample dataset (300 samples = 100 per class)
        df = create_sample_dataset(args.output, num_samples=300)
    
    # Apply balancing if requested
    if args.balance:
        print(f"\nApplying {args.balance} balancing...")
        df = balance_dataset(df, method=args.balance)
        balanced_output = args.output.replace('.csv', f'_balanced_{args.balance}.csv')
        df.to_csv(balanced_output, index=False)
        print(f"Balanced dataset saved to: {balanced_output}")