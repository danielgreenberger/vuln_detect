"""
Evaluation and Inference Script for Vulnerability Detection
Loads a fine-tuned model and makes predictions on new social media posts.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import List, Dict, Union

# dgreenbe TODO: what kind of data preprocessing should I do?
# Explanation: Currently using raw text. For mental health detection, original capitalization and punctuation may carry important emotional signals (e.g., "HELP!!!" vs "help").
# Basic cleaning is done in data_utils.py.
class VulnerabilityDetector:
    """Class for loading and using the fine-tuned vulnerability detection model"""
    
    def __init__(self, model_path="vulnerability_detector_model"):
        """
        Initialize the detector with a trained model
        
        Args:
            model_path: Path to the saved model directory
        """
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model from: {model_path}")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        # Explanation: model.eval() switches the model to evaluation mode. This disables dropout layers (which randomly drop neurons during training)
        # and puts batch normalization in inference mode. Essential for consistent predictions.
        self.model.eval() # dgreenbe TODO: what is this? Why do we need this?
        
        # Load training metadata
        metadata_path = Path(model_path) / "training_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
                self.id2label = self.metadata.get('id2label', {0: "Neutral", 1: "Moderate", 2: "Severe"})
                self.label_names = self.metadata.get('label_names', ["Neutral", "Moderate", "Severe"])
        else:
            # Default labels
            self.id2label = {0: "Neutral", 1: "Moderate", 2: "Severe"}
            self.label_names = ["Neutral", "Moderate", "Severe"]
        
        print(f"Model loaded successfully!")
        print(f"Labels: {self.label_names}")
    
    def predict_single(self, text: str, return_probabilities: bool = False) -> Union[Dict, str]:
        """
        Predict vulnerability level for a single text
        
        Args:
            text: Input text to analyze
            return_probabilities: If True, return probabilities for all classes
            
        Returns:
            Prediction result (label and optionally probabilities)
        """
        # Tokenize input
            # Explanation: truncation=True cuts text to 512 tokens (RoBERTa's max). This is a valid concern - critical mental health signals might appear at the end of long posts.
            # Alternatives: summarization, hierarchical models, or finding patterns in where crisis signals typically appear.
        inputs = self.tokenizer(
            text,
            truncation=True, # dgreenbe TODO: do we really want to trunk it? Can't we figure out the entire post? Could be dangerous is the last sentence is the major red-flag...
            max_length=512,
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # Explanation: The tokenizer converts text to token IDs that the model understands. Separating them allows flexibility -
        # you can swap tokenizers, apply different preprocessing, or use the same tokenizer for multiple models.
        
        # Get prediction
        with torch.no_grad(): # dgreenbe TODO: why do we even need to break the model into tokenizer and the rest? Couldn't we just use it as a black box?
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
        
        # Get predicted label
        predicted_label = self.id2label.get(str(predicted_class), "Unknown")
        
        if return_probabilities:
            probs_dict = {
                self.label_names[i]: float(probabilities[0][i])
                for i in range(len(self.label_names))
            }
            return {
                "text": text,
                "predicted_label": predicted_label,
                "predicted_class": predicted_class,
                "probabilities": probs_dict,
                "confidence": float(probabilities[0][predicted_class])
            }
        else:
            return predicted_label
    
    def predict_batch(self, texts: List[str], batch_size: int = 16) -> List[Dict]:
        """
        Predict vulnerability levels for multiple texts
        
        Args:
            texts: List of texts to analyze
            batch_size: Batch size for processing
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predicted_classes = torch.argmax(probabilities, dim=-1)
            
            # Process results
            for j, text in enumerate(batch_texts):
                predicted_class = predicted_classes[j].item()
                predicted_label = self.id2label.get(str(predicted_class), "Unknown")
                
                probs_dict = {
                    self.label_names[k]: float(probabilities[j][k])
                    for k in range(len(self.label_names))
                }
                
                results.append({
                    "text": text,
                    "predicted_label": predicted_label,
                    "predicted_class": predicted_class,
                    "probabilities": probs_dict,
                    "confidence": float(probabilities[j][predicted_class])
                })
        
        return results
    
    def evaluate_on_data(self, texts: List[str], labels: List[int]) -> Dict:
        """
        Evaluate model on labeled data
        
        Args:
            texts: List of texts
            labels: List of true labels (0, 1, or 2)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if len(texts) != len(labels):
            raise ValueError("Number of texts and labels must match")
        
        # Get predictions
        predictions = self.predict_batch(texts)
        predicted_classes = [p['predicted_class'] for p in predictions]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        from sklearn.metrics import confusion_matrix, classification_report
        
        accuracy = accuracy_score(labels, predicted_classes)
        f1 = f1_score(labels, predicted_classes, average='weighted')
        precision = precision_score(labels, predicted_classes, average='weighted')
        recall = recall_score(labels, predicted_classes, average='weighted')
        # Explanation: Confusion matrix shows prediction accuracy breakdown: rows=actual labels, columns=predicted labels. Diagonal=correct predictions.
        # Off-diagonal=errors. Critical for identifying which classes the model confuses (e.g., does it miss severe cases?).
        
        # Confusion matrix
        cm = confusion_matrix(labels, predicted_classes) # dgreenbe TODO: what's this?
        
        # Classification report
        report = classification_report(
            labels,
            predicted_classes,
            target_names=self.label_names,
            output_dict=True
        )
        
        results = {
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "num_samples": len(texts)
        }
        
        return results
    
    def print_evaluation_results(self, results: Dict):
        """Pretty print evaluation results"""
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        print(f"\nTotal Samples: {results['num_samples']}")
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {results['accuracy']:.4f}")
        print(f"  F1 Score:  {results['f1_score']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        
        print(f"\nPer-Class Metrics:")
        for label_name in self.label_names:
            metrics = results['classification_report'][label_name]
            print(f"\n  {label_name}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall:    {metrics['recall']:.4f}")
            print(f"    F1-Score:  {metrics['f1-score']:.4f}")
            print(f"    Support:   {int(metrics['support'])}")
        
        print(f"\nConfusion Matrix:")
        print(f"  (rows = true labels, columns = predictions)")
        cm = np.array(results['confusion_matrix'])
        
        # Print header
        header = "       " + "  ".join([f"{name:8s}" for name in self.label_names])
        print(header)
        print("  " + "-" * len(header))
        
        # Print rows
        for i, label_name in enumerate(self.label_names):
            row = f"  {label_name:8s} "
            row += "  ".join([f"{cm[i][j]:8d}" for j in range(len(self.label_names))])
            print(row)


def load_and_split_test_data(csv_path: str, test_size: float = 0.15):
    """
    Load CSV and extract the test set using the same split logic as training.
    This ensures we evaluate on the exact same held-out test set.
    
    Args:
        csv_path: Path to processed CSV file
        test_size: Percentage for test set (default 0.15 = 15%)
    
    Returns:
        DataFrame with test set only
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    # Load full dataset
    df = pd.read_csv(csv_path)
    
    # First split: 70% train, 30% val+test
    train_df, val_test_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df['label']
    )
    
    # Second split: split the 30% into 15% val, 15% test
    val_df, test_df = train_test_split(
        val_test_df, test_size=0.5, random_state=42, stratify=val_test_df['label']
    )
    
    print(f"Loaded test set from {csv_path}")
    print(f"Test set size: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    print(f"Class distribution: {test_df['label'].value_counts().sort_index().to_dict()}")
    
    return test_df


def load_kaggle_csv_for_testing(csv_path: str, n_samples: int = None):
    """
    Load raw Kaggle CSV and convert to test format.
    Kaggle CSV has 'class' column with 'suicide'/'non-suicide' values.
    Converts to our 3-class system for testing.
    
    Args:
        csv_path: Path to raw Kaggle Suicide_Detection.csv
        n_samples: Number of random samples to take (None = all)
    
    Returns:
        DataFrame with 'text' and 'label' columns (0=neutral, 2=severe)
    """
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    # Check for Kaggle format
    if 'class' not in df.columns:
        raise ValueError("Kaggle CSV must have 'class' column with 'suicide'/'non-suicide' values")
    
    # Convert binary labels to our system
    # suicide → 2 (severe), non-suicide → 0 (neutral)
    # Note: Binary Kaggle data doesn't have 'moderate' (1) class
    df['label'] = df['class'].apply(lambda x: 2 if x == 'suicide' else 0)
    
    # Sample if requested
    if n_samples is not None and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=42)
        print(f"Sampled {n_samples} random posts from {csv_path}")
    else:
        print(f"Loaded {len(df)} posts from {csv_path}")
    
    # Print class distribution
    print(f"Class distribution:")
    print(f"  Neutral (0): {(df['label'] == 0).sum()} ({(df['label'] == 0).sum() / len(df) * 100:.1f}%)")
    print(f"  Severe (2): {(df['label'] == 2).sum()} ({(df['label'] == 2).sum() / len(df) * 100:.1f}%)")
    print(f"Note: Binary Kaggle data has no 'Moderate' (1) class")
    
    return df[['text', 'label']]


def demo_prediction():
    """Demo function showing how to use the detector"""
    print("=" * 70)
    print("VULNERABILITY DETECTOR - DEMO")
    print("=" * 70)
    
    # Initialize detector
    detector = VulnerabilityDetector()
    
    # Test examples
    test_texts = [
        "Just finished a great workout at the gym! Feeling energized.",
        "I've been feeling really down lately, struggling to get out of bed.",
        "I don't think I can keep going. Everything feels hopeless.",
    ]
    
    print("\n" + "=" * 70)
    print("PREDICTIONS")
    print("=" * 70)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n--- Example {i} ---")
        print(f"Text: {text}")
        
        result = detector.predict_single(text, return_probabilities=True)
        print(f"\nPredicted Risk Level: {result['predicted_label']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"\nProbabilities:")
        for label, prob in result['probabilities'].items():
            print(f"  {label:10s}: {prob:.4f}")


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Vulnerability Detection - Prediction and Evaluation')
    parser.add_argument("--model", type=str, default="vulnerability_detector_model",
                       help="Path to model directory")
    parser.add_argument("--text", type=str, default=None,
                       help="Single text to predict")
    parser.add_argument("--demo", action="store_true",
                       help="Run demo predictions")
    parser.add_argument("--eval-data", type=str, default=None,
                       help="Path to CSV file for evaluation (with text and label columns)")
    parser.add_argument('--test-split', action='store_true',
                       help='Use with --eval-data to automatically extract test split (15%% held-out set)')
    parser.add_argument('--kaggle-format', action='store_true',
                       help='Input is raw Kaggle CSV (binary labels: suicide/non-suicide)')
    parser.add_argument('--n-samples', type=int, default=None,
                       help='Randomly sample N posts from CSV for testing')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = VulnerabilityDetector(args.model)
    
    # Demo mode
    if args.demo:
        demo_texts = [
            "I'm feeling great today, everything is wonderful!",
            "I've been feeling down lately, might need to talk to someone",
            "I can't take this anymore, I want to end it all"
        ]
        print("\n" + "="*70)
        print("DEMO MODE - Sample Predictions")
        print("="*70)
        
        for i, text in enumerate(demo_texts, 1):
            result = detector.predict_single(text, return_probabilities=True)
            print(f"\n{i}. Text: {text}")
            print(f"   Prediction: {result['predicted_label']}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Probabilities: {result['probabilities']}")
    
    # Single text prediction
    elif args.text:
        result = detector.predict_single(args.text, return_probabilities=True)
        print("\n" + "="*70)
        print("PREDICTION")
        print("="*70)
        print(f"Text: {args.text}")
        print(f"Predicted Class: {result['predicted_label']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"\nProbabilities:")
        for label, prob in result['probabilities'].items():
            print(f"  {label}: {prob:.4f}")
    
    # Test split evaluation (NEW)
    elif args.test_split and args.eval_data:
        print("\n" + "="*70)
        print("TEST SET EVALUATION")
        print("="*70)
        
        # Load test set using same split as training
        test_df = load_and_split_test_data(args.eval_data)
        
        # Evaluate
        results = detector.evaluate_on_data(
            texts=test_df['text'].tolist(),
            labels=test_df['label'].tolist()
        )
        
        detector.print_evaluation_results(results)
        
        # Safety check for severe class
        severe_report = results['classification_report']['Severe']
        severe_recall = severe_report['recall']
        
        print(f"\n{'='*70}")
        print("SAFETY CHECK: Severe Class Recall")
        print(f"{'='*70}")
        print(f"Recall: {severe_recall:.4f} ({severe_recall:.1%})")
        
        SAFETY_THRESHOLD = 0.90
        if severe_recall >= SAFETY_THRESHOLD:
            print(f"✓ PASSED: Meets {SAFETY_THRESHOLD:.0%} safety threshold")
        else:
            print(f"✗ WARNING: Below {SAFETY_THRESHOLD:.0%} safety threshold")
            print(f"  Model misses {(1-severe_recall):.1%} of severe cases")
    
    # Full CSV evaluation
    elif args.eval_data:
        import pandas as pd
        
        # Load data based on format
        if args.kaggle_format:
            print("\n" + "="*70)
            print("LOADING RAW KAGGLE CSV")
            print("="*70)
            df = load_kaggle_csv_for_testing(args.eval_data, n_samples=args.n_samples)
        else:
            df = pd.read_csv(args.eval_data)
            
            # Sample if requested
            if args.n_samples is not None and args.n_samples < len(df):
                df = df.sample(n=args.n_samples, random_state=42)
                print(f"Sampled {args.n_samples} random posts")
        
        print("\n" + "="*70)
        print(f"EVALUATING ON: {args.eval_data}")
        print("="*70)
        print(f"Dataset size: {len(df)} samples")
        
        results = detector.evaluate_on_data(
            texts=df['text'].tolist(),
            labels=df['label'].tolist()
        )
        
        detector.print_evaluation_results(results)
        
        # Note for binary evaluation
        if args.kaggle_format:
            print("\n" + "="*70)
            print("NOTE: Binary Evaluation")
            print("="*70)
            print("Kaggle CSV has binary labels (neutral/severe only)")
            print("Your model outputs 3 classes (neutral/moderate/severe)")
            print("Metrics show how well model handles clear-cut cases")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()