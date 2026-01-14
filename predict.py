#!/usr/bin/env python3
"""
A script for making predictions with a trained vulnerability detection model.

This script can be used as a command-line tool or imported as a module.
It encapsulates the model loading, tokenization, and prediction logic in a
`Predictor` class.

Usage (CLI):
    python predict.py --model-path "outputs/batch_1_run/final_model" --text "I feel so down today."

Usage (Module):
    from predict import Predictor
    predictor = Predictor("path/to/your/model")
    result = predictor.predict("Some input text.")
    print(result)
"""

import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Mapping from label index to class name
LABEL_MAP = {0: "Safe", 1: "Depression", 2: "Suicide"}


class Predictor:
    """Encapsulates a trained model and provides an interface for prediction."""

    def __init__(self, model_path: str):
        """
        Initializes the Predictor by loading the model and tokenizer.

        Args:
            model_path (str): The path to the directory containing the saved
                              Hugging Face model and tokenizer.
        """
        self.device = torch.device("cpu")
        logger.info(f"Using device: {self.device}")

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        logger.info(f"Loading model and tokenizer from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()  # Set the model to evaluation mode

    def predict(self, text: str) -> dict:
        """
        Makes a prediction on a single piece of text.

        Args:
            text (str): The input text to classify.

        Returns:
            dict: A dictionary containing the predicted label, confidence score,
                  and probabilities for all classes.
                  Example: {
                      'label': 'Suicide',
                      'confidence': 0.95,
                      'probabilities': {'Safe': 0.02, 'Depression': 0.03, 'Suicide': 0.95}
                  }
        """
        with torch.no_grad():
            # Tokenize the input text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.model.config.max_position_embeddings,
                padding=True,
            ).to(self.device)

            # Get model outputs
            outputs = self.model(**inputs)
            logits = outputs.logits

            # Apply softmax to get probabilities
            probabilities = torch.softmax(logits, dim=-1)

            # Get all class probabilities
            all_probs = probabilities[0].tolist()
            prob_dict = {LABEL_MAP[i]: all_probs[i] for i in range(len(all_probs))}

            # Get the top prediction
            confidence, predicted_class_idx = torch.max(probabilities, dim=-1)
            predicted_label = LABEL_MAP.get(
                predicted_class_idx.item(), "Unknown"
            )

            return {
                "label": predicted_label,
                "confidence": confidence.item(),
                "probabilities": prob_dict,
            }


def main():
    """Command-line interface for the prediction script."""
    parser = argparse.ArgumentParser(
        description="Predict vulnerability from text using a trained model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the directory containing the saved model and tokenizer.",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="The input text to classify.",
    )
    args = parser.parse_args()

    try:
        predictor = Predictor(args.model_path)
        result = predictor.predict(args.text)

        logger.info("\n" + "=" * 30)
        logger.info("PREDICTION RESULT")
        logger.info("=" * 30)
        logger.info(f"  Input Text: '{args.text}'")
        logger.info(f"  Predicted Label: {result['label']}")
        logger.info(f"  Confidence: {result['confidence']:.4f}")
        logger.info("=" * 30)

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()