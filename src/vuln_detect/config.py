"""
Unified configuration management for the vulnerability detection system.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Unified configuration for data, model, and training."""

    # --- Preprocessing Configuration ---
    run_preprocessing: bool = False
    raw_data_path: str = "data/SuicideAndDepression_Detection.csv"
    processed_data_dir: str = "data/processed"
    balance_strategy: Optional[str] = None  # 'undersample' or 'oversample'
    train_split: float = 0.7
    val_split: float = 0.15

    # --- Data Configuration ---
    train_data_path: str = "data/processed/train.csv"
    val_data_path: str = "data/processed/validation.csv"
    test_data_path: str = "data/processed/test.csv"
    val_sample_size: Optional[int] = None # Number of samples to use for validation during training
    max_length: int = 256  # Matched from original project
    random_seed: int = 42
    text_column: str = "text"
    label_column: str = "label"

    # --- Model Configuration ---
    model_name: str = "cardiffnlp/twitter-roberta-base"
    num_labels: int = 3
    use_gradient_checkpointing: bool = True  # Matched from original project
    dropout: float = 0.1

    # --- Training Configuration ---
    output_dir: str = "models/vuln_detector"
    num_epochs: int = 3
    batch_size: int = 4  # Matched from original project
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 0  # Original project did not use warmup
    logging_steps: int = 100
    gradient_accumulation_steps: int = 4  # Matched from original project
    fp16: bool = False
    device: str = "auto"
    report_to: str = "none"
    save_strategy: str = "epoch"  # Matched from original project
    eval_strategy: str = "epoch"  # Matched from original project
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"

    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from a YAML file."""
        logger.info(f"Loading configuration from: {config_path}")
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Flatten the nested structure
        flat_config = {}
        for key, value in config_dict.items():
            if isinstance(value, dict):
                flat_config.update(value)
            else:
                flat_config[key] = value
        
        return cls(**flat_config)
