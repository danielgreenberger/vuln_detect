"""Vulnerability Detection package.

A streamlined package for detecting mental health vulnerabilities in text,
using transformer-based models for classification.
"""

from .config import Config
from .data.loaders import load_data

__version__ = "0.2.0"

__all__ = [
    # Configuration
    "Config",
    # Data
    "load_data",
]