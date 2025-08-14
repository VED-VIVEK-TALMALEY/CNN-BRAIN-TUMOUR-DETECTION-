"""
Training package for Brain Tumor Detection
"""

from .train_model import main as train_main, validate_dataset

__all__ = ['train_main', 'validate_dataset']
