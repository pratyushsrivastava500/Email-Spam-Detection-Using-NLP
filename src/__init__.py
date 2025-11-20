"""Source package for Email Spam Detection System."""

from .data_preprocessing import DataPreprocessor
from .model_training import ModelTrainer
from .prediction import SpamPredictor
from .utils import *

__all__ = [
    'DataPreprocessor',
    'ModelTrainer',
    'SpamPredictor'
]
