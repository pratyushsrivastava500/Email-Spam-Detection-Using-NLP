"""
Configuration settings for Email Spam Detection System.
Contains all constants, paths, and hyperparameters.
"""

from pathlib import Path
from typing import Dict, Any

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Data paths
DATA_DIR = BASE_DIR / "Data"
MODELS_DIR = BASE_DIR / "models"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Dataset paths
DATASET_PATH = DATA_DIR / "spam.csv"
MODEL_PATH = MODELS_DIR / "model.pkl"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# Model hyperparameters
MAX_FEATURES = 3000
TEST_SIZE = 0.2
RANDOM_STATE = 2
CONFIDENCE_THRESHOLD = 0.5

# Streamlit UI configuration
PAGE_TITLE = "SpamShield: Email & SMS Classifier"
PAGE_ICON = "📰"
BACKGROUND_IMAGE = "Image.webp"

# Text preprocessing settings
REMOVE_STOPWORDS = True
APPLY_STEMMING = True
LANGUAGE = "english"

# Model training settings
CLASSIFIERS_CONFIG: Dict[str, Dict[str, Any]] = {
    "MultinomialNB": {
        "enabled": True,
        "params": {}
    },
    "BernoulliNB": {
        "enabled": True,
        "params": {}
    },
    "RandomForest": {
        "enabled": True,
        "params": {
            "n_estimators": 50,
            "random_state": RANDOM_STATE
        }
    },
    "SVC": {
        "enabled": True,
        "params": {
            "kernel": "sigmoid",
            "gamma": 1.0
        }
    },
    "DecisionTree": {
        "enabled": True,
        "params": {
            "max_depth": 5,
            "random_state": RANDOM_STATE
        }
    },
    "KNeighbors": {
        "enabled": True,
        "params": {
            "n_neighbors": 5
        }
    },
    "ExtraTree": {
        "enabled": True,
        "params": {
            "random_state": RANDOM_STATE
        }
    },
    "GaussianNB": {
        "enabled": True,
        "params": {}
    }
}

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class Config:
    """Configuration class for easy access to settings."""
    
    def __init__(self):
        self.base_dir = BASE_DIR
        self.data_dir = DATA_DIR
        self.models_dir = MODELS_DIR
        self.dataset_path = DATASET_PATH
        self.model_path = MODEL_PATH
        self.vectorizer_path = VECTORIZER_PATH
        self.scaler_path = SCALER_PATH
        self.max_features = MAX_FEATURES
        self.test_size = TEST_SIZE
        self.random_state = RANDOM_STATE
        self.page_title = PAGE_TITLE
        self.page_icon = PAGE_ICON
        self.classifiers_config = CLASSIFIERS_CONFIG
        
    @staticmethod
    def ensure_directories():
        """Create necessary directories if they don't exist."""
        DATA_DIR.mkdir(exist_ok=True, parents=True)
        MODELS_DIR.mkdir(exist_ok=True, parents=True)
        NOTEBOOKS_DIR.mkdir(exist_ok=True, parents=True)
