"""
Prediction module for Email Spam Detection System.
Handles inference and spam classification for new messages.
"""

import pickle
import numpy as np
from typing import Tuple, Optional, Any
from pathlib import Path
from .data_preprocessing import DataPreprocessor


class SpamPredictor:
    """
    Handles spam prediction for email and SMS messages.
    Loads trained models and makes predictions on new text.
    """
    
    def __init__(self, model_path: Path, vectorizer_path: Path, 
                 scaler_path: Optional[Path] = None):
        """
        Initialize the SpamPredictor.
        
        Args:
            model_path: Path to the trained model file
            vectorizer_path: Path to the TF-IDF vectorizer file
            scaler_path: Path to the scaler file (optional)
        """
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.scaler_path = scaler_path
        
        # Initialize preprocessor
        self.preprocessor = DataPreprocessor()
        
        # Load model and preprocessing objects
        self.model = self._load_model()
        self.vectorizer = self._load_vectorizer()
        self.scaler = self._load_scaler() if scaler_path else None
        
    def _load_model(self) -> Any:
        """
        Load the trained model from disk.
        
        Returns:
            Loaded model object
        """
        try:
            with open(self.model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"Model loaded successfully from {self.model_path}")
            return model
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def _load_vectorizer(self) -> Any:
        """
        Load the TF-IDF vectorizer from disk.
        
        Returns:
            Loaded vectorizer object
        """
        try:
            with open(self.vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            print(f"Vectorizer loaded successfully from {self.vectorizer_path}")
            return vectorizer
        except FileNotFoundError:
            raise FileNotFoundError(f"Vectorizer file not found at {self.vectorizer_path}")
        except Exception as e:
            raise Exception(f"Error loading vectorizer: {str(e)}")
    
    def _load_scaler(self) -> Optional[Any]:
        """
        Load the MinMax scaler from disk.
        
        Returns:
            Loaded scaler object or None
        """
        if self.scaler_path is None:
            return None
        
        try:
            with open(self.scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print(f"Scaler loaded successfully from {self.scaler_path}")
            return scaler
        except FileNotFoundError:
            print(f"Warning: Scaler file not found at {self.scaler_path}")
            return None
        except Exception as e:
            print(f"Warning: Error loading scaler: {str(e)}")
            return None
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess input text using the same pipeline as training.
        
        Args:
            text: Raw input text
            
        Returns:
            Preprocessed text
        """
        return self.preprocessor.transform_text(text)
    
    def predict(self, text: str) -> Tuple[int, str]:
        """
        Predict whether a message is spam or not.
        
        Args:
            text: Input message text
            
        Returns:
            Tuple of (prediction, label) where:
                - prediction: 0 for ham, 1 for spam
                - label: "Ham" or "Spam"
        """
        # Preprocess the text
        transformed_text = self.preprocess_text(text)
        
        # Vectorize the text
        vector_input = self.vectorizer.transform([transformed_text]).toarray()
        
        # Apply scaling if scaler is available
        if self.scaler is not None:
            vector_input = self.scaler.transform(vector_input)
        
        # Make prediction
        prediction = self.model.predict(vector_input)[0]
        
        # Convert to label
        label = "Spam" if prediction == 1 else "Ham"
        
        return int(prediction), label
    
    def predict_proba(self, text: str) -> Tuple[float, float]:
        """
        Predict probability of message being spam.
        
        Args:
            text: Input message text
            
        Returns:
            Tuple of (ham_probability, spam_probability)
        """
        # Preprocess the text
        transformed_text = self.preprocess_text(text)
        
        # Vectorize the text
        vector_input = self.vectorizer.transform([transformed_text]).toarray()
        
        # Apply scaling if scaler is available
        if self.scaler is not None:
            vector_input = self.scaler.transform(vector_input)
        
        # Get probability predictions if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(vector_input)[0]
            return float(probabilities[0]), float(probabilities[1])
        else:
            # If model doesn't support probability, return binary prediction
            prediction = self.model.predict(vector_input)[0]
            if prediction == 0:
                return 1.0, 0.0
            else:
                return 0.0, 1.0
    
    def predict_batch(self, texts: list) -> list:
        """
        Predict spam classification for multiple messages.
        
        Args:
            texts: List of input messages
            
        Returns:
            List of tuples (prediction, label) for each message
        """
        results = []
        for text in texts:
            prediction, label = self.predict(text)
            results.append((prediction, label))
        return results
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'model_type': type(self.model).__name__,
            'model_path': str(self.model_path),
            'vectorizer_path': str(self.vectorizer_path),
            'scaler_loaded': self.scaler is not None
        }
        
        # Add model-specific parameters if available
        if hasattr(self.model, 'get_params'):
            info['model_params'] = self.model.get_params()
        
        return info
