"""
Data preprocessing module for Email Spam Detection System.
Handles text cleaning, transformation, and feature extraction.
"""

import re
import string
import pandas as pd
import numpy as np
import nltk
from typing import List, Tuple, Optional
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


class DataPreprocessor:
    """
    Handles all data preprocessing tasks including:
    - Text cleaning and transformation
    - Feature extraction
    - Data loading and preparation
    """
    
    def __init__(self, language: str = 'english', apply_stemming: bool = True):
        """
        Initialize the DataPreprocessor.
        
        Args:
            language: Language for stopwords (default: 'english')
            apply_stemming: Whether to apply stemming (default: True)
        """
        self.language = language
        self.apply_stemming = apply_stemming
        self.stemmer = PorterStemmer() if apply_stemming else None
        
        # Download required NLTK data
        self._download_nltk_resources()
        
        # Load stopwords
        self.stop_words = set(stopwords.words(self.language))
        
        # Initialize encoders
        self.label_encoder = LabelEncoder()
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.scaler: Optional[MinMaxScaler] = None
    
    @staticmethod
    def _download_nltk_resources():
        """Download required NLTK resources."""
        required_resources = ['stopwords', 'punkt']
        for resource in required_resources:
            try:
                if resource == 'punkt':
                    nltk.data.find('tokenizers/punkt')
                else:
                    nltk.data.find(f'corpora/{resource}')
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                except:
                    pass
    
    def load_data(self, filepath: Path) -> pd.DataFrame:
        """
        Load and perform initial cleaning of the dataset.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            Cleaned pandas DataFrame
        """
        # Read the dataset
        data = pd.read_csv(filepath, encoding='latin')
        
        # Drop unnecessary columns
        columns_to_drop = [col for col in data.columns if 'Unnamed' in col]
        if columns_to_drop:
            data.drop(columns_to_drop, axis=1, inplace=True)
        
        # Rename columns to standard names
        if 'v1' in data.columns and 'v2' in data.columns:
            data.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
        
        # Handle duplicates
        data.drop_duplicates(keep='first', inplace=True)
        
        # Handle missing values
        data.dropna(inplace=True)
        
        return data
    
    def transform_text(self, text: str) -> str:
        """
        Transform text through multiple preprocessing steps:
        1. Convert to lowercase
        2. Tokenization
        3. Remove non-alphanumeric characters
        4. Remove stopwords
        5. Apply stemming (optional)
        
        Args:
            text: Input text string
            
        Returns:
            Processed text string
        """
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize
        text = nltk.word_tokenize(text)
        
        # Remove non-alphanumeric tokens
        text = [word for word in text if word.isalnum()]
        
        # Remove stopwords and punctuation
        text = [word for word in text if word not in self.stop_words 
                and word not in string.punctuation]
        
        # Apply stemming
        if self.apply_stemming and self.stemmer:
            text = [self.stemmer.stem(word) for word in text]
        
        # Join back to string
        return ' '.join(text)
    
    def add_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered features to the dataset.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with additional features
        """
        # Number of characters
        data['num_chars'] = data['text'].apply(len)
        
        # Number of words
        data['num_words'] = data['text'].apply(lambda x: len(nltk.word_tokenize(x)))
        
        # Number of sentences
        data['num_sentences'] = data['text'].apply(
            lambda x: len(nltk.sent_tokenize(x)) if x else 0
        )
        
        return data
    
    def encode_labels(self, data: pd.DataFrame, target_col: str = 'target') -> pd.DataFrame:
        """
        Encode target labels to numerical values.
        
        Args:
            data: Input DataFrame
            target_col: Name of the target column
            
        Returns:
            DataFrame with encoded labels
        """
        data[target_col] = self.label_encoder.fit_transform(data[target_col])
        return data
    
    def prepare_features(self, data: pd.DataFrame, max_features: int = 3000,
                        apply_scaling: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for model training using TF-IDF vectorization.
        
        Args:
            data: Input DataFrame with 'transformed_text' column
            max_features: Maximum number of features for TF-IDF
            apply_scaling: Whether to apply MinMax scaling
            
        Returns:
            Tuple of (X, y) arrays
        """
        # Initialize vectorizer if not already done
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(max_features=max_features)
            X = self.vectorizer.fit_transform(data['transformed_text']).toarray()
        else:
            X = self.vectorizer.transform(data['transformed_text']).toarray()
        
        # Apply scaling if requested
        if apply_scaling:
            if self.scaler is None:
                self.scaler = MinMaxScaler()
                X = self.scaler.fit_transform(X)
            else:
                X = self.scaler.transform(X)
        
        # Get target variable
        y = data['target'].values
        
        return X, y
    
    def preprocess_pipeline(self, filepath: Path, max_features: int = 3000,
                           apply_scaling: bool = True) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Complete preprocessing pipeline.
        
        Args:
            filepath: Path to the dataset
            max_features: Maximum number of features for TF-IDF
            apply_scaling: Whether to apply MinMax scaling
            
        Returns:
            Tuple of (processed_dataframe, X, y)
        """
        # Load data
        data = self.load_data(filepath)
        
        # Encode labels
        data = self.encode_labels(data)
        
        # Add features
        data = self.add_features(data)
        
        # Transform text
        data['transformed_text'] = data['text'].apply(self.transform_text)
        
        # Prepare features
        X, y = self.prepare_features(data, max_features, apply_scaling)
        
        return data, X, y
    
    def get_corpus(self, data: pd.DataFrame, target_value: int) -> List[str]:
        """
        Extract word corpus for a specific target class.
        
        Args:
            data: DataFrame with 'transformed_text' and 'target' columns
            target_value: Target class value (0 or 1)
            
        Returns:
            List of words in the corpus
        """
        corpus = []
        for message in data[data['target'] == target_value]['transformed_text']:
            for word in message.split():
                corpus.append(word)
        return corpus
