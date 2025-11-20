"""
Model training module for Email Spam Detection System.
Handles model training, evaluation, and comparison.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


class ModelTrainer:
    """
    Handles model training, evaluation, and selection for spam detection.
    Supports multiple classifiers and provides comprehensive evaluation metrics.
    """
    
    def __init__(self, test_size: float = 0.2, random_state: int = 2):
        """
        Initialize the ModelTrainer.
        
        Args:
            test_size: Fraction of data to use for testing
            random_state: Random state for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.best_model = None
        self.best_model_name = None
        self.performance_results = None
        
    def initialize_classifiers(self) -> Dict[str, Any]:
        """
        Initialize all classifiers for comparison.
        
        Returns:
            Dictionary of classifier instances
        """
        classifiers = {
            'Multinomial Naive Bayes': MultinomialNB(),
            'Bernoulli Naive Bayes': BernoulliNB(),
            'Gaussian Naive Bayes': GaussianNB(),
            'Random Forest': RandomForestClassifier(
                n_estimators=50, 
                random_state=self.random_state
            ),
            'Support Vector Classifier': SVC(
                kernel='sigmoid', 
                gamma=1.0
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=5,
                random_state=self.random_state
            ),
            'Extra Tree': ExtraTreeClassifier(
                random_state=self.random_state
            ),
            'K-Neighbors': KNeighborsClassifier(
                n_neighbors=5
            )
        }
        self.models = classifiers
        return classifiers
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        return train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y
        )
    
    def train_single_model(self, model: Any, X_train: np.ndarray, 
                          y_train: np.ndarray) -> Any:
        """
        Train a single model.
        
        Args:
            model: Classifier instance
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained model
        """
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, 
                      y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate a trained model on test data.
        
        Args:
            model: Trained classifier
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0)
        }
        
        return metrics
    
    def train_and_evaluate_all(self, X_train: np.ndarray, X_test: np.ndarray,
                               y_train: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Train and evaluate all classifiers.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            
        Returns:
            DataFrame with performance metrics for all models
        """
        if not self.models:
            self.initialize_classifiers()
        
        results = []
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train model
            trained_model = self.train_single_model(model, X_train, y_train)
            
            # Evaluate model
            metrics = self.evaluate_model(trained_model, X_test, y_test)
            
            # Store results
            results.append({
                'Model': name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1_score']
            })
            
            # Update model reference
            self.models[name] = trained_model
        
        # Create performance DataFrame
        performance_df = pd.DataFrame(results)
        performance_df = performance_df.sort_values(by='Precision', ascending=False)
        self.performance_results = performance_df
        
        return performance_df
    
    def select_best_model(self, metric: str = 'Precision') -> Tuple[str, Any]:
        """
        Select the best model based on a specific metric.
        
        Args:
            metric: Metric to use for selection (default: 'Precision')
            
        Returns:
            Tuple of (model_name, model_instance)
        """
        if self.performance_results is None:
            raise ValueError("No performance results available. Train models first.")
        
        best_row = self.performance_results.iloc[0]
        best_model_name = best_row['Model']
        best_model = self.models[best_model_name]
        
        self.best_model = best_model
        self.best_model_name = best_model_name
        
        return best_model_name, best_model
    
    def get_confusion_matrix(self, X_test: np.ndarray, y_test: np.ndarray,
                            model: Any = None) -> np.ndarray:
        """
        Generate confusion matrix for a model.
        
        Args:
            X_test: Test features
            y_test: Test labels
            model: Model to evaluate (uses best model if None)
            
        Returns:
            Confusion matrix
        """
        if model is None:
            model = self.best_model
        
        if model is None:
            raise ValueError("No model available. Train a model first.")
        
        y_pred = model.predict(X_test)
        return confusion_matrix(y_test, y_pred)
    
    def get_classification_report(self, X_test: np.ndarray, y_test: np.ndarray,
                                 model: Any = None) -> str:
        """
        Generate classification report for a model.
        
        Args:
            X_test: Test features
            y_test: Test labels
            model: Model to evaluate (uses best model if None)
            
        Returns:
            Classification report as string
        """
        if model is None:
            model = self.best_model
        
        if model is None:
            raise ValueError("No model available. Train a model first.")
        
        y_pred = model.predict(X_test)
        return classification_report(y_test, y_pred, 
                                    target_names=['Ham', 'Spam'])
    
    def save_model(self, model_path: Path, vectorizer=None, 
                   scaler=None, vectorizer_path: Path = None,
                   scaler_path: Path = None):
        """
        Save trained model and preprocessing objects.
        
        Args:
            model_path: Path to save the model
            vectorizer: TF-IDF vectorizer to save
            scaler: MinMax scaler to save
            vectorizer_path: Path to save vectorizer
            scaler_path: Path to save scaler
        """
        if self.best_model is None:
            raise ValueError("No best model selected. Train and select a model first.")
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        print(f"Model saved to {model_path}")
        
        # Save vectorizer
        if vectorizer is not None and vectorizer_path is not None:
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(vectorizer, f)
            print(f"Vectorizer saved to {vectorizer_path}")
        
        # Save scaler
        if scaler is not None and scaler_path is not None:
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"Scaler saved to {scaler_path}")
    
    @staticmethod
    def load_model(model_path: Path) -> Any:
        """
        Load a saved model.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded model
        """
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    
    def complete_training_pipeline(self, X: np.ndarray, y: np.ndarray,
                                   vectorizer=None, scaler=None,
                                   model_path: Path = None,
                                   vectorizer_path: Path = None,
                                   scaler_path: Path = None) -> Dict[str, Any]:
        """
        Complete training pipeline: split, train, evaluate, select, and save.
        
        Args:
            X: Feature matrix
            y: Target vector
            vectorizer: TF-IDF vectorizer
            scaler: MinMax scaler
            model_path: Path to save model
            vectorizer_path: Path to save vectorizer
            scaler_path: Path to save scaler
            
        Returns:
            Dictionary with training results and best model info
        """
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Initialize classifiers
        self.initialize_classifiers()
        
        # Train and evaluate all models
        performance_df = self.train_and_evaluate_all(X_train, X_test, y_train, y_test)
        
        # Select best model
        best_model_name, best_model = self.select_best_model()
        
        # Get detailed metrics for best model
        best_metrics = self.evaluate_model(best_model, X_test, y_test)
        confusion_mat = self.get_confusion_matrix(X_test, y_test)
        classification_rep = self.get_classification_report(X_test, y_test)
        
        # Save model if paths provided
        if model_path is not None:
            self.save_model(model_path, vectorizer, scaler, 
                          vectorizer_path, scaler_path)
        
        return {
            'performance_df': performance_df,
            'best_model_name': best_model_name,
            'best_model': best_model,
            'best_metrics': best_metrics,
            'confusion_matrix': confusion_mat,
            'classification_report': classification_rep
        }
