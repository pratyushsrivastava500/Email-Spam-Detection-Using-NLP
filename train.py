"""
Training script for Email Spam Detection System.
Trains multiple models, evaluates performance, and saves the best model.
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config.config import Config
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.utils import (
    print_data_summary,
    plot_class_distribution,
    plot_feature_distributions,
    plot_model_comparison,
    plot_confusion_matrix,
    save_results_to_file
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train spam detection models'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to the dataset CSV file'
    )
    parser.add_argument(
        '--max-features',
        type=int,
        default=3000,
        help='Maximum number of features for TF-IDF (default: 3000)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set size as fraction (default: 0.2)'
    )
    parser.add_argument(
        '--no-scaling',
        action='store_true',
        help='Disable feature scaling'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization plots'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save models and results'
    )
    
    return parser.parse_args()


def main():
    """Main training pipeline."""
    
    # Parse arguments
    args = parse_arguments()
    
    # Initialize configuration
    config = Config()
    config.ensure_directories()
    
    # Set paths
    data_path = Path(args.data_path) if args.data_path else config.dataset_path
    output_dir = Path(args.output_dir) if args.output_dir else config.models_dir
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 80)
    print("EMAIL SPAM DETECTION - MODEL TRAINING")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Dataset: {data_path}")
    print(f"  Max Features: {args.max_features}")
    print(f"  Test Size: {args.test_size}")
    print(f"  Apply Scaling: {not args.no_scaling}")
    print(f"  Output Directory: {output_dir}")
    print(f"  Visualizations: {args.visualize}")
    print()
    
    # Step 1: Data Preprocessing
    print("=" * 80)
    print("STEP 1: DATA PREPROCESSING")
    print("=" * 80)
    
    preprocessor = DataPreprocessor()
    
    print("\n[1/4] Loading and preprocessing data...")
    data, X, y = preprocessor.preprocess_pipeline(
        filepath=data_path,
        max_features=args.max_features,
        apply_scaling=not args.no_scaling
    )
    
    print(f"✓ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    print("\n[2/4] Dataset summary:")
    print_data_summary(data)
    
    # Visualizations
    if args.visualize:
        print("\n[3/4] Generating visualizations...")
        
        # Class distribution
        plot_class_distribution(data, save_path=output_dir / "class_distribution.png")
        
        # Feature distributions
        features_to_plot = ['num_chars', 'num_words', 'num_sentences']
        plot_feature_distributions(
            data, 
            features=features_to_plot,
            save_path=output_dir / "feature_distributions.png"
        )
        
        print("✓ Visualizations saved")
    else:
        print("\n[3/4] Skipping visualizations (use --visualize to enable)")
    
    # Step 2: Model Training
    print("\n" + "=" * 80)
    print("STEP 2: MODEL TRAINING AND EVALUATION")
    print("=" * 80)
    
    trainer = ModelTrainer(
        test_size=args.test_size,
        random_state=config.random_state
    )
    
    print("\n[4/4] Training and evaluating models...")
    print("-" * 80)
    
    results = trainer.complete_training_pipeline(
        X=X,
        y=y,
        vectorizer=preprocessor.vectorizer,
        scaler=preprocessor.scaler,
        model_path=output_dir / "model.pkl",
        vectorizer_path=output_dir / "vectorizer.pkl",
        scaler_path=output_dir / "scaler.pkl" if not args.no_scaling else None
    )
    
    # Display results
    print("\n" + "=" * 80)
    print("TRAINING RESULTS")
    print("=" * 80)
    
    print("\nPerformance of All Models:")
    print("-" * 80)
    print(results['performance_df'].to_string(index=False))
    
    print("\n" + "=" * 80)
    print(f"BEST MODEL: {results['best_model_name']}")
    print("=" * 80)
    
    print("\nBest Model Metrics:")
    for metric, value in results['best_metrics'].items():
        print(f"  {metric.capitalize()}: {value:.4f}")
    
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
    
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Visualize results
    if args.visualize:
        print("\nGenerating performance visualizations...")
        
        # Model comparison
        plot_model_comparison(
            results['performance_df'],
            save_path=output_dir / "model_comparison.png"
        )
        
        # Confusion matrix
        plot_confusion_matrix(
            results['confusion_matrix'],
            save_path=output_dir / "confusion_matrix.png"
        )
        
        print("✓ Performance visualizations saved")
    
    # Save results to file
    results_file = output_dir / "training_results.txt"
    save_results_to_file(results, results_file)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nModels and results saved to: {output_dir}")
    print(f"  • model.pkl")
    print(f"  • vectorizer.pkl")
    if not args.no_scaling:
        print(f"  • scaler.pkl")
    print(f"  • training_results.txt")
    if args.visualize:
        print(f"  • Visualization images")
    
    print("\nYou can now run the web application:")
    print("  python -m streamlit run app.py")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
