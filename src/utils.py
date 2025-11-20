"""
Utility functions for Email Spam Detection System.
Provides helper functions for visualization, data analysis, and logging.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from collections import Counter
from wordcloud import WordCloud
from pathlib import Path


def plot_class_distribution(data: pd.DataFrame, target_col: str = 'target',
                            figsize: Tuple[int, int] = (8, 6),
                            save_path: Optional[Path] = None):
    """
    Plot the distribution of target classes.
    
    Args:
        data: DataFrame with target column
        target_col: Name of the target column
        figsize: Figure size
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=figsize)
    
    # Count plot
    sns.countplot(data=data, x=target_col, palette='Set2')
    plt.title('Distribution of Spam vs Ham Messages', fontsize=14)
    plt.xlabel('Class (0: Ham, 1: Spam)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Add counts on bars
    ax = plt.gca()
    for container in ax.containers:
        ax.bar_label(container)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_distributions(data: pd.DataFrame, features: List[str],
                               target_col: str = 'target',
                               figsize: Tuple[int, int] = (15, 10),
                               save_path: Optional[Path] = None):
    """
    Plot distributions of multiple features by target class.
    
    Args:
        data: DataFrame with features and target
        features: List of feature column names
        target_col: Name of the target column
        figsize: Figure size
        save_path: Path to save the plot (optional)
    """
    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for idx, feature in enumerate(features):
        sns.histplot(data=data, x=feature, hue=target_col, 
                    bins=30, ax=axes[idx], palette='Set2', kde=True)
        axes[idx].set_title(f'Distribution of {feature}', fontsize=12)
        axes[idx].set_xlabel(feature, fontsize=10)
        axes[idx].set_ylabel('Count', fontsize=10)
    
    # Remove extra subplots
    for idx in range(n_features, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def generate_wordcloud(corpus: List[str], title: str = 'Word Cloud',
                      width: int = 800, height: int = 400,
                      max_words: int = 100, background_color: str = 'white',
                      figsize: Tuple[int, int] = (12, 6),
                      save_path: Optional[Path] = None):
    """
    Generate and display a word cloud from a corpus of text.
    
    Args:
        corpus: List of words or text
        title: Title for the plot
        width: Width of the word cloud
        height: Height of the word cloud
        max_words: Maximum number of words to display
        background_color: Background color
        figsize: Figure size
        save_path: Path to save the plot (optional)
    """
    # Combine corpus into single string
    text = ' '.join(corpus)
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=width,
        height=height,
        max_words=max_words,
        background_color=background_color,
        colormap='viridis'
    ).generate(text)
    
    # Plot
    plt.figure(figsize=figsize)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_top_words(corpus: List[str], n_words: int = 30,
                  title: str = 'Most Common Words',
                  figsize: Tuple[int, int] = (15, 6),
                  save_path: Optional[Path] = None):
    """
    Plot the most common words in a corpus.
    
    Args:
        corpus: List of words
        n_words: Number of top words to display
        title: Title for the plot
        figsize: Figure size
        save_path: Path to save the plot (optional)
    """
    # Count word frequencies
    counter = Counter(corpus)
    most_common = counter.most_common(n_words)
    
    # Extract words and counts
    words = [item[0] for item in most_common]
    counts = [item[1] for item in most_common]
    
    # Plot
    plt.figure(figsize=figsize)
    sns.barplot(x=words, y=counts, palette='viridis')
    plt.title(title, fontsize=16)
    plt.xlabel('Words', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, labels: List[str] = ['Ham', 'Spam'],
                         figsize: Tuple[int, int] = (8, 6),
                         save_path: Optional[Path] = None):
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        cm: Confusion matrix
        labels: Class labels
        figsize: Figure size
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=figsize)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_model_comparison(performance_df: pd.DataFrame,
                         metrics: List[str] = ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                         figsize: Tuple[int, int] = (14, 10),
                         save_path: Optional[Path] = None):
    """
    Plot comparison of multiple models across different metrics.
    
    Args:
        performance_df: DataFrame with model performance metrics
        metrics: List of metrics to plot
        figsize: Figure size
        save_path: Path to save the plot (optional)
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        if metric in performance_df.columns:
            sns.barplot(data=performance_df, x='Model', y=metric, 
                       ax=axes[idx], palette='viridis')
            axes[idx].set_title(f'{metric} Comparison', fontsize=14)
            axes[idx].set_ylabel(metric, fontsize=12)
            axes[idx].set_xlabel('Model' if idx == n_metrics - 1 else '', fontsize=12)
            axes[idx].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for container in axes[idx].containers:
                axes[idx].bar_label(container, fmt='%.3f')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_performance_comparison_multi(performance_dfs: Dict[str, pd.DataFrame],
                                     metric: str = 'Accuracy',
                                     figsize: Tuple[int, int] = (14, 8),
                                     save_path: Optional[Path] = None):
    """
    Plot comparison of model performance across different configurations.
    
    Args:
        performance_dfs: Dictionary of {config_name: performance_df}
        metric: Metric to compare
        figsize: Figure size
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=figsize)
    
    # Get model names from first dataframe
    first_df = list(performance_dfs.values())[0]
    models = first_df['Model'].values
    
    # Plot each configuration
    x = np.arange(len(models))
    width = 0.8 / len(performance_dfs)
    
    for idx, (config_name, df) in enumerate(performance_dfs.items()):
        if metric in df.columns:
            offset = width * idx - (width * len(performance_dfs) / 2) + width / 2
            plt.bar(x + offset, df[metric].values, width, label=config_name)
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.title(f'{metric} Comparison Across Configurations', fontsize=14)
    plt.xticks(x, models, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def get_data_statistics(data: pd.DataFrame, target_col: str = 'target') -> Dict:
    """
    Get comprehensive statistics about the dataset.
    
    Args:
        data: Input DataFrame
        target_col: Name of the target column
        
    Returns:
        Dictionary with various statistics
    """
    stats = {
        'total_samples': len(data),
        'num_features': len(data.columns) - 1,
        'class_distribution': data[target_col].value_counts().to_dict(),
        'missing_values': data.isnull().sum().to_dict(),
        'duplicate_rows': data.duplicated().sum()
    }
    
    # Add class percentages
    total = len(data)
    stats['class_percentages'] = {
        k: f"{(v/total)*100:.2f}%" 
        for k, v in stats['class_distribution'].items()
    }
    
    return stats


def print_data_summary(data: pd.DataFrame, target_col: str = 'target'):
    """
    Print a formatted summary of the dataset.
    
    Args:
        data: Input DataFrame
        target_col: Name of the target column
    """
    stats = get_data_statistics(data, target_col)
    
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total Samples: {stats['total_samples']}")
    print(f"Number of Features: {stats['num_features']}")
    print(f"Duplicate Rows: {stats['duplicate_rows']}")
    print("\nClass Distribution:")
    for cls, count in stats['class_distribution'].items():
        percentage = stats['class_percentages'][cls]
        label = "Ham" if cls == 0 else "Spam"
        print(f"  {label} ({cls}): {count} ({percentage})")
    print("\nMissing Values:")
    missing_total = sum(stats['missing_values'].values())
    if missing_total == 0:
        print("  No missing values found")
    else:
        for col, count in stats['missing_values'].items():
            if count > 0:
                print(f"  {col}: {count}")
    print("=" * 60)


def save_results_to_file(results: Dict, filepath: Path):
    """
    Save training results to a text file.
    
    Args:
        results: Dictionary with training results
        filepath: Path to save the file
    """
    with open(filepath, 'w') as f:
        f.write("EMAIL SPAM DETECTION - TRAINING RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Best Model: {results.get('best_model_name', 'N/A')}\n\n")
        
        if 'best_metrics' in results:
            f.write("Best Model Metrics:\n")
            for metric, value in results['best_metrics'].items():
                f.write(f"  {metric.capitalize()}: {value:.4f}\n")
            f.write("\n")
        
        if 'performance_df' in results:
            f.write("All Models Performance:\n")
            f.write(results['performance_df'].to_string())
            f.write("\n\n")
        
        if 'classification_report' in results:
            f.write("Classification Report:\n")
            f.write(results['classification_report'])
            f.write("\n")
    
    print(f"Results saved to {filepath}")
