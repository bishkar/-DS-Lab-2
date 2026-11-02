"""
Task 3: Comprehensive Data Scaling and Normalization Study

This module implements various data scaling and normalization techniques,
analyzes their impact on machine learning algorithms, and provides
comprehensive visualizations of scaling effects.

Author: MLDS Course
Date: December 2025
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, RobustScaler, 
    MaxAbsScaler, PowerTransformer
)
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.datasets import make_classification
from typing import Tuple, Dict, List
import warnings
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings('ignore')


class ScalingAnalyzer:
    """
    A comprehensive scaling analyzer that implements various scaling techniques
    and evaluates their impact on machine learning models.
    
    This class provides 5 scaling methods (MinMax, Standard, Robust, MaxAbs, Power Transform)
    and evaluates their effects on different ML algorithms.
    
    Attributes:
        data (pd.DataFrame): Current working copy of the dataset
        original_data (pd.DataFrame): Original unmodified dataset
        target_col (str): Name of target column (excluded from scaling)
        scaled_datasets (Dict[str, pd.DataFrame]): Datasets after different scaling methods
        scalers (Dict[str, object]): Fitted scaler objects for each method
    """
    
    def __init__(self, data: pd.DataFrame, target_col: str = None):
        """
        Initialize the analyzer with a dataset.
        
        Parameters:
            data (pd.DataFrame): Input DataFrame to analyze
            target_col (str, optional): Name of target column to exclude from scaling. Defaults to None.
        """
        self.data = data.copy()
        self.original_data = data.copy()
        self.target_col = target_col
        self.scaled_datasets = {}
        self.scalers = {}
        
    def apply_minmax_scaling(self, feature_range: Tuple[float, float] = (0, 1)) -> pd.DataFrame:
        """
        Apply Min-Max scaling to rescale features to a fixed range.
        
        Parameters:
            feature_range (Tuple[float, float], optional): Desired range of transformed data. Defaults to (0, 1).
            
        Returns:
            pd.DataFrame: DataFrame with features scaled to specified range.
        """
        pass
    
    def apply_standard_scaling(self) -> pd.DataFrame:
        """
        Apply Z-score normalization (standardization) to have mean=0 and std=1.
        
        Returns:
            pd.DataFrame: DataFrame with standardized features.
        """
        pass
    
    def apply_robust_scaling(self) -> pd.DataFrame:
        """
        Apply Robust scaling using median and IQR (effective for outliers).
        
        Returns:
            pd.DataFrame: DataFrame with robust scaled features.
        """
        pass
    
    def apply_maxabs_scaling(self) -> pd.DataFrame:
        """
        Apply MaxAbs scaling to scale by maximum absolute value.
        
        Returns:
            pd.DataFrame: DataFrame with maxabs scaled features.
        """
        pass
    
    def apply_power_transform(self, method: str = 'yeo-johnson') -> pd.DataFrame:
        """
        Apply power transformation to make data more Gaussian-like.
        
        Parameters:
            method (str, optional): Transformation method ('yeo-johnson' or 'box-cox'). Defaults to 'yeo-johnson'.
            
        Returns:
            pd.DataFrame: DataFrame with power transformed and standardized features.
        """
        pass
    
    def get_scaling_summary(self) -> pd.DataFrame:
        """
        Get summary statistics for each scaling method.
        
        Returns:
            DataFrame with statistics for each scaling method
        """
        pass
    
    def evaluate_model_performance(self, model_type: str = 'linear_regression',
                                   dataset_name: str = 'original') -> Dict[str, float]:
        """
        Evaluate model performance with different scaling methods.
        
        Parameters:
            model_type (str, optional): Type of model to evaluate ('linear_regression', 'knn', 'svm', 'neural_network'). Defaults to 'linear_regression'.
            dataset_name (str, optional): Dataset to evaluate ('original' or scaled version name). Defaults to 'original'.
            
        Returns:
            Dict[str, float]: Dictionary containing cv_score and cv_std.
        """
        pass
    
    def compare_model_performance(self, model_types: List[str] = None) -> pd.DataFrame:
        """
        Compare performance across different scaling methods and models.
        
        Parameters:
            model_types (List[str], optional): List of model types to evaluate. Defaults to ['linear_regression', 'knn'].
            
        Returns:
            pd.DataFrame: Comparison table with model, scaling, score, and std columns.
        """
        if model_types is None:
            model_types = ['linear_regression', 'knn']
        
        results = []
        datasets = ['original'] + list(self.scaled_datasets.keys())
        
        for model_type in model_types:
            for dataset_name in datasets:
                perf = self.evaluate_model_performance(model_type, dataset_name)
                results.append({
                    'model': model_type,
                    'scaling': dataset_name,
                    'score': perf['cv_score'],
                    'std': perf['cv_std']
                })
        
        return pd.DataFrame(results)
    
    def visualize_distributions(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Compare distributions after different scaling methods using histograms.
        
        Parameters:
            figsize (Tuple[int, int], optional): Figure size in inches. Defaults to (15, 10).
        
        Returns:
            matplotlib.figure.Figure: The generated figure object.
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if self.target_col:
            numeric_cols = [col for col in numeric_cols if col != self.target_col]
        
        if len(numeric_cols) == 0:
            print("No numeric columns to visualize")
            return
        
        # Select first feature for demonstration
        feature = numeric_cols[0]
        
        n_methods = len(self.scaled_datasets) + 1
        n_cols = 3
        n_rows = (n_methods + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_methods > 1 else [axes]
        
        # Original data
        ax = axes[0]
        self.original_data[feature].hist(bins=30, ax=ax, color='skyblue', 
                                        edgecolor='black', alpha=0.7)
        ax.axvline(self.original_data[feature].mean(), color='red', 
                  linestyle='--', linewidth=2, label='Mean')
        ax.set_title('Original Data', fontweight='bold')
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Scaled datasets
        for idx, (method_name, scaled_df) in enumerate(self.scaled_datasets.items(), start=1):
            ax = axes[idx]
            scaled_df[feature].hist(bins=30, ax=ax, color='lightcoral', 
                                   edgecolor='black', alpha=0.7)
            ax.axvline(scaled_df[feature].mean(), color='red', 
                      linestyle='--', linewidth=2, label='Mean')
            ax.set_title(f'{method_name.upper()} Scaling', fontweight='bold')
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_methods, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Distribution Comparison: {feature}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    def visualize_boxplots(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Show range and spread of scaled features using box plots.
        
        Args:
            figsize: Figure size for plots
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if self.target_col:
            numeric_cols = [col for col in numeric_cols if col != self.target_col]
        
        if len(numeric_cols) == 0:
            print("No numeric columns to visualize")
            return
        
        n_features = min(6, len(numeric_cols))
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for idx, feature in enumerate(numeric_cols[:n_features]):
            ax = axes[idx]
            
            data_to_plot = [self.original_data[feature].dropna()]
            labels = ['Original']
            
            for method_name, scaled_df in self.scaled_datasets.items():
                data_to_plot.append(scaled_df[feature].dropna())
                labels.append(method_name.upper())
            
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            # Color the boxes
            colors = ['skyblue'] + ['lightcoral'] * len(self.scaled_datasets)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_title(f'{feature}', fontweight='bold')
            ax.set_ylabel('Value')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Box Plots: Range and Spread Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    def visualize_correlation_comparison(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Visualize correlation structure under different scaling methods.
        
        Args:
            figsize: Figure size for plots
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if self.target_col:
            numeric_cols = [col for col in numeric_cols if col != self.target_col]
        
        if len(numeric_cols) < 2:
            print("Need at least 2 numeric columns for correlation comparison")
            return
        
        n_methods = len(self.scaled_datasets) + 1
        n_cols = 3
        n_rows = (n_methods + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_methods > 1 else [axes]
        
        # Original correlation
        ax = axes[0]
        corr_original = self.original_data[numeric_cols].corr()
        sns.heatmap(corr_original, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Original Data', fontweight='bold')
        
        # Scaled correlations
        for idx, (method_name, scaled_df) in enumerate(self.scaled_datasets.items(), start=1):
            ax = axes[idx]
            corr_scaled = scaled_df[numeric_cols].corr()
            sns.heatmap(corr_scaled, annot=True, fmt='.2f', cmap='coolwarm',
                       center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
            ax.set_title(f'{method_name.upper()} Scaling', fontweight='bold')
        
        # Hide unused subplots
        for idx in range(n_methods, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Correlation Structure Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    def visualize_performance_comparison(self, comparison_df: pd.DataFrame,
                                        figsize: Tuple[int, int] = (12, 6)):
        """
        Visualize model performance comparison using radar chart.
        
        Args:
            comparison_df: DataFrame with performance comparison
            figsize: Figure size for plots
        """
        if comparison_df.empty:
            print("No comparison data available")
            return
        
        # Group by scaling method and average across models
        scaling_perf = comparison_df.groupby('scaling')['score'].mean()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Bar chart
        scaling_perf.plot(kind='bar', ax=ax1, color='steelblue', alpha=0.7)
        ax1.set_title('Average Performance by Scaling Method', fontweight='bold')
        ax1.set_xlabel('Scaling Method')
        ax1.set_ylabel('Average Score')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Grouped bar chart by model
        pivot_df = comparison_df.pivot(index='scaling', columns='model', values='score')
        pivot_df.plot(kind='bar', ax=ax2, alpha=0.7)
        ax2.set_title('Performance by Scaling Method and Model', fontweight='bold')
        ax2.set_xlabel('Scaling Method')
        ax2.set_ylabel('Score')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def visualize_feature_space_2d(self, x_feature: str, y_feature: str,
                                   figsize: Tuple[int, int] = (15, 5)):
        """
        Visualize how scaling transforms the 2D feature space.
        
        Args:
            x_feature: Feature for x-axis
            y_feature: Feature for y-axis
            figsize: Figure size for plots
        """
        if x_feature not in self.data.columns or y_feature not in self.data.columns:
            print(f"Features '{x_feature}' or '{y_feature}' not found")
            return
        
        n_methods = min(3, len(self.scaled_datasets) + 1)
        fig, axes = plt.subplots(1, n_methods, figsize=figsize)
        if n_methods == 1:
            axes = [axes]
        
        # Original data
        ax = axes[0]
        if self.target_col and self.target_col in self.data.columns:
            scatter = ax.scatter(self.original_data[x_feature], 
                               self.original_data[y_feature],
                               c=self.original_data[self.target_col],
                               cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(self.original_data[x_feature], 
                      self.original_data[y_feature],
                      alpha=0.6, color='blue')
        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        ax.set_title('Original Data', fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Scaled data
        for idx, (method_name, scaled_df) in enumerate(list(self.scaled_datasets.items())[:n_methods-1], start=1):
            ax = axes[idx]
            if self.target_col and self.target_col in scaled_df.columns:
                scatter = ax.scatter(scaled_df[x_feature], 
                                   scaled_df[y_feature],
                                   c=scaled_df[self.target_col],
                                   cmap='viridis', alpha=0.6)
                plt.colorbar(scatter, ax=ax)
            else:
                ax.scatter(scaled_df[x_feature], 
                          scaled_df[y_feature],
                          alpha=0.6, color='coral')
            ax.set_xlabel(x_feature)
            ax.set_ylabel(y_feature)
            ax.set_title(f'{method_name.upper()} Scaling', fontweight='bold')
            ax.grid(alpha=0.3)
        
        plt.suptitle(f'Feature Space Transformation: {x_feature} vs {y_feature}',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    def visualize_feature_space_3d(self, x_feature: str, y_feature: str, z_feature: str,
                                   figsize: Tuple[int, int] = (15, 5)):
        """
        Visualize how scaling transforms the 3D feature space.
        
        Args:
            x_feature, y_feature, z_feature: Features for 3D axes
            figsize: Figure size for plots
        """
        fig = plt.figure(figsize=figsize)
        
        n_methods = min(3, len(self.scaled_datasets) + 1)
        
        # Original data
        ax = fig.add_subplot(1, n_methods, 1, projection='3d')
        if self.target_col and self.target_col in self.data.columns:
            scatter = ax.scatter(self.original_data[x_feature],
                               self.original_data[y_feature],
                               self.original_data[z_feature],
                               c=self.original_data[self.target_col],
                               cmap='viridis', alpha=0.6)
        else:
            ax.scatter(self.original_data[x_feature],
                      self.original_data[y_feature],
                      self.original_data[z_feature],
                      alpha=0.6, color='blue')
        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        ax.set_zlabel(z_feature)
        ax.set_title('Original Data', fontweight='bold')
        
        # Scaled data
        for idx, (method_name, scaled_df) in enumerate(list(self.scaled_datasets.items())[:n_methods-1], start=2):
            ax = fig.add_subplot(1, n_methods, idx, projection='3d')
            if self.target_col and self.target_col in scaled_df.columns:
                scatter = ax.scatter(scaled_df[x_feature],
                                   scaled_df[y_feature],
                                   scaled_df[z_feature],
                                   c=scaled_df[self.target_col],
                                   cmap='viridis', alpha=0.6)
            else:
                ax.scatter(scaled_df[x_feature],
                          scaled_df[y_feature],
                          scaled_df[z_feature],
                          alpha=0.6, color='coral')
            ax.set_xlabel(x_feature)
            ax.set_ylabel(y_feature)
            ax.set_zlabel(z_feature)
            ax.set_title(f'{method_name.upper()}', fontweight='bold')
        
        plt.suptitle('3D Feature Space Transformation', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        return fig


def create_sample_dataset() -> Tuple[pd.DataFrame, str]:
    """
    Create a sample dataset for scaling demonstration.
    
    Returns:
        Tuple[pd.DataFrame, str]: Dataset and target column name.
    """
    np.random.seed(42)
    n_samples = 200
    
    # Create features with different scales
    data = {
        'age': np.random.uniform(18, 80, n_samples),
        'income': np.random.lognormal(10, 1, n_samples),  # Large scale
        'credit_score': np.random.normal(700, 50, n_samples),
        'loan_amount': np.random.uniform(1000, 50000, n_samples),
        'debt_ratio': np.random.beta(2, 5, n_samples),  # Small scale (0-1)
        'years_employed': np.random.exponential(5, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable (binary classification)
    df['approved'] = (
        (df['income'] > 30000) & 
        (df['credit_score'] > 680) & 
        (df['debt_ratio'] < 0.4)
    ).astype(int)
    
    # Add some noise
    noise_indices = np.random.choice(df.index, size=30, replace=False)
    df.loc[noise_indices, 'approved'] = 1 - df.loc[noise_indices, 'approved']
    
    return df, 'approved'


def main():
    """
    Main function demonstrating the complete Task 3 workflow.
    
    Returns:
        ScalingAnalyzer: Configured analyzer object with all scalings performed.
    """
    print("=" * 80)
    print("Task 3: Comprehensive Data Scaling and Normalization Study")
    print("=" * 80)
    print()
    
    # Create sample dataset
    print("Creating sample dataset with features at different scales...")
    df, target_col = create_sample_dataset()
    print(f"Dataset created: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Target column: {target_col}")
    print()
    
    # Show original scale ranges
    print("Original feature ranges:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col != target_col]
    for col in feature_cols:
        print(f"  {col}: [{df[col].min():.2f}, {df[col].max():.2f}]")
    print()
    
    # Initialize analyzer
    analyzer = ScalingAnalyzer(df, target_col=target_col)
    
    # Apply different scaling methods
    print("1. APPLYING SCALING METHODS")
    print("-" * 80)
    
    print("  a) Min-Max Scaling...")
    analyzer.apply_minmax_scaling(feature_range=(0, 1))
    
    print("  b) Z-Score Normalization (Standardization)...")
    analyzer.apply_standard_scaling()
    
    print("  c) Robust Scaling...")
    analyzer.apply_robust_scaling()
    
    print("  d) MaxAbs Scaling...")
    analyzer.apply_maxabs_scaling()
    
    print("  e) Power Transformation (Yeo-Johnson)...")
    analyzer.apply_power_transform()
    print()
    
    # Get scaling summary
    print("2. SCALING SUMMARY STATISTICS")
    print("-" * 80)
    summary = analyzer.get_scaling_summary()
    print(summary.to_string())
    print()
    
    # Compare model performance
    print("3. MODEL PERFORMANCE COMPARISON")
    print("-" * 80)
    
    print("  Evaluating models with different scaling methods...")
    comparison = analyzer.compare_model_performance(model_types=['linear_regression', 'knn'])
    print(comparison.to_string(index=False))
    print()
    
    # Generate visualizations
    print("4. GENERATING VISUALIZATIONS")
    print("-" * 80)
    
    print("  Creating distribution comparison plots...")
    analyzer.visualize_distributions()
    plt.savefig('task3_distributions.png', dpi=300, bbox_inches='tight')
    print("  Saved: task3_distributions.png")
    
    print("  Creating box plots...")
    analyzer.visualize_boxplots()
    plt.savefig('task3_boxplots.png', dpi=300, bbox_inches='tight')
    print("  Saved: task3_boxplots.png")
    
    print("  Creating correlation comparison...")
    analyzer.visualize_correlation_comparison()
    plt.savefig('task3_correlations.png', dpi=300, bbox_inches='tight')
    print("  Saved: task3_correlations.png")
    
    print("  Creating performance comparison...")
    analyzer.visualize_performance_comparison(comparison)
    plt.savefig('task3_performance.png', dpi=300, bbox_inches='tight')
    print("  Saved: task3_performance.png")
    
    print("  Creating 2D feature space visualization...")
    analyzer.visualize_feature_space_2d('age', 'income')
    plt.savefig('task3_feature_space_2d.png', dpi=300, bbox_inches='tight')
    print("  Saved: task3_feature_space_2d.png")
    
    print("  Creating 3D feature space visualization...")
    analyzer.visualize_feature_space_3d('age', 'income', 'credit_score')
    plt.savefig('task3_feature_space_3d.png', dpi=300, bbox_inches='tight')
    print("  Saved: task3_feature_space_3d.png")
    print()
    
    print("=" * 80)
    print("Task 3 completed successfully!")
    print("=" * 80)
    print()
    print("RECOMMENDATIONS:")
    print("  - Use Min-Max Scaling for neural networks and algorithms sensitive to feature ranges")
    print("  - Use Standard Scaling for algorithms assuming normally distributed data (SVM, LR)")
    print("  - Use Robust Scaling when data contains outliers")
    print("  - Use Power Transform to make data more Gaussian-like")
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()
    plt.show()
