"""
Task 2: Outlier Detection and Treatment with Advanced Visualization

This module implements various outlier detection and treatment techniques,
including statistical methods, machine learning-based detection, and 
comprehensive visualization of outliers and their impact.

Author: MLDS Course
Date: December 2025
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from scipy import stats
from typing import Callable, Tuple, Dict, List, Literal
import warnings

from pydantic import NonNegativeInt, PositiveFloat, PositiveInt

from common import (DEFAULT_K, Contamination, OutlierMethod, Schema, clip_to_bounds, lookup_dataset,
                    median_filled, numeric_columns, outlier_bounds, outside, shift_non_negative,
                    validated)

warnings.filterwarnings('ignore')


DetectionMethod = Literal['zscore', 'iqr', 'isolation_forest', 'lof']
SCATTER_METHODS = ('zscore', 'iqr', 'isolation_forest')


class OutlierStats(Schema):
    outlier_values: NonNegativeInt
    outlier_rows: NonNegativeInt
    outlier_rows_pct: float


class RegressionPerformance(Schema):
    r2_score: float
    r2_std: float
    mean_cv_score: float
    cv_scores: List[float]
    n_samples: PositiveInt


class OutlierAnalyzer:
    """
    A comprehensive outlier analyzer that detects and treats outliers using
    multiple methods and provides detailed visualizations.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the analyzer with a dataset.
        
        Parameters:
            data (pd.DataFrame): Input DataFrame to analyze
        """
        self.data = data.copy()
        self.original_data = data.copy()
        self.outlier_masks = {}
        self.treated_datasets = {}
        
    @validated
    def detect_outliers_zscore(self, threshold: PositiveFloat = 3.0)-> pd.DataFrame:
        """
        Detect outliers using Z-score method.
        
        Parameters:
            threshold (float): Z-score threshold for outlier detection
            
        Returns:
            pd.DataFrame: Boolean mask indicating outliers
        """

        return self._store_mask('zscore', self._bounds_mask('zscore', threshold))

    def _bounds_mask(self, method: OutlierMethod, k: float) -> pd.DataFrame:
        return pd.DataFrame(
            {col: outside(self.data[col], outlier_bounds(self.data[col], method, k))
             for col in numeric_columns(self.data)},
            index=self.data.index, dtype=bool,
        )

    def _model_mask(self, model, min_rows: int) -> pd.DataFrame:
        X = median_filled(self.data[numeric_columns(self.data)])
        flags = np.zeros(len(X), dtype=bool)
        if X.shape[0] >= min_rows and X.shape[1] > 0:
            flags = model.fit_predict(X) == -1
        return pd.DataFrame({'outlier': flags}, index=self.data.index)

    def _store_mask(self, name: str, mask: pd.DataFrame) -> pd.DataFrame:
        self.outlier_masks[name] = mask
        return mask

    def _detectors(self) -> Dict[str, Callable[[], pd.DataFrame]]:
        return {
            'zscore': self.detect_outliers_zscore,
            'iqr': self.detect_outliers_iqr,
            'isolation_forest': self.detect_outliers_isolation_forest,
            'lof': self.detect_outliers_lof,
        }

    @validated
    def detect_outliers_iqr(self, multiplier: PositiveFloat = 1.5)-> pd.DataFrame:
        """
        Detect outliers using Interquartile Range (IQR) method.
        
        Parameters:
            multiplier (float): IQR multiplier for boundary calculation
            
        Returns:
            pd.DataFrame: Boolean mask indicating outliers
        """

        return self._store_mask('iqr', self._bounds_mask('iqr', multiplier))

    @validated
    def detect_outliers_isolation_forest(self, contamination: Contamination = 0.1,
                                        random_state: int = 42) -> pd.DataFrame:
        """
        Detect outliers using Isolation Forest algorithm.
        
        Parameters:
            contamination (float): Expected proportion of outliers in the dataset
            random_state (int): Random seed for reproducibility
            
        Returns:
            pd.DataFrame: Boolean mask indicating outliers
        """

        model = IsolationForest(contamination=contamination, random_state=random_state)
        return self._store_mask('isolation_forest', self._model_mask(model, min_rows=2))

    @validated
    def detect_outliers_lof(self, n_neighbors: PositiveInt = 20,
                           contamination: Contamination = 0.1)-> pd.DataFrame:
        """
        Detect outliers using Local Outlier Factor (LOF) algorithm.
        
        Parameters:
            n_neighbors (int): Number of neighbors to consider
            contamination (float): Expected proportion of outliers
            
        Returns:
            pd.DataFrame: Boolean mask indicating outliers
        """

        model = LocalOutlierFactor(n_neighbors=max(1, min(n_neighbors, len(self.data) - 1)),
                                   contamination=contamination)
        return self._store_mask('lof', self._model_mask(model, min_rows=3))

    def get_outlier_summary(self) -> pd.DataFrame:
        """
        Get summary statistics of outliers detected by different methods.
        
        Returns:
            DataFrame with outlier counts for each method
        """

        n_rows = len(self.data)

        def stats(mask: pd.DataFrame) -> dict:
            rows = int(mask.any(axis=1).sum())
            return OutlierStats(
                outlier_values=int(mask.values.sum()),
                outlier_rows=rows,
                outlier_rows_pct=rows / n_rows * 100 if n_rows else 0.0,
            ).model_dump()

        return pd.DataFrame({method: stats(mask) for method, mask in self.outlier_masks.items()})

    @validated
    def remove_outliers(self, method: DetectionMethod = 'iqr')-> pd.DataFrame:
        """
        Remove outliers from the dataset.
        
        Parameters:
            method (str): Detection method to use
            
        Returns:
            pd.DataFrame: Dataset with outliers removed
        """

        if method not in self.outlier_masks:
            self._detectors()[method]()
        cleaned = self.data.loc[~self.outlier_masks[method].any(axis=1)].copy()
        return self._store_treated(f'removed_{method}', cleaned)

    def _store_treated(self, name: str, data: pd.DataFrame) -> pd.DataFrame:
        self.treated_datasets[name] = data
        return data

    @validated
    def cap_outliers(self, method: OutlierMethod = 'iqr', multiplier: PositiveFloat = 1.5)-> pd.DataFrame:
        """
        Cap outliers using winsorization.
        
        Parameters:
            method (str): Detection method to use
            multiplier (float): IQR multiplier for bounds
            
        Returns:
            pd.DataFrame: Dataset with capped outliers
        """

        k = multiplier if method == 'iqr' else DEFAULT_K['zscore']
        capped = self.data.copy()
        for col in numeric_columns(capped):
            capped[col] = clip_to_bounds(capped[col], outlier_bounds(capped[col], method, k))
        return self._store_treated('capped', capped)

    def transform_outliers_log(self) -> pd.DataFrame:
        """
        Transform data using log transformation.
        
        Returns:
            pd.DataFrame: Dataset with log-transformed values
        """

        transformed = self.data.copy()
        for col in numeric_columns(transformed):
            transformed[col] = np.log1p(shift_non_negative(transformed[col].astype(float)))
        return self._store_treated('log_transform', transformed)

    def transform_outliers_boxcox(self) -> pd.DataFrame:
        """
        Transform data using Box-Cox/Yeo-Johnson transformation.
        
        Returns:
            pd.DataFrame: Dataset with power-transformed values
        """

        transformed = self.data.copy()
        varying = [col for col in numeric_columns(transformed) if transformed[col].nunique() > 1]
        if varying:
            transformer = PowerTransformer(method='yeo-johnson', standardize=False)
            transformed[varying] = transformer.fit_transform(transformed[varying].astype(float))
        return self._store_treated('boxcox_transform', transformed)

    def evaluate_model_performance(self, target_col: str, 
                                   dataset_name: str = 'original') -> Dict[str, float]:
        """
        Evaluate model performance with and without outliers.
        
        Args:
            target_col: Name of target column
            dataset_name: Dataset to evaluate ('original' or treated dataset name)
            
        Returns:
            Dictionary with performance metrics
        """

        df = lookup_dataset(dataset_name, self.data, self.treated_datasets)
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        numeric = df[numeric_columns(df)].dropna()
        X, y = numeric.drop(columns=[target_col]), numeric[target_col]
        if X.shape[1] == 0 or len(X) < 4:
            raise ValueError("Not enough data/features to evaluate model")

        scores = cross_val_score(LinearRegression(), X, y, cv=min(5, len(X)), scoring='r2')
        return RegressionPerformance(
            r2_score=scores.mean(),
            r2_std=scores.std(),
            mean_cv_score=scores.mean(),
            cv_scores=scores.tolist(),
            n_samples=len(X),
        ).model_dump()

    def visualize_outliers_boxplot(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Create box plots showing outliers.
        
        Parameters:
            figsize (Tuple[int, int]): Figure size for plots
            
        Returns:
            matplotlib.figure.Figure: The generated figure object
        """

        return self._feature_grid(numeric_columns(self.data)[:6], figsize,
                                  'Box Plots - Outlier Detection (IQR)', self._draw_box)

    def _feature_grid(self, columns: List[str], figsize: Tuple[int, int], title: str,
                      draw: Callable) -> plt.Figure:
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        for ax, col in zip(axes, columns):
            draw(ax, col, self.data[col].dropna())
            ax.set_ylabel('Value')
            ax.grid(axis='y', alpha=0.3)
        for ax in axes[len(columns):]:
            ax.axis('off')
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig

    @staticmethod
    def _draw_box(ax, col: str, values: pd.Series):
        ax.boxplot(values, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7),
                   flierprops=dict(marker='o', markerfacecolor='red', markersize=5, alpha=0.6))
        n_outliers = int(outside(values, outlier_bounds(values, 'iqr', DEFAULT_K['iqr'])).sum())
        ax.set_title(f'{col} ({n_outliers} outliers)', fontweight='bold')

    @staticmethod
    def _draw_violin(ax, col: str, values: pd.Series):
        if values.nunique() > 1:
            parts = ax.violinplot(values, showmeans=True, showmedians=True)
            for body in parts['bodies']:
                body.set_facecolor('mediumpurple')
                body.set_alpha(0.6)
            parts['cmeans'].set_color('red')
            parts['cmedians'].set_color('green')
        ax.set_title(col, fontweight='bold')
        ax.set_xticks([])

    def visualize_outliers_violin(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Create violin plots showing distribution with outliers.
        
        Args:
            figsize: Figure size for plots
        """

        return self._feature_grid(numeric_columns(self.data)[:6], figsize,
                                  'Violin Plots (red = mean, green = median)', self._draw_violin)

    def visualize_outliers_scatter(self, x_col: str, y_col: str, 
                                   figsize: Tuple[int, int] = (15, 5)):
        """
        Create scatter plots showing outliers detected by different methods.
        
        Args:
            x_col: Column for x-axis
            y_col: Column for y-axis
            figsize: Figure size for plots
        """

        if not self._require_columns(x_col, y_col):
            return None
        masks = self._scatter_masks()
        fig, axes = plt.subplots(1, len(masks), figsize=figsize, squeeze=False)
        for ax, (method, mask) in zip(axes.flatten(), masks.items()):
            pair = [col for col in (x_col, y_col) if col in mask.columns] or list(mask.columns)
            self._plot_split(ax, mask[pair].any(axis=1), (x_col, y_col))
            ax.set_title(method.replace('_', ' ').title(), fontweight='bold')
            ax.grid(alpha=0.3)
        plt.suptitle(f'Outlier Detection: {x_col} vs {y_col}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig

    def _require_columns(self, *columns: str) -> bool:
        missing = [col for col in columns if col not in self.data.columns]
        if missing:
            print(f"Column '{missing[0]}' not found")
        return not missing

    def _scatter_masks(self) -> Dict[str, pd.DataFrame]:
        stored = {m: self.outlier_masks[m] for m in SCATTER_METHODS if m in self.outlier_masks}
        if stored:
            return stored
        saved = dict(self.outlier_masks)
        fresh = {m: self._detectors()[m]() for m in SCATTER_METHODS}
        self.outlier_masks = saved
        return fresh

    def _plot_split(self, ax, row_mask: pd.Series, columns: Tuple[str, ...]):
        groups = {
            'Normal': (self.data.loc[~row_mask], dict(c='steelblue', alpha=0.5, s=20)),
            f'Outlier ({int(row_mask.sum())})': (self.data.loc[row_mask],
                                                 dict(c='red', alpha=0.9, s=45, marker='x')),
        }
        for label, (subset, style) in groups.items():
            ax.scatter(*(subset[col] for col in columns), label=label, **style)
        for set_label, col in zip((ax.set_xlabel, ax.set_ylabel, getattr(ax, 'set_zlabel', None)), columns):
            set_label(col)
        ax.legend()

    def visualize_3d_outliers(self, x_col: str, y_col: str, z_col: str, 
                             method: str = 'isolation_forest',
                             figsize: Tuple[int, int] = (12, 8)):
        """
        Create three-dimensional scatter plot showing multivariate outliers.
        
        Args:
            x_col, y_col, z_col: Columns for three-dimensional axes
            method: Detection method to use
            figsize: Figure size for plot
        """

        if not self._require_columns(x_col, y_col, z_col):
            return None
        if method not in self.outlier_masks:
            print(f"Outlier mask for method '{method}' not found. Run detection first.")
            return None
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        self._plot_split(ax, self.outlier_masks[method].any(axis=1), (x_col, y_col, z_col))
        ax.set_title(f'3D Outliers ({method.replace("_", " ").title()})', fontweight='bold')
        plt.tight_layout()
        return fig

    def visualize_treatment_comparison(self, column: str, 
                                      figsize: Tuple[int, int] = (15, 10)):
        """
        Compare distributions before and after different treatments.
        
        Args:
            column: Column to visualize
            figsize: Figure size for plots
        """
        if column not in self.data.columns:
            print(f"Column '{column}' not found")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        # Original data
        ax = axes[0]
        self.original_data[column].hist(bins=30, ax=ax, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(self.original_data[column].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        ax.set_title('Original Data', fontweight='bold')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Treated datasets
        treatment_names = list(self.treated_datasets.keys())[:5]
        for idx, name in enumerate(treatment_names, start=1):
            ax = axes[idx]
            treated_data = self.treated_datasets[name]
            if column in treated_data.columns:
                treated_data[column].hist(bins=30, ax=ax, color='lightcoral', 
                                        edgecolor='black', alpha=0.7)
                ax.axvline(treated_data[column].mean(), color='red', 
                          linestyle='--', linewidth=2, label='Mean')
                ax.set_title(f'{name.replace("_", " ").title()}', fontweight='bold')
                ax.set_xlabel(column)
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(treatment_names) + 1, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Treatment Comparison: {column}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    def visualize_qq_plots(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Create Q-Q plots to assess normality before and after treatment.
        
        Args:
            figsize: Figure size for plots
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns[:6]
        
        if len(numeric_cols) == 0:
            print("No numeric columns to visualize")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            
            stats.probplot(self.data[col].dropna(), dist="norm", plot=ax)
            ax.set_title(f'Q-Q Plot: {col}', fontweight='bold')
            ax.grid(alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Q-Q Plots for Normality Assessment', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig


def create_sample_dataset_with_outliers() -> pd.DataFrame:
    """
    Create a sample dataset with outliers for demonstration.
    
    Returns:
        pd.DataFrame: Dataset with outliers
    """
    np.random.seed(42)
    n_samples = 300
    
    # Create base data
    data = {
        'feature1': np.random.normal(100, 15, n_samples),
        'feature2': np.random.exponential(50, n_samples),
        'feature3': np.random.normal(500, 100, n_samples),
        'feature4': np.random.uniform(0, 100, n_samples),
        'target': np.random.normal(200, 30, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add synthetic target relationship
    df['target'] = 0.5 * df['feature1'] + 0.3 * df['feature2'] + np.random.normal(0, 10, n_samples)
    
    # Inject outliers
    # Extreme high values in feature1
    outlier_indices = np.random.choice(df.index, size=15, replace=False)
    df.loc[outlier_indices, 'feature1'] = np.random.uniform(200, 300, len(outlier_indices))
    
    # Extreme low values in feature2
    outlier_indices = np.random.choice(df.index, size=10, replace=False)
    df.loc[outlier_indices, 'feature2'] = np.random.uniform(200, 400, len(outlier_indices))
    
    # Random extreme values in feature3
    outlier_indices = np.random.choice(df.index, size=20, replace=False)
    # Split outliers into low and high
    low_outliers = outlier_indices[:len(outlier_indices)//2]
    high_outliers = outlier_indices[len(outlier_indices)//2:]
    df.loc[low_outliers, 'feature3'] = np.random.uniform(0, 100, len(low_outliers))
    df.loc[high_outliers, 'feature3'] = np.random.uniform(1000, 1500, len(high_outliers))
    
    return df


def main():
    """
    Main function demonstrating the complete Task 2 workflow.
    
    Returns:
        OutlierAnalyzer: Configured analyzer object
    """
    print("=" * 80)
    print("Task 2: Outlier Detection and Treatment with Advanced Visualization")
    print("=" * 80)
    print()
    
    # Create sample dataset
    print("Creating sample dataset with outliers...")
    df = create_sample_dataset_with_outliers()
    print(f"Dataset created: {df.shape[0]} rows, {df.shape[1]} columns")
    print()
    
    # Initialize analyzer
    analyzer = OutlierAnalyzer(df)
    
    # Detect outliers using different methods
    print("1. OUTLIER DETECTION")
    print("-" * 80)
    
    print("  a) Z-Score Detection...")
    zscore_outliers = analyzer.detect_outliers_zscore(threshold=3.0)
    print(f"     Outliers detected: {zscore_outliers.sum().sum()}")
    
    print("  b) IQR Detection...")
    iqr_outliers = analyzer.detect_outliers_iqr(multiplier=1.5)
    print(f"     Outliers detected: {iqr_outliers.sum().sum()}")
    
    print("  c) Isolation Forest Detection...")
    iso_outliers = analyzer.detect_outliers_isolation_forest(contamination=0.1)
    print(f"     Outliers detected: {iso_outliers.sum().sum()}")
    
    print("  d) Local Outlier Factor Detection...")
    lof_outliers = analyzer.detect_outliers_lof(n_neighbors=20, contamination=0.1)
    print(f"     Outliers detected: {lof_outliers.sum().sum()}")
    print()
    
    # Get outlier summary
    print("2. OUTLIER SUMMARY")
    print("-" * 80)
    summary = analyzer.get_outlier_summary()
    print(summary)
    print()
    
    # Apply different treatment strategies
    print("3. OUTLIER TREATMENT")
    print("-" * 80)
    
    print("  a) Removing outliers (IQR method)...")
    removed_data = analyzer.remove_outliers(method='iqr')
    print(f"     Rows after removal: {len(removed_data)} (removed {len(df) - len(removed_data)})")
    
    print("  b) Capping outliers (Winsorization)...")
    capped_data = analyzer.cap_outliers(multiplier=1.5)
    print(f"     Data capped at IQR bounds")
    
    print("  c) Log transformation...")
    log_data = analyzer.transform_outliers_log()
    print(f"     Log transformation applied")
    
    print("  d) Box-Cox transformation...")
    boxcox_data = analyzer.transform_outliers_boxcox()
    print(f"     Box-Cox/Yeo-Johnson transformation applied")
    print()
    
    # Evaluate model performance
    print("4. MODEL PERFORMANCE COMPARISON")
    print("-" * 80)
    
    datasets_to_evaluate = ['original', 'removed_iqr', 'capped', 'log_transform']
    for dataset_name in datasets_to_evaluate:
        try:
            performance = analyzer.evaluate_model_performance('target', dataset_name)
            print(f"  {dataset_name.replace('_', ' ').title()}:")
            print(f"    R² Score: {performance['r2_score']:.4f} (±{performance['r2_std']:.4f})")
        except Exception as e:
            print(f"  {dataset_name}: Could not evaluate - {str(e)}")
    print()
    
    # Generate visualizations
    print("5. GENERATING VISUALIZATIONS")
    print("-" * 80)
    
    print("  Creating box plots...")
    analyzer.visualize_outliers_boxplot()
    plt.savefig('task2_boxplots.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_boxplots.png")
    
    print("  Creating violin plots...")
    analyzer.visualize_outliers_violin()
    plt.savefig('task2_violin_plots.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_violin_plots.png")
    
    print("  Creating scatter plots with outlier detection...")
    analyzer.visualize_outliers_scatter('feature1', 'feature2')
    plt.savefig('task2_scatter_outliers.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_scatter_outliers.png")
    
    print("  Creating 3D outlier visualization...")
    analyzer.visualize_3d_outliers('feature1', 'feature2', 'feature3', method='isolation_forest')
    plt.savefig('task2_3d_outliers.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_3d_outliers.png")
    
    print("  Creating treatment comparison plots...")
    analyzer.visualize_treatment_comparison('feature1')
    plt.savefig('task2_treatment_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_treatment_comparison.png")
    
    print("  Creating Q-Q plots...")
    analyzer.visualize_qq_plots()
    plt.savefig('task2_qq_plots.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_qq_plots.png")
    print()
    
    print("=" * 80)
    print("Task 2 completed successfully!")
    print("=" * 80)
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()
    plt.show()
