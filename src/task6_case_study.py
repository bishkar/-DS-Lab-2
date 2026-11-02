"""
Task 6: Real-World Case Study - Complete Data Preprocessing Challenge

This module implements a complete end-to-end data preprocessing workflow
for a complex synthetic dataset with multiple data quality issues.
Demonstrates best practices for EDA, preprocessing, and model evaluation.

Author: MLDS Course
Date: December 2025
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.utils import resample
from scipy import stats
from typing import Tuple, Dict, List, Optional, Any, Literal, Union
from functools import partial
from itertools import combinations
from datetime import datetime, timedelta
import warnings
import json

from pydantic import NonNegativeInt

from common import (DEFAULT_K, SCALERS, OutlierMethod, Schema, clip_to_bounds, count_duplicates,
                    dtype_map, fill_mode, memory_mb, missing_summary, outlier_bounds, outside,
                    safe_divide, validated)

warnings.filterwarnings('ignore')


MissingStrategy = Literal['auto', 'simple', 'knn', 'iterative']
OutlierAction = Literal['cap', 'remove']
EncodingMethod = Literal['auto', 'onehot', 'label']
ScalingMethod = Literal['standard', 'robust']
FeatureKind = Literal['numeric', 'categorical', 'datetime']

_median_imputer = partial(SimpleImputer, strategy='median', keep_empty_features=True)
NUMERIC_IMPUTERS = {
    'auto': _median_imputer,
    'simple': _median_imputer,
    'knn': partial(KNNImputer, n_neighbors=5, keep_empty_features=True),
    'iterative': partial(IterativeImputer, max_iter=10, random_state=42, keep_empty_features=True),
}
DATETIME_PARTS = {'year': 'year', 'month': 'month', 'day': 'day', 'dayofweek': 'day_of_week'}
ROW_STATISTICS = ('mean', 'std', 'min', 'max')


class MissingValues(Schema):
    total: NonNegativeInt
    by_column: Dict[str, NonNegativeInt]
    percentage_by_column: Dict[str, float]


class QualityReport(Schema):
    shape: Tuple[NonNegativeInt, NonNegativeInt]
    total_samples: NonNegativeInt
    total_features: NonNegativeInt
    memory_usage_mb: float
    duplicates: NonNegativeInt
    missing_values: MissingValues
    data_types: Dict[str, str]
    numeric_features: List[str]
    categorical_features: List[str]
    datetime_features: List[str]
    target_distribution: Dict[Union[int, float, str], NonNegativeInt]
    class_balance: float


def feature_kind(series: pd.Series) -> FeatureKind:
    if pd.api.types.is_datetime64_any_dtype(series):
        return 'datetime'
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return 'numeric'
    return 'categorical'


class DataPreprocessor:
    """
    Comprehensive data preprocessor for real-world case study.
    Handles EDA, data quality assessment, and preprocessing.
    """
    
    def __init__(self, data: pd.DataFrame, target_col: str):
        """
        Initialize the data preprocessor.
        
        Parameters:
            data (pd.DataFrame): Input dataset
            target_col (str): Name of target column
        """
        self.original_data = data.copy()
        self.data = data.copy()
        self.target_col = target_col
        self.preprocessed_data = None
        self.quality_report = {}
        self.preprocessing_steps = []
        self.encoders = {}
        self.scaler = None

    def assess_data_quality(self) -> Dict[str, Any]:
        """
        Perform comprehensive data quality assessment.
        
        Returns:
            Dict: Quality assessment report
        """

        missing = missing_summary(self.data)
        kinds = {col: feature_kind(self.data[col]) for col in self.data.columns}
        counts = (self.data[self.target_col].value_counts()
                  if self.target_col in self.data.columns else pd.Series(dtype=int))

        def of_kind(kind: FeatureKind) -> List[str]:
            return [col for col, k in kinds.items() if k == kind]

        self.quality_report = QualityReport(
            shape=self.data.shape,
            total_samples=len(self.data),
            total_features=len(self.data.columns),
            memory_usage_mb=memory_mb(self.data),
            duplicates=count_duplicates(self.data),
            missing_values=MissingValues(
                total=missing.total,
                by_column=missing.nonzero_counts,
                percentage_by_column=missing.nonzero_percentages,
            ),
            data_types=dtype_map(self.data),
            numeric_features=of_kind('numeric'),
            categorical_features=[col for col in of_kind('categorical') if col != self.target_col],
            datetime_features=of_kind('datetime'),
            target_distribution={(key.item() if hasattr(key, 'item') else key): int(n)
                                 for key, n in counts.items()},
            class_balance=float(counts.min() / counts.max()) if len(counts) else 0.0,
        ).model_dump()
        return self.quality_report

    def print_quality_summary(self):
        self._ensure_quality_report()
        self.print_quality_report()

    def _ensure_quality_report(self):
        if not self.quality_report:
            self.assess_data_quality()

    def _report_columns(self, key: str) -> List[str]:
        self._ensure_quality_report()
        return [col for col in self.quality_report[key]
                if col != self.target_col and col in self.data.columns]

    def _columns_of_kind(self, data: pd.DataFrame, kind: FeatureKind) -> List[str]:
        return [col for col in data.columns if col != self.target_col and feature_kind(data[col]) == kind]

    def _commit(self, data: pd.DataFrame, step: str) -> pd.DataFrame:
        self.data = data
        self.preprocessing_steps.append(step)
        return data

    def print_quality_report(self):
        """
        Print formatted data quality report.
        """
        print("=" * 80)
        print("DATA QUALITY ASSESSMENT REPORT")
        print("=" * 80)
        print()
        
        print(f"Dataset Shape: {self.quality_report['shape']}")
        print(f"Total Samples: {self.quality_report['total_samples']:,}")
        print(f"Total Features: {self.quality_report['total_features']}")
        print(f"Memory Usage: {self.quality_report['memory_usage_mb']:.2f} MB")
        print(f"Duplicate Rows: {self.quality_report['duplicates']}")
        print()
        
        print("MISSING VALUES:")
        print(f"  Total Missing: {self.quality_report['missing_values']['total']:,}")
        if self.quality_report['missing_values']['by_column']:
            print("  Columns with Missing Values:")
            for col, count in self.quality_report['missing_values']['by_column'].items():
                pct = self.quality_report['missing_values']['percentage_by_column'][col]
                print(f"    {col}: {count} ({pct:.2f}%)")
        print()
        
        print("FEATURE TYPES:")
        print(f"  Numeric: {len(self.quality_report['numeric_features'])}")
        print(f"  Categorical: {len(self.quality_report['categorical_features'])}")
        print(f"  Datetime: {len(self.quality_report['datetime_features'])}")
        print()
        
        if self.quality_report['target_distribution']:
            print("TARGET DISTRIBUTION:")
            for cls, count in self.quality_report['target_distribution'].items():
                pct = (count / self.quality_report['total_samples']) * 100
                print(f"  Class {cls}: {count} ({pct:.2f}%)")
            print(f"  Class Balance Ratio: {self.quality_report['class_balance']:.3f}")
        print()
    
    @validated
    def handle_missing_values(self, strategy: MissingStrategy = 'auto')-> pd.DataFrame:
        """
        Handle missing values with specified strategy.
        
        Parameters:
            strategy (str): Imputation strategy
            
        Returns:
            pd.DataFrame: Data with missing values handled
        """

        data = self.data.copy()
        numeric_cols = self._report_columns('numeric_features')
        if numeric_cols:
            data[numeric_cols] = NUMERIC_IMPUTERS[strategy]().fit_transform(data[numeric_cols])
        fill_mode(data, self._report_columns('categorical_features'))
        return self._commit(data, f"Missing values handled: {strategy}")

    @validated
    def detect_and_handle_outliers(self, method: OutlierMethod = 'iqr',
                                   action: OutlierAction = 'cap')-> pd.DataFrame:
        """
        Detect and handle outliers.
        
        Parameters:
            method (str): Detection method
            action (str): Action to take on outliers
            
        Returns:
            pd.DataFrame: Data with outliers handled
        """

        data = self.data.copy()
        bounds = {col: outlier_bounds(data[col], method, DEFAULT_K[method])
                  for col in self._columns_of_kind(data, 'numeric')}
        masks = pd.DataFrame({col: outside(data[col], b) for col, b in bounds.items()},
                             index=data.index, dtype=bool)
        if action == 'cap':
            for col, b in bounds.items():
                data[col] = clip_to_bounds(data[col], b)
        else:
            data = data.loc[~masks.any(axis=1)].reset_index(drop=True)
        return self._commit(
            data, f"Outliers handled: {method} method, {action} action, {int(masks.values.sum())} outliers"
        )

    @validated
    def encode_categorical_features(self, method: EncodingMethod = 'auto')-> pd.DataFrame:
        """
        Encode categorical features.
        
        Parameters:
            method (str): Encoding method
            
        Returns:
            pd.DataFrame: Data with encoded categorical features
        """

        data = self.data.copy()
        for col in self._columns_of_kind(data, 'datetime'):
            dates = pd.to_datetime(data[col]).dt
            for name, attribute in DATETIME_PARTS.items():
                data[f'{col}_{name}'] = getattr(dates, attribute)
            data = data.drop(columns=[col])

        for col in self._columns_of_kind(data, 'categorical'):
            if method == 'onehot' or (method == 'auto' and data[col].nunique() <= 10):
                dummies = pd.get_dummies(data[col], prefix=col, dtype=int)
                self.encoders[col] = list(dummies.columns)
                data = pd.concat([data.drop(columns=[col]), dummies], axis=1)
            else:
                self.encoders[col] = LabelEncoder()
                data[col] = self.encoders[col].fit_transform(data[col].astype(str))
        return self._commit(data, f"Categorical encoding: {method}, datetime features extracted")

    @validated
    def engineer_features(self, domain_features: bool = True) -> pd.DataFrame:
        """
        Engineer new features.
        
        Parameters:
            domain_features (bool): Whether to create domain-specific features
            
        Returns:
            pd.DataFrame: Data with engineered features
        """

        data = self.data.copy()
        numeric_cols = self._report_columns('numeric_features')
        new_features = []
        if len(numeric_cols) >= 2:
            for stat in ROW_STATISTICS:
                data[f'{stat}_all_features'] = getattr(data[numeric_cols], stat)(axis=1)
                new_features.append(f'{stat}_all_features')
        if domain_features:
            for a, b in combinations(numeric_cols[:3], 2):
                data[f'{a}_x_{b}'] = data[a] * data[b]
                data[f'{a}_div_{b}'] = safe_divide(data[a], data[b])
                new_features += [f'{a}_x_{b}', f'{a}_div_{b}']
        return self._commit(data, f"Feature engineering: {len(new_features)} new features created")

    @validated
    def scale_features(self, method: ScalingMethod = 'standard')-> pd.DataFrame:
        """
        Scale numerical features.
        
        Parameters:
            method (str): Scaling method
            
        Returns:
            pd.DataFrame: Data with scaled features
        """

        data = self.data.copy()
        numeric_cols = self._columns_of_kind(data, 'numeric')
        if numeric_cols and len(data) > 0:
            self.scaler = SCALERS[method]()
            data[numeric_cols] = self.scaler.fit_transform(data[numeric_cols].astype(float))
        return self._commit(data, f"Feature scaling: {method}")

    def select_features(self, method: str = 'kbest', n_features: int = 20) -> pd.DataFrame:
        """
        Select most important features.
        
        Parameters:
            method (str): Feature selection method
            n_features (int): Number of features to select
            
        Returns:
            pd.DataFrame: Data with selected features
        """
        data = self.data.copy()
        
        if self.target_col not in data.columns:
            return data
        
        X = data.drop(columns=[self.target_col])
        y = data[self.target_col]
        
        # Ensure we don't select more features than available
        n_features = min(n_features, X.shape[1])
        
        if method == 'kbest':
            selector = SelectKBest(score_func=f_classif, k=n_features)
        elif method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
        else:
            return data
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        data = pd.DataFrame(X_selected, columns=selected_features)
        data[self.target_col] = y.values
        
        self.data = data
        self.preprocessing_steps.append(
            f"Feature selection: {method}, {len(selected_features)} features selected"
        )
        return data
    
    def balance_classes(self, method: str = 'undersample') -> pd.DataFrame:
        """
        Balance class distribution.
        
        Parameters:
            method (str): Balancing method
            
        Returns:
            pd.DataFrame: Balanced dataset
        """
        data = self.data.copy()
        
        if self.target_col not in data.columns:
            return data
        
        # Separate majority and minority classes
        class_counts = data[self.target_col].value_counts()
        majority_class = class_counts.idxmax()
        minority_class = class_counts.idxmin()
        
        df_majority = data[data[self.target_col] == majority_class]
        df_minority = data[data[self.target_col] == minority_class]
        
        if method == 'undersample':
            # Downsample majority class
            df_majority_downsampled = resample(
                df_majority, 
                replace=False,
                n_samples=len(df_minority),
                random_state=42
            )
            data = pd.concat([df_majority_downsampled, df_minority])
        
        elif method == 'oversample':
            # Upsample minority class
            df_minority_upsampled = resample(
                df_minority,
                replace=True,
                n_samples=len(df_majority),
                random_state=42
            )
            data = pd.concat([df_majority, df_minority_upsampled])
        
        self.data = data
        self.preprocessing_steps.append(f"Class balancing: {method}")
        return data
    
    def preprocess_pipeline(self, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Execute complete preprocessing pipeline.
        
        Parameters:
            config (Dict): Configuration for preprocessing steps
            
        Returns:
            pd.DataFrame: Fully preprocessed data
        """
        print("Starting preprocessing pipeline...")
        print("-" * 80)
        
        # Step 1: Handle missing values
        if config.get('handle_missing', True):
            print("Step 1: Handling missing values...")
            self.data = self.handle_missing_values(config.get('missing_strategy', 'auto'))
        
        # Step 2: Handle outliers
        if config.get('handle_outliers', True):
            print("Step 2: Handling outliers...")
            self.data = self.detect_and_handle_outliers(
                method=config.get('outlier_method', 'iqr'),
                action=config.get('outlier_action', 'cap')
            )
        
        # Step 3: Encode categorical features
        if config.get('encode_categorical', True):
            print("Step 3: Encoding categorical features...")
            self.data = self.encode_categorical_features(config.get('encoding_method', 'auto'))
        
        # Step 4: Engineer features
        if config.get('engineer_features', True):
            print("Step 4: Engineering features...")
            self.data = self.engineer_features()
        
        # Step 5: Scale features
        if config.get('scale_features', True):
            print("Step 5: Scaling features...")
            self.data = self.scale_features(config.get('scaling_method', 'standard'))
        
        # Step 6: Select features
        if config.get('select_features', False):
            print("Step 6: Selecting features...")
            self.data = self.select_features(
                method=config.get('selection_method', 'kbest'),
                n_features=config.get('n_features', 20)
            )
        
        # Step 7: Balance classes
        if config.get('balance_classes', False):
            print("Step 7: Balancing classes...")
            self.data = self.balance_classes(config.get('balance_method', 'undersample'))
        
        print("-" * 80)
        print(f"Preprocessing complete! Final shape: {self.data.shape}")
        print()
        
        self.preprocessed_data = self.data.copy()
        return self.data
    
    
    def visualize_eda(self, figsize: Tuple[int, int] = (20, 15)):
        """
        Create comprehensive EDA visualizations.
        
        Parameters:
            figsize (Tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        numeric_cols = [col for col in self.quality_report['numeric_features'] 
                       if col != self.target_col]
        categorical_cols = self.quality_report['categorical_features']
        
        # Create subplot grid
        n_numeric = min(len(numeric_cols), 6)
        n_categorical = min(len(categorical_cols), 3)
        n_rows = 3
        n_cols = 3
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(n_rows, n_cols, hspace=0.3, wspace=0.3)
        
        # Numeric distributions
        for i, col in enumerate(numeric_cols[:6]):
            row = i // 3
            col_idx = i % 3
            ax = fig.add_subplot(gs[row, col_idx])
            self.original_data[col].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
            ax.set_title(f'Distribution: {col}', fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.grid(alpha=0.3)
        
        # Categorical distributions
        for i, col in enumerate(categorical_cols[:3]):
            if i + 6 < 9:
                ax = fig.add_subplot(gs[2, i])
                value_counts = self.original_data[col].value_counts().head(10)
                value_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
                ax.set_title(f'Distribution: {col}', fontweight='bold')
                ax.set_xlabel('Category')
                ax.set_ylabel('Count')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(alpha=0.3, axis='y')
        
        plt.suptitle('Exploratory Data Analysis - Feature Distributions', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        return fig
    
    def visualize_correlations(self, figsize: Tuple[int, int] = (12, 10)):
        """
        Visualize feature correlations.
        
        Parameters:
            figsize (Tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        numeric_data = self.original_data.select_dtypes(include=[np.number])
        
        # Limit to reasonable number of features
        if numeric_data.shape[1] > 20:
            numeric_data = numeric_data.iloc[:, :20]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        corr_matrix = numeric_data.corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
        
        ax.set_title('Feature Correlation Matrix', fontweight='bold', pad=20, fontsize=14)
        plt.tight_layout()
        
        return fig
    
    def visualize_missing_patterns(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Visualize missing value patterns.
        
        Parameters:
            figsize (Tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Missing values heatmap
        missing_matrix = self.original_data.isnull()
        
        # Sample if too many rows
        if len(missing_matrix) > 500:
            missing_matrix = missing_matrix.sample(500, random_state=42)
        
        sns.heatmap(missing_matrix, cbar=True, yticklabels=False, 
                   cmap='viridis', ax=ax1)
        ax1.set_title('Missing Values Heatmap\n(Yellow = Missing)', 
                     fontweight='bold', pad=10)
        ax1.set_xlabel('Features')
        ax1.set_ylabel('Samples')
        
        # Missing values bar plot
        missing_counts = self.original_data.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
        
        if len(missing_counts) > 0:
            missing_counts.plot(kind='barh', ax=ax2, color='coral', edgecolor='black')
            ax2.set_title('Missing Values by Feature', fontweight='bold', pad=10)
            ax2.set_xlabel('Count')
            ax2.set_ylabel('Feature')
            ax2.grid(alpha=0.3, axis='x')
        else:
            ax2.text(0.5, 0.5, 'No Missing Values', 
                    ha='center', va='center', fontsize=14)
            ax2.set_title('Missing Values by Feature', fontweight='bold', pad=10)
        
        plt.tight_layout()
        return fig
    
    
    def visualize_pca(self, n_components: int = 2, figsize: Tuple[int, int] = (10, 8)):
        """
        Visualize PCA dimensionality reduction.
        
        Parameters:
            n_components (int): Number of PCA components
            figsize (Tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        if self.target_col not in self.data.columns:
            raise ValueError("Target column not found")
        
        X = self.data.drop(columns=[self.target_col])
        y = self.data[self.target_col]
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot by class
        for class_val in y.unique():
            mask = y == class_val
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                      label=f'Class {class_val}', alpha=0.6, s=50)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', 
                     fontweight='bold')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', 
                     fontweight='bold')
        ax.set_title('PCA - Feature Space Visualization', fontweight='bold', 
                    pad=20, fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_report(self, filepath: str = 'preprocessing_report.json'):
        """
        Generate comprehensive preprocessing report.
        
        Parameters:
            filepath (str): Path to save report
        """
        report = {
            'data_quality': self.quality_report,
            'preprocessing_steps': self.preprocessing_steps,
            'final_shape': self.data.shape,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Report saved to {filepath}")


def create_complex_synthetic_dataset(n_samples: int = 2000,
                                     n_features: int = 25,
                                     missing_rate: float = 0.15,
                                     outlier_rate: float = 0.08,
                                     imbalance_ratio: float = 0.3,
                                     random_state: int = 42) -> pd.DataFrame:
    """
    Create a complex synthetic dataset with multiple data quality issues.
    
    Parameters:
        n_samples (int): Number of samples
        n_features (int): Number of features
        missing_rate (float): Proportion of missing values
        outlier_rate (float): Proportion of outliers
        imbalance_ratio (float): Class imbalance ratio
        random_state (int): Random seed
        
    Returns:
        pd.DataFrame: Complex synthetic dataset
    """
    np.random.seed(random_state)
    
    # Generate numerical features
    numeric_features = {}
    for i in range(n_features):
        if i % 3 == 0:
            # Normal distribution
            numeric_features[f'feature_{i}'] = np.random.randn(n_samples) * 10 + 50
        elif i % 3 == 1:
            # Exponential distribution
            numeric_features[f'feature_{i}'] = np.random.exponential(scale=20, size=n_samples)
        else:
            # Uniform distribution
            numeric_features[f'feature_{i}'] = np.random.uniform(0, 100, n_samples)
    
    df = pd.DataFrame(numeric_features)
    
    # Add categorical features
    categories = ['A', 'B', 'C', 'D', 'E']
    df['category_1'] = np.random.choice(categories, size=n_samples)
    df['category_2'] = np.random.choice(['Type1', 'Type2', 'Type3'], size=n_samples)
    df['category_3'] = np.random.choice(['Low', 'Medium', 'High', 'Very High'], size=n_samples)
    
    # Add datetime feature
    start_date = datetime(2020, 1, 1)
    df['date'] = [start_date + timedelta(days=int(x)) 
                  for x in np.random.uniform(0, 1000, n_samples)]
    
    # Create target variable with imbalance
    # Based on some features
    target_score = (
        df['feature_0'] * 0.3 + 
        df['feature_1'] * 0.2 + 
        df['feature_2'] * 0.1 +
        np.random.randn(n_samples) * 10
    )
    
    threshold = np.percentile(target_score, imbalance_ratio * 100)
    df['target'] = (target_score > threshold).astype(int)
    
    # Inject missing values
    n_missing = int(n_samples * len(df.columns) * missing_rate)
    missing_indices = np.random.choice(n_samples * len(df.columns), 
                                      size=n_missing, replace=False)
    for idx in missing_indices:
        row = idx // len(df.columns)
        col = idx % len(df.columns)
        if col < len(numeric_features):  # Only numeric features
            df.iloc[row, col] = np.nan
    
    # Inject outliers
    n_outliers = int(n_samples * outlier_rate)
    numeric_cols = list(numeric_features.keys())
    for _ in range(n_outliers):
        row = np.random.randint(0, n_samples)
        col = np.random.choice(numeric_cols)
        df.loc[row, col] = df[col].mean() + np.random.choice([-1, 1]) * df[col].std() * 5
    
    # Add some duplicate rows
    n_duplicates = int(n_samples * 0.02)
    duplicate_indices = np.random.choice(n_samples, size=n_duplicates, replace=False)
    df = pd.concat([df, df.iloc[duplicate_indices]], ignore_index=True)
    
    return df


def main():
    """
    Main function demonstrating the complete Task 6 workflow.
    
    Returns:
        DataPreprocessor: Configured preprocessor object
    """
    print("=" * 80)
    print("Task 6: Real-World Case Study - Complete Data Preprocessing Challenge")
    print("=" * 80)
    print()
    
    # Create complex synthetic dataset
    print("Creating complex synthetic dataset...")
    df = create_complex_synthetic_dataset(
        n_samples=2000,
        n_features=25,
        missing_rate=0.15,
        outlier_rate=0.08,
        imbalance_ratio=0.3
    )
    print(f"Dataset created: {df.shape[0]} samples, {df.shape[1]} features")
    print()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(df, target_col='target')
    
    # Step 1: Data Quality Assessment
    print("STEP 1: DATA QUALITY ASSESSMENT")
    print("=" * 80)
    preprocessor.assess_data_quality()
    preprocessor.print_quality_report()
    
    # Step 2: EDA Visualizations
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    print("Generating EDA visualizations...")
    
    preprocessor.visualize_eda()
    plt.savefig('task6_eda_distributions.png', dpi=300, bbox_inches='tight')
    print("  Saved: task6_eda_distributions.png")
    
    preprocessor.visualize_correlations()
    plt.savefig('task6_correlations.png', dpi=300, bbox_inches='tight')
    print("  Saved: task6_correlations.png")
    
    preprocessor.visualize_missing_patterns()
    plt.savefig('task6_missing_patterns.png', dpi=300, bbox_inches='tight')
    print("  Saved: task6_missing_patterns.png")
    print()
    
    # Step 3: Preprocessing Pipeline
    print("STEP 3: PREPROCESSING PIPELINE")
    print("=" * 80)
    
    config = {
        'handle_missing': True,
        'missing_strategy': 'auto',
        'handle_outliers': True,
        'outlier_method': 'iqr',
        'outlier_action': 'cap',
        'encode_categorical': True,
        'encoding_method': 'auto',
        'engineer_features': True,
        'scale_features': True,
        'scaling_method': 'robust',
        'select_features': True,
        'selection_method': 'kbest',
        'n_features': 30,
        'balance_classes': True,
        'balance_method': 'undersample'
    }
    
    preprocessor.preprocess_pipeline(config)
    print()
    
    # Step 4: Additional Visualizations
    print("STEP 4: GENERATING ADDITIONAL VISUALIZATIONS")
    print("=" * 80)
    
    print("  Creating PCA visualization...")
    preprocessor.visualize_pca()
    plt.savefig('task6_pca_visualization.png', dpi=300, bbox_inches='tight')
    print("  Saved: task6_pca_visualization.png")
    print()
    
    # Step 5: Generate Report
    print("STEP 5: GENERATING REPORT")
    print("=" * 80)
    preprocessor.generate_report('task6_preprocessing_report.json')
    print()
    
    # Summary
    print("=" * 80)
    print("TASK 6 COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print()
    print("SUMMARY:")
    print(f"  Original shape: {preprocessor.original_data.shape}")
    print(f"  Final shape: {preprocessor.data.shape}")
    print(f"  Preprocessing steps: {len(preprocessor.preprocessing_steps)}")
    print()
    print("FILES GENERATED:")
    print("  - task6_eda_distributions.png")
    print("  - task6_correlations.png")
    print("  - task6_missing_patterns.png")
    print("  - task6_pca_visualization.png")
    print("  - task6_preprocessing_report.json")
    print()
    
    return preprocessor


if __name__ == "__main__":
    preprocessor = main()
    plt.show()
