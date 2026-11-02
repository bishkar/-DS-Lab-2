"""
Task 5: Integrated Data Preprocessing Pipeline with Cross-Validation

This module implements an end-to-end data preprocessing pipeline that combines
cleaning, scaling, and feature engineering with proper cross-validation to
prevent data leakage.

Author: MLDS Course
Date: December 2025
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split, cross_val_score, learning_curve, 
    GridSearchCV, StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    roc_curve, auc, roc_auc_score
)
from sklearn.datasets import make_classification
from scipy import stats
from typing import Tuple, Dict, List, Optional, Any
import warnings
import joblib

warnings.filterwarnings('ignore')


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Custom transformer for outlier detection and handling.
    
    Parameters:
        method (str): Method for outlier detection
        strategy (str): Strategy for handling outliers
        threshold (float): Threshold for detection
    """
    
    def __init__(self, method: str = 'iqr', strategy: str = 'cap', threshold: float = 1.5):
        self.method = method
        self.strategy = strategy
        self.threshold = threshold
        self.bounds_ = {}
        
    def fit(self, X, y=None):
        """
        Fit the outlier handler by computing bounds.
        
        Parameters:
            X: Feature matrix
            y: Target vector (ignored)
            
        Returns:
            self: Fitted transformer
        """
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        for col in X.columns:
            if self.method == 'iqr':
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - self.threshold * IQR
                upper = Q3 + self.threshold * IQR
                self.bounds_[col] = (lower, upper)
            elif self.method == 'zscore':
                mean = X[col].mean()
                std = X[col].std()
                lower = mean - self.threshold * std
                upper = mean + self.threshold * std
                self.bounds_[col] = (lower, upper)
                
        return self
    
    def transform(self, X):
        """
        Transform data by handling outliers.
        
        Parameters:
            X: Feature matrix
            
        Returns:
            np.ndarray: Transformed feature matrix
        """
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        if self.strategy == 'cap':
            for col in X.columns:
                if col in self.bounds_:
                    lower, upper = self.bounds_[col]
                    X[col] = X[col].clip(lower=lower, upper=upper)
        elif self.strategy == 'remove':
            # Mark outliers as NaN for imputation
            for col in X.columns:
                if col in self.bounds_:
                    lower, upper = self.bounds_[col]
                    mask = (X[col] < lower) | (X[col] > upper)
                    X.loc[mask, col] = np.nan
                    
        return X.values


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for feature engineering.
    
    Parameters:
        interactions (bool): Whether to create interaction features
        polynomials (bool): Whether to create polynomial features
        transformations (bool): Whether to create mathematical transformations
    """
    
    def __init__(self, interactions: bool = True, polynomials: bool = False, 
                 transformations: bool = True):
        self.interactions = interactions
        self.polynomials = polynomials
        self.transformations = transformations
        self.feature_names_ = []
        self.n_features_in_ = 0
        
    def fit(self, X, y=None):
        """
        Fit the feature engineer.
        
        Parameters:
            X: Feature matrix
            y: Target vector (ignored)
            
        Returns:
            self: Fitted transformer
        """
        self.n_features_in_ = X.shape[1]
        return self
    
    def transform(self, X):
        """
        Transform data by creating new features.
        
        Parameters:
            X: Feature matrix
            
        Returns:
            np.ndarray: Transformed feature matrix with new features
        """
        X_array = np.array(X)
        features = [X_array]
        
        # Mathematical transformations
        if self.transformations:
            # Log transform (add small constant to avoid log(0))
            log_features = np.log1p(np.abs(X_array) + 1e-8)
            features.append(log_features)
            
            # Square root transform
            sqrt_features = np.sqrt(np.abs(X_array))
            features.append(sqrt_features)
        
        # Interaction features (limited to avoid explosion)
        if self.interactions and X_array.shape[1] <= 10:
            n_features = X_array.shape[1]
            for i in range(min(3, n_features)):
                for j in range(i + 1, min(3, n_features)):
                    interaction = X_array[:, i] * X_array[:, j]
                    features.append(interaction.reshape(-1, 1))
        
        # Polynomial features (degree 2, limited features)
        if self.polynomials and X_array.shape[1] <= 5:
            square_features = X_array ** 2
            features.append(square_features)
        
        return np.hstack(features)


class PreprocessingPipeline:
    """
    Integrated preprocessing pipeline combining multiple preprocessing steps
    with proper cross-validation to prevent data leakage.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the preprocessing pipeline.
        
        Parameters:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.pipeline = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.cv_results = {}
        self.feature_names = []
        
    def create_pipeline(self, imputer_strategy: str = 'mean',
                       outlier_method: str = 'iqr',
                       scaler_type: str = 'standard',
                       feature_engineering: bool = True,
                       feature_selection: bool = True,
                       n_features: int = 20) -> Pipeline:
        """
        Create a sklearn pipeline with specified preprocessing steps.
        
        Parameters:
            imputer_strategy (str): Strategy for imputation
            outlier_method (str): Method for outlier detection
            scaler_type (str): Type of scaler to use
            feature_engineering (bool): Whether to apply feature engineering
            feature_selection (bool): Whether to apply feature selection
            n_features (int): Number of features to select
            
        Returns:
            Pipeline: Configured sklearn pipeline
        """
        steps = []
        
        # Step 1: Imputation
        pass
        
        # Step 2: Outlier handling
        pass
        
        # Step 3: Feature engineering
        pass
        
        # Step 4: Scaling
        pass
        
        # Step 5: Feature selection
        pass
        
        # Step 6: Classifier
        pass
        
        self.pipeline = Pipeline(steps)
        return self.pipeline
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, 
                    test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, 
                                                      np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets.
        
        Parameters:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target vector
            test_size (float): Proportion of test set
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def fit_pipeline(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fit the pipeline on training data.
        
        Parameters:
            X_train: Training feature matrix
            y_train: Training target vector
            
        Returns:
            self: Fitted pipeline
        """
        if self.pipeline is None:
            self.create_pipeline()
        
        self.pipeline.fit(X_train, y_train)
        return self
    
    def evaluate_pipeline(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the fitted pipeline on test data.
        
        Parameters:
            X_test: Test feature matrix
            y_test: Test target vector
            
        Returns:
            Dict: Evaluation metrics
        """
        y_pred = self.pipeline.predict(X_test)
        y_pred_proba = self.pipeline.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                      cv: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation on the pipeline.
        
        Parameters:
            X: Feature matrix
            y: Target vector
            cv (int): Number of cross-validation folds
            
        Returns:
            Dict: Cross-validation results
        """
        if self.pipeline is None:
            self.create_pipeline()
        
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, 
                                     random_state=self.random_state)
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.pipeline, X, y, cv=cv_splitter, 
                                   scoring='accuracy')
        
        self.cv_results = {
            'scores': cv_scores,
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'min': cv_scores.min(),
            'max': cv_scores.max()
        }
        
        return self.cv_results
    
    def compare_configurations(self, X: np.ndarray, y: np.ndarray, 
                              configurations: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare different pipeline configurations.
        
        Parameters:
            X: Feature matrix
            y: Target vector
            configurations (List[Dict]): List of configuration dictionaries
            
        Returns:
            pd.DataFrame: Comparison results
        """
        results = []
        
        for i, config in enumerate(configurations):
            print(f"Evaluating configuration {i+1}/{len(configurations)}...")
            
            # Extract config name and create pipeline config without it
            config_name = config.get('name', f'Config_{i+1}')
            pipeline_config = {k: v for k, v in config.items() if k != 'name'}
            
            # Create pipeline with this configuration
            self.create_pipeline(**pipeline_config)
            
            # Cross-validate
            cv_results = self.cross_validate(X, y, cv=5)
            
            result = {
                'config_name': config_name,
                'mean_accuracy': cv_results['mean'],
                'std_accuracy': cv_results['std'],
                **pipeline_config
            }
            results.append(result)
        
        return pd.DataFrame(results)
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray,
                                param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Optimize pipeline hyperparameters using GridSearchCV.
        
        Parameters:
            X: Feature matrix
            y: Target vector
            param_grid (Dict): Parameter grid for search
            
        Returns:
            Dict: Best parameters and scores
        """
        if self.pipeline is None:
            self.create_pipeline()
        
        grid_search = GridSearchCV(
            self.pipeline, param_grid, cv=5, scoring='accuracy',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def get_learning_curves(self, X: np.ndarray, y: np.ndarray,
                           train_sizes: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Generate learning curves for the pipeline.
        
        Parameters:
            X: Feature matrix
            y: Target vector
            train_sizes: Training set sizes to evaluate
            
        Returns:
            Dict: Learning curve data
        """
        if self.pipeline is None:
            self.create_pipeline()
        
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_sizes_abs, train_scores, val_scores = learning_curve(
            self.pipeline, X, y, train_sizes=train_sizes,
            cv=5, scoring='accuracy', n_jobs=-1, random_state=self.random_state
        )
        
        return {
            'train_sizes': train_sizes_abs,
            'train_scores_mean': train_scores.mean(axis=1),
            'train_scores_std': train_scores.std(axis=1),
            'val_scores_mean': val_scores.mean(axis=1),
            'val_scores_std': val_scores.std(axis=1)
        }
    
    def save_pipeline(self, filepath: str):
        """
        Save the fitted pipeline to disk.
        
        Parameters:
            filepath (str): Path to save the pipeline
        """
        if self.pipeline is None:
            raise ValueError("No pipeline to save. Create and fit a pipeline first.")
        
        joblib.dump(self.pipeline, filepath)
        print(f"Pipeline saved to {filepath}")
    
    def load_pipeline(self, filepath: str):
        """
        Load a pipeline from disk.
        
        Parameters:
            filepath (str): Path to load the pipeline from
        """
        self.pipeline = joblib.load(filepath)
        print(f"Pipeline loaded from {filepath}")
    
    def visualize_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   title: str = 'Confusion Matrix',
                                   figsize: Tuple[int, int] = (8, 6)):
        """
        Visualize confusion matrix.
        
        Parameters:
            y_true: True labels
            y_pred: Predicted labels
            title (str): Plot title
            figsize (Tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted Label', fontweight='bold')
        ax.set_ylabel('True Label', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig
    
    def visualize_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                           title: str = 'ROC Curve',
                           figsize: Tuple[int, int] = (8, 6)):
        """
        Visualize ROC curve.
        
        Parameters:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            title (str): Plot title
            figsize (Tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
               label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=20)
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def visualize_learning_curves(self, learning_curve_data: Dict[str, np.ndarray],
                                  title: str = 'Learning Curves',
                                  figsize: Tuple[int, int] = (10, 6)):
        """
        Visualize learning curves.
        
        Parameters:
            learning_curve_data (Dict): Data from get_learning_curves
            title (str): Plot title
            figsize (Tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        train_sizes = learning_curve_data['train_sizes']
        train_mean = learning_curve_data['train_scores_mean']
        train_std = learning_curve_data['train_scores_std']
        val_mean = learning_curve_data['val_scores_mean']
        val_std = learning_curve_data['val_scores_std']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Training scores
        ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                       alpha=0.2, color='blue')
        
        # Validation scores
        ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                       alpha=0.2, color='red')
        
        ax.set_xlabel('Training Set Size', fontweight='bold')
        ax.set_ylabel('Accuracy Score', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=20)
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def visualize_cv_scores(self, cv_scores: np.ndarray,
                           title: str = 'Cross-Validation Scores',
                           figsize: Tuple[int, int] = (10, 6)):
        """
        Visualize cross-validation scores as box plot.
        
        Parameters:
            cv_scores: Array of cross-validation scores
            title (str): Plot title
            figsize (Tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        bp = ax.boxplot([cv_scores], labels=['CV Scores'], patch_artist=True,
                       boxprops=dict(facecolor='lightblue'),
                       medianprops=dict(color='red', linewidth=2),
                       widths=0.5)
        
        # Add individual points
        ax.scatter([1] * len(cv_scores), cv_scores, color='darkblue', 
                  alpha=0.5, s=50, zorder=3)
        
        ax.set_ylabel('Accuracy Score', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)
        
        # Add statistics text
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()
        ax.text(1.2, mean_score, f'Mean: {mean_score:.4f}\nStd: {std_score:.4f}',
               verticalalignment='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def visualize_pipeline_comparison(self, comparison_df: pd.DataFrame,
                                     title: str = 'Pipeline Configuration Comparison',
                                     figsize: Tuple[int, int] = (12, 6)):
        """
        Visualize comparison of different pipeline configurations.
        
        Parameters:
            comparison_df (pd.DataFrame): Results from compare_configurations
            title (str): Plot title
            figsize (Tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        config_names = comparison_df['config_name'].values
        means = comparison_df['mean_accuracy'].values
        stds = comparison_df['std_accuracy'].values
        
        x_pos = np.arange(len(config_names))
        
        bars = ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.7,
                     capsize=10, color='steelblue', edgecolor='black')
        
        # Color the best performing bar differently
        best_idx = np.argmax(means)
        bars[best_idx].set_color('gold')
        bars[best_idx].set_edgecolor('darkgoldenrod')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(config_names, rotation=45, ha='right')
        ax.set_ylabel('Mean Accuracy', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 0.01, f'{mean:.4f}', 
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        return fig


def create_sample_dataset_with_issues(n_samples: int = 1000, 
                                      n_features: int = 20,
                                      missing_rate: float = 0.1,
                                      outlier_rate: float = 0.05,
                                      random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create a sample classification dataset with data quality issues.
    
    Parameters:
        n_samples (int): Number of samples
        n_features (int): Number of features
        missing_rate (float): Proportion of missing values
        outlier_rate (float): Proportion of outliers
        random_state (int): Random seed
        
    Returns:
        Tuple: (X, y) feature matrix and target vector
    """
    np.random.seed(random_state)
    
    # Create classification dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.7),
        n_redundant=int(n_features * 0.2),
        n_classes=2,
        class_sep=1.0,
        random_state=random_state
    )
    
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    
    # Inject missing values
    n_missing = int(n_samples * n_features * missing_rate)
    missing_indices = np.random.choice(n_samples * n_features, size=n_missing, replace=False)
    for idx in missing_indices:
        row = idx // n_features
        col = idx % n_features
        X_df.iloc[row, col] = np.nan
    
    # Inject outliers
    n_outliers = int(n_samples * outlier_rate)
    outlier_indices = np.random.choice(n_samples, size=n_outliers, replace=False)
    for idx in outlier_indices:
        col = np.random.randint(0, n_features)
        X_df.iloc[idx, col] = X_df.iloc[idx, col] * np.random.uniform(5, 10)
    
    return X_df, pd.Series(y, name='target')


def main():
    """
    Main function demonstrating the complete Task 5 workflow.
    
    Returns:
        PreprocessingPipeline: Configured pipeline object
    """
    print("=" * 80)
    print("Task 5: Integrated Data Preprocessing Pipeline with Cross-Validation")
    print("=" * 80)
    print()
    
    # Create sample dataset
    print("Creating sample dataset with quality issues...")
    X, y = create_sample_dataset_with_issues(
        n_samples=1000, n_features=20, missing_rate=0.1, outlier_rate=0.05
    )
    print(f"Dataset created: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Missing values: {X.isna().sum().sum()}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    print()
    
    # Initialize pipeline
    pipeline_manager = PreprocessingPipeline(random_state=42)
    
    # Prepare data
    print("1. DATA SPLITTING")
    print("-" * 80)
    X_train, X_test, y_train, y_test = pipeline_manager.prepare_data(X, y, test_size=0.2)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print()
    
    # Create and evaluate baseline pipeline
    print("2. BASELINE PIPELINE (No Feature Engineering)")
    print("-" * 80)
    pipeline_manager.create_pipeline(
        imputer_strategy='mean',
        outlier_method='iqr',
        scaler_type='standard',
        feature_engineering=False,
        feature_selection=False
    )
    
    baseline_cv = pipeline_manager.cross_validate(X_train, y_train, cv=5)
    print(f"Cross-validation accuracy: {baseline_cv['mean']:.4f} (±{baseline_cv['std']:.4f})")
    print()
    
    # Fit baseline and evaluate on test set
    pipeline_manager.fit_pipeline(X_train, y_train)
    baseline_test = pipeline_manager.evaluate_pipeline(X_test, y_test)
    print(f"Test set accuracy: {baseline_test['accuracy']:.4f}")
    print(f"Test set ROC-AUC: {baseline_test['roc_auc']:.4f}")
    print()
    
    # Compare different configurations
    print("3. COMPARING PIPELINE CONFIGURATIONS")
    print("-" * 80)
    
    configurations = [
        {
            'name': 'Baseline',
            'imputer_strategy': 'mean',
            'outlier_method': 'iqr',
            'scaler_type': 'standard',
            'feature_engineering': False,
            'feature_selection': False
        },
        {
            'name': 'With Feature Eng',
            'imputer_strategy': 'mean',
            'outlier_method': 'iqr',
            'scaler_type': 'standard',
            'feature_engineering': True,
            'feature_selection': False
        },
        {
            'name': 'With Selection',
            'imputer_strategy': 'mean',
            'outlier_method': 'iqr',
            'scaler_type': 'standard',
            'feature_engineering': False,
            'feature_selection': True,
            'n_features': 15
        },
        {
            'name': 'Full Pipeline',
            'imputer_strategy': 'knn',
            'outlier_method': 'iqr',
            'scaler_type': 'robust',
            'feature_engineering': True,
            'feature_selection': True,
            'n_features': 20
        }
    ]
    
    comparison_df = pipeline_manager.compare_configurations(X_train, y_train, configurations)
    print("\nConfiguration Comparison:")
    print(comparison_df[['config_name', 'mean_accuracy', 'std_accuracy']].to_string(index=False))
    print()
    
    # Generate learning curves for best configuration
    print("4. GENERATING LEARNING CURVES")
    print("-" * 80)
    best_config = comparison_df.loc[comparison_df['mean_accuracy'].idxmax()]
    print(f"Best configuration: {best_config['config_name']}")
    
    # Recreate pipeline with best config
    pipeline_manager.create_pipeline(
        imputer_strategy=best_config.get('imputer_strategy', 'mean'),
        outlier_method=best_config.get('outlier_method', 'iqr'),
        scaler_type=best_config.get('scaler_type', 'standard'),
        feature_engineering=best_config.get('feature_engineering', False),
        feature_selection=best_config.get('feature_selection', False),
        n_features=best_config.get('n_features', 20)
    )
    
    lc_data = pipeline_manager.get_learning_curves(X_train, y_train)
    print("Learning curves generated")
    print()
    
    # Fit final pipeline and evaluate
    print("5. FINAL PIPELINE EVALUATION")
    print("-" * 80)
    pipeline_manager.fit_pipeline(X_train, y_train)
    final_metrics = pipeline_manager.evaluate_pipeline(X_test, y_test)
    print(f"Final test accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Final test ROC-AUC: {final_metrics['roc_auc']:.4f}")
    print()
    
    # Generate predictions for visualizations
    y_pred = pipeline_manager.pipeline.predict(X_test)
    y_pred_proba = pipeline_manager.pipeline.predict_proba(X_test)[:, 1]
    
    # Create visualizations
    print("6. GENERATING VISUALIZATIONS")
    print("-" * 80)
    
    print("  Creating confusion matrix...")
    pipeline_manager.visualize_confusion_matrix(y_test, y_pred, 
                                               title='Confusion Matrix - Final Pipeline')
    plt.savefig('task5_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("  Saved: task5_confusion_matrix.png")
    
    print("  Creating ROC curve...")
    pipeline_manager.visualize_roc_curve(y_test, y_pred_proba,
                                        title='ROC Curve - Final Pipeline')
    plt.savefig('task5_roc_curve.png', dpi=300, bbox_inches='tight')
    print("  Saved: task5_roc_curve.png")
    
    print("  Creating learning curves...")
    pipeline_manager.visualize_learning_curves(lc_data,
                                              title='Learning Curves - Best Configuration')
    plt.savefig('task5_learning_curves.png', dpi=300, bbox_inches='tight')
    print("  Saved: task5_learning_curves.png")
    
    print("  Creating CV scores visualization...")
    cv_scores = baseline_cv['scores']
    pipeline_manager.visualize_cv_scores(cv_scores,
                                        title='Cross-Validation Score Distribution')
    plt.savefig('task5_cv_scores.png', dpi=300, bbox_inches='tight')
    print("  Saved: task5_cv_scores.png")
    
    print("  Creating pipeline comparison...")
    pipeline_manager.visualize_pipeline_comparison(comparison_df,
                                                   title='Pipeline Configuration Performance')
    plt.savefig('task5_pipeline_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: task5_pipeline_comparison.png")
    print()
    
    # Save pipeline
    print("7. SAVING PIPELINE")
    print("-" * 80)
    pipeline_manager.save_pipeline('task5_preprocessing_pipeline.pkl')
    print()
    
    print("=" * 80)
    print("Task 5 completed successfully!")
    print("=" * 80)
    print()
    print("Key Findings:")
    print(f"  - Baseline accuracy: {baseline_test['accuracy']:.4f}")
    print(f"  - Final accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  - Improvement: {(final_metrics['accuracy'] - baseline_test['accuracy']):.4f}")
    print(f"  - Best configuration: {best_config['config_name']}")
    
    return pipeline_manager


if __name__ == "__main__":
    pipeline = main()
    plt.show()
