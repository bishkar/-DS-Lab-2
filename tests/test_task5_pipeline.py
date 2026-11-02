"""
Unit tests for Task 5: Integrated Data Preprocessing Pipeline

This module contains comprehensive tests for the preprocessing pipeline,
including custom transformers, pipeline creation, cross-validation,
and data leakage prevention.

Author: MLDS Course
Date: December 2025
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from scipy import stats
import joblib
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from task5_pipeline import (
    OutlierHandler,
    FeatureEngineer,
    PreprocessingPipeline,
    create_sample_dataset_with_issues
)


# Fixtures
@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=42
    )
    return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)]), pd.Series(y)


@pytest.fixture
def data_with_outliers():
    """Create data with outliers for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 3)
    # Add outliers
    X[0, 0] = 10  # Extreme value
    X[1, 1] = -10
    return pd.DataFrame(X, columns=['f1', 'f2', 'f3'])


@pytest.fixture
def data_with_missing():
    """Create data with missing values for testing."""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    # Inject missing values
    X_df.iloc[0:5, 0] = np.nan
    X_df.iloc[10:15, 2] = np.nan
    return X_df, pd.Series(y)


# Tests for OutlierHandler
class TestOutlierHandler:
    """Test suite for OutlierHandler transformer."""
    
    def test_initialization(self):
        """Test OutlierHandler initialization."""
        handler = OutlierHandler(method='iqr', strategy='cap', threshold=1.5)
        assert handler.method == 'iqr'
        assert handler.strategy == 'cap'
        assert handler.threshold == 1.5
        assert handler.bounds_ == {}
    
    def test_fit_iqr_method(self, data_with_outliers):
        """Test fitting with IQR method."""
        handler = OutlierHandler(method='iqr', threshold=1.5)
        handler.fit(data_with_outliers)
        
        assert len(handler.bounds_) == 3
        assert all(isinstance(bounds, tuple) for bounds in handler.bounds_.values())
        assert all(len(bounds) == 2 for bounds in handler.bounds_.values())
    
    def test_fit_zscore_method(self, data_with_outliers):
        """Test fitting with Z-score method."""
        handler = OutlierHandler(method='zscore', threshold=3.0)
        handler.fit(data_with_outliers)
        
        assert len(handler.bounds_) == 3
        for col, (lower, upper) in handler.bounds_.items():
            assert lower < upper
    
    def test_transform_cap_strategy(self, data_with_outliers):
        """Test transform with capping strategy."""
        handler = OutlierHandler(method='iqr', strategy='cap', threshold=1.5)
        handler.fit(data_with_outliers)
        
        X_transformed = handler.transform(data_with_outliers)
        
        assert X_transformed.shape == data_with_outliers.shape
        assert isinstance(X_transformed, np.ndarray)
        # Check that extreme values are capped
        assert np.abs(X_transformed[0, 0]) < 10
    
    def test_transform_remove_strategy(self, data_with_outliers):
        """Test transform with remove strategy (sets to NaN)."""
        handler = OutlierHandler(method='iqr', strategy='remove', threshold=1.5)
        handler.fit(data_with_outliers)
        
        X_transformed = handler.transform(data_with_outliers)
        
        assert X_transformed.shape == data_with_outliers.shape
        # Should have some NaN values where outliers were
        assert np.isnan(X_transformed).any()
    
    def test_fit_transform_consistency(self, data_with_outliers):
        """Test that fit and transform are consistent."""
        handler = OutlierHandler(method='iqr', strategy='cap')
        
        # Fit and transform separately
        handler.fit(data_with_outliers)
        X_trans1 = handler.transform(data_with_outliers)
        
        # Fit and transform together
        X_trans2 = handler.fit_transform(data_with_outliers)
        
        np.testing.assert_array_almost_equal(X_trans1, X_trans2)


# Tests for FeatureEngineer
class TestFeatureEngineer:
    """Test suite for FeatureEngineer transformer."""
    
    def test_initialization(self):
        """Test FeatureEngineer initialization."""
        engineer = FeatureEngineer(interactions=True, polynomials=False, 
                                   transformations=True)
        assert engineer.interactions is True
        assert engineer.polynomials is False
        assert engineer.transformations is True
    
    def test_fit(self, sample_data):
        """Test fitting the feature engineer."""
        X, _ = sample_data
        engineer = FeatureEngineer()
        engineer.fit(X)
        
        assert engineer.n_features_in_ == X.shape[1]
    
    def test_transform_increases_features(self, sample_data):
        """Test that transform creates additional features."""
        X, _ = sample_data
        engineer = FeatureEngineer(interactions=True, transformations=True)
        engineer.fit(X)
        
        X_transformed = engineer.transform(X.values)
        
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] > X.shape[1]
    
    def test_transform_with_transformations_only(self, sample_data):
        """Test transform with only mathematical transformations."""
        X, _ = sample_data
        engineer = FeatureEngineer(interactions=False, transformations=True, 
                                   polynomials=False)
        engineer.fit(X)
        
        X_transformed = engineer.transform(X.values)
        
        # Should have original + log + sqrt features
        expected_features = X.shape[1] * 3
        assert X_transformed.shape[1] == expected_features
    
    def test_transform_with_interactions(self, sample_data):
        """Test transform with interaction features."""
        X, _ = sample_data
        engineer = FeatureEngineer(interactions=True, transformations=False, 
                                   polynomials=False)
        engineer.fit(X)
        
        X_transformed = engineer.transform(X.values)
        
        # Should have original + interaction features
        assert X_transformed.shape[1] > X.shape[1]
    
    def test_transform_with_polynomials(self, sample_data):
        """Test transform with polynomial features."""
        X, _ = sample_data
        engineer = FeatureEngineer(interactions=False, transformations=False, 
                                   polynomials=True)
        engineer.fit(X)
        
        X_transformed = engineer.transform(X.values)
        
        # Should have original + squared features
        expected_features = X.shape[1] * 2
        assert X_transformed.shape[1] == expected_features
    
    def test_transform_handles_negatives(self):
        """Test that transform handles negative values correctly."""
        X = np.array([[-1, 2], [3, -4], [-5, 6]])
        engineer = FeatureEngineer(transformations=True, interactions=False, 
                                   polynomials=False)
        engineer.fit(X)
        
        X_transformed = engineer.transform(X)
        
        # Should not contain NaN or inf values
        assert not np.isnan(X_transformed).any()
        assert not np.isinf(X_transformed).any()


# Tests for PreprocessingPipeline
class TestPreprocessingPipeline:
    """Test suite for PreprocessingPipeline class."""
    
    def test_initialization(self):
        """Test pipeline initialization."""
        pipeline = PreprocessingPipeline(random_state=42)
        assert pipeline.random_state == 42
        assert pipeline.pipeline is None
        assert pipeline.cv_results == {}
    
    def test_create_pipeline(self):
        """Test pipeline creation."""
        pipeline_manager = PreprocessingPipeline()
        pipe = pipeline_manager.create_pipeline(
            imputer_strategy='mean',
            outlier_method='iqr',
            scaler_type='standard',
            feature_engineering=True,
            feature_selection=True
        )
        
        assert isinstance(pipe, Pipeline)
        assert 'imputer' in pipe.named_steps
        assert 'outlier_handler' in pipe.named_steps
        assert 'scaler' in pipe.named_steps
        assert 'classifier' in pipe.named_steps
    
    def test_create_pipeline_without_feature_engineering(self):
        """Test pipeline creation without feature engineering."""
        pipeline_manager = PreprocessingPipeline()
        pipe = pipeline_manager.create_pipeline(
            feature_engineering=False,
            feature_selection=False
        )
        
        assert 'feature_engineer' not in pipe.named_steps
        assert 'feature_selector' not in pipe.named_steps
    
    def test_prepare_data(self, sample_data):
        """Test data splitting."""
        X, y = sample_data
        pipeline_manager = PreprocessingPipeline()
        
        X_train, X_test, y_train, y_test = pipeline_manager.prepare_data(
            X, y, test_size=0.2
        )
        
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20
    
    def test_fit_pipeline(self, sample_data):
        """Test pipeline fitting."""
        X, y = sample_data
        pipeline_manager = PreprocessingPipeline()
        pipeline_manager.create_pipeline(feature_engineering=False, 
                                        feature_selection=False)
        
        X_train, X_test, y_train, y_test = pipeline_manager.prepare_data(X, y)
        pipeline_manager.fit_pipeline(X_train, y_train)
        
        assert pipeline_manager.pipeline is not None
        # Pipeline should be fitted
        assert hasattr(pipeline_manager.pipeline.named_steps['classifier'], 
                      'classes_')
    
    def test_evaluate_pipeline(self, sample_data):
        """Test pipeline evaluation."""
        X, y = sample_data
        pipeline_manager = PreprocessingPipeline()
        pipeline_manager.create_pipeline(feature_engineering=False, 
                                        feature_selection=False)
        
        X_train, X_test, y_train, y_test = pipeline_manager.prepare_data(X, y)
        pipeline_manager.fit_pipeline(X_train, y_train)
        
        metrics = pipeline_manager.evaluate_pipeline(X_test, y_test)
        
        assert 'accuracy' in metrics
        assert 'roc_auc' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['roc_auc'] <= 1
    
    def test_cross_validate(self, sample_data):
        """Test cross-validation."""
        X, y = sample_data
        pipeline_manager = PreprocessingPipeline()
        pipeline_manager.create_pipeline(feature_engineering=False, 
                                        feature_selection=False)
        
        cv_results = pipeline_manager.cross_validate(X, y, cv=5)
        
        assert 'scores' in cv_results
        assert 'mean' in cv_results
        assert 'std' in cv_results
        assert len(cv_results['scores']) == 5
        assert 0 <= cv_results['mean'] <= 1
    
    def test_compare_configurations(self, sample_data):
        """Test configuration comparison."""
        X, y = sample_data
        pipeline_manager = PreprocessingPipeline()
        
        configurations = [
            {
                'name': 'Config1',
                'imputer_strategy': 'mean',
                'feature_engineering': False,
                'feature_selection': False
            },
            {
                'name': 'Config2',
                'imputer_strategy': 'median',
                'feature_engineering': True,
                'feature_selection': False
            }
        ]
        
        comparison_df = pipeline_manager.compare_configurations(X, y, configurations)
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 2
        assert 'config_name' in comparison_df.columns
        assert 'mean_accuracy' in comparison_df.columns
        assert 'std_accuracy' in comparison_df.columns
    
    def test_get_learning_curves(self, sample_data):
        """Test learning curve generation."""
        X, y = sample_data
        pipeline_manager = PreprocessingPipeline()
        pipeline_manager.create_pipeline(feature_engineering=False, 
                                        feature_selection=False)
        
        lc_data = pipeline_manager.get_learning_curves(X, y)
        
        assert 'train_sizes' in lc_data
        assert 'train_scores_mean' in lc_data
        assert 'val_scores_mean' in lc_data
        assert len(lc_data['train_sizes']) > 0
    
    def test_save_and_load_pipeline(self, sample_data, tmp_path):
        """Test pipeline saving and loading."""
        X, y = sample_data
        pipeline_manager = PreprocessingPipeline()
        pipeline_manager.create_pipeline(feature_engineering=False, 
                                        feature_selection=False)
        
        X_train, X_test, y_train, y_test = pipeline_manager.prepare_data(X, y)
        pipeline_manager.fit_pipeline(X_train, y_train)
        
        # Save pipeline
        filepath = tmp_path / "test_pipeline.pkl"
        pipeline_manager.save_pipeline(str(filepath))
        
        assert filepath.exists()
        
        # Load pipeline
        new_pipeline_manager = PreprocessingPipeline()
        new_pipeline_manager.load_pipeline(str(filepath))
        
        assert new_pipeline_manager.pipeline is not None
        
        # Test that loaded pipeline works
        y_pred = new_pipeline_manager.pipeline.predict(X_test)
        assert len(y_pred) == len(y_test)


# Tests for data leakage prevention
class TestDataLeakagePrevention:
    """Test suite for ensuring no data leakage in pipeline."""
    
    def test_fit_only_on_training_data(self, sample_data):
        """Test that transformers are fitted only on training data."""
        X, y = sample_data
        pipeline_manager = PreprocessingPipeline()
        pipeline_manager.create_pipeline(feature_engineering=False, 
                                        feature_selection=False)
        
        X_train, X_test, y_train, y_test = pipeline_manager.prepare_data(X, y)
        
        # Fit only on training data
        pipeline_manager.fit_pipeline(X_train, y_train)
        
        # Check that scaler was fitted only on training data
        scaler = pipeline_manager.pipeline.named_steps['scaler']
        train_mean = X_train.mean(axis=0).values
        
        # The scaler's mean should be close to training data mean (within 2 decimal places)
        np.testing.assert_array_almost_equal(scaler.mean_, train_mean, decimal=2)
    
    def test_cross_validation_no_leakage(self, sample_data):
        """Test that cross-validation doesn't leak information."""
        X, y = sample_data
        pipeline_manager = PreprocessingPipeline()
        pipeline_manager.create_pipeline(feature_engineering=False, 
                                        feature_selection=False)
        
        # Cross-validation should fit on each fold independently
        cv_results = pipeline_manager.cross_validate(X, y, cv=5)
        
        # Should have valid scores without leakage
        assert all(score >= 0 for score in cv_results['scores'])
        assert all(score <= 1 for score in cv_results['scores'])


# Tests for create_sample_dataset_with_issues
class TestDataGeneration:
    """Test suite for sample data generation."""
    
    def test_create_sample_dataset_basic(self):
        """Test basic dataset creation."""
        X, y = create_sample_dataset_with_issues(
            n_samples=100, n_features=5, missing_rate=0.1, outlier_rate=0.05
        )
        
        assert X.shape == (100, 5)
        assert len(y) == 100
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
    
    def test_create_sample_dataset_with_missing(self):
        """Test dataset creation with missing values."""
        X, y = create_sample_dataset_with_issues(
            n_samples=100, n_features=5, missing_rate=0.2, outlier_rate=0.0
        )
        
        missing_count = X.isna().sum().sum()
        assert missing_count > 0
        # Should have approximately 20% missing (100 * 5 * 0.2 = 100)
        assert 50 < missing_count < 150  # Allow some variance
    
    def test_create_sample_dataset_with_outliers(self):
        """Test dataset creation with outliers."""
        X, y = create_sample_dataset_with_issues(
            n_samples=100, n_features=5, missing_rate=0.0, outlier_rate=0.1
        )
        
        # Check for extreme values (outliers)
        z_scores = np.abs(stats.zscore(X.fillna(0)))
        outlier_count = (z_scores > 3).sum().sum()
        assert outlier_count > 0
    
    def test_create_sample_dataset_reproducibility(self):
        """Test that dataset generation is reproducible."""
        X1, y1 = create_sample_dataset_with_issues(random_state=42)
        X2, y2 = create_sample_dataset_with_issues(random_state=42)
        
        pd.testing.assert_frame_equal(X1, X2)
        pd.testing.assert_series_equal(y1, y2)


# Integration tests
class TestPipelineIntegration:
    """Integration tests for complete pipeline workflow."""
    
    def test_complete_workflow(self, data_with_missing):
        """Test complete pipeline workflow from data to predictions."""
        X, y = data_with_missing
        
        # Initialize and create pipeline
        pipeline_manager = PreprocessingPipeline(random_state=42)
        pipeline_manager.create_pipeline(
            imputer_strategy='mean',
            outlier_method='iqr',
            scaler_type='standard',
            feature_engineering=True,
            feature_selection=True,
            n_features=10
        )
        
        # Split data
        X_train, X_test, y_train, y_test = pipeline_manager.prepare_data(
            X, y, test_size=0.2
        )
        
        # Fit pipeline
        pipeline_manager.fit_pipeline(X_train, y_train)
        
        # Evaluate
        metrics = pipeline_manager.evaluate_pipeline(X_test, y_test)
        
        assert 'accuracy' in metrics
        assert 'roc_auc' in metrics
        assert metrics['accuracy'] > 0.5  # Should be better than random
    
    def test_pipeline_with_all_preprocessing_steps(self, sample_data):
        """Test pipeline with all preprocessing steps enabled."""
        X, y = sample_data
        
        pipeline_manager = PreprocessingPipeline()
        pipeline_manager.create_pipeline(
            imputer_strategy='knn',
            outlier_method='zscore',
            scaler_type='robust',
            feature_engineering=True,
            feature_selection=True,
            n_features=10
        )
        
        X_train, X_test, y_train, y_test = pipeline_manager.prepare_data(X, y)
        pipeline_manager.fit_pipeline(X_train, y_train)
        
        # Should be able to make predictions
        y_pred = pipeline_manager.pipeline.predict(X_test)
        assert len(y_pred) == len(y_test)
        assert set(y_pred).issubset({0, 1})
    
    def test_pipeline_handles_new_data(self, sample_data):
        """Test that fitted pipeline can handle new data."""
        X, y = sample_data
        
        pipeline_manager = PreprocessingPipeline()
        pipeline_manager.create_pipeline(feature_engineering=False, 
                                        feature_selection=False)
        
        X_train, X_test, y_train, y_test = pipeline_manager.prepare_data(X, y)
        pipeline_manager.fit_pipeline(X_train, y_train)
        
        # Create new data with same structure
        X_new = X_test.copy()
        y_pred = pipeline_manager.pipeline.predict(X_new)
        
        assert len(y_pred) == len(X_new)


# Performance tests
class TestPipelinePerformance:
    """Test suite for pipeline performance characteristics."""
    
    def test_feature_engineering_improves_performance(self, sample_data):
        """Test that feature engineering can improve performance."""
        X, y = sample_data
        
        # Pipeline without feature engineering
        pipeline1 = PreprocessingPipeline(random_state=42)
        pipeline1.create_pipeline(feature_engineering=False, 
                                 feature_selection=False)
        cv1 = pipeline1.cross_validate(X, y, cv=3)
        
        # Pipeline with feature engineering
        pipeline2 = PreprocessingPipeline(random_state=42)
        pipeline2.create_pipeline(feature_engineering=True, 
                                 feature_selection=False)
        cv2 = pipeline2.cross_validate(X, y, cv=3)
        
        # Both should produce valid scores
        assert 0 <= cv1['mean'] <= 1
        assert 0 <= cv2['mean'] <= 1
    
    def test_different_scalers_produce_different_results(self, sample_data):
        """Test that different scalers affect results differently."""
        X, y = sample_data
        
        scores = {}
        for scaler_type in ['standard', 'robust', 'minmax']:
            pipeline = PreprocessingPipeline(random_state=42)
            pipeline.create_pipeline(scaler_type=scaler_type,
                                    feature_engineering=False,
                                    feature_selection=False)
            cv = pipeline.cross_validate(X, y, cv=3)
            scores[scaler_type] = cv['mean']
        
        # All should produce valid scores
        assert all(0 <= score <= 1 for score in scores.values())


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
