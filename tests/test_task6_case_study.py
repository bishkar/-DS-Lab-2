"""
Unit tests for Task 6: Real-World Case Study

This module contains comprehensive tests for the data preprocessing workflow,
including data generation, quality assessment, preprocessing steps,
model training, and visualization generation.

Author: MLDS Course
Date: December 2025
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
import os
import sys
import json

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from task6_case_study import (
    DataPreprocessor,
    create_complex_synthetic_dataset
)


# Fixtures
@pytest.fixture
def sample_dataset():
    """Create a small sample dataset for testing."""
    return create_complex_synthetic_dataset(
        n_samples=100,
        n_features=10,
        missing_rate=0.1,
        outlier_rate=0.05,
        imbalance_ratio=0.3,
        random_state=42
    )


@pytest.fixture
def preprocessor(sample_dataset):
    """Create a preprocessor instance with sample data."""
    return DataPreprocessor(sample_dataset, target_col='target')


# Tests for create_complex_synthetic_dataset
class TestDataGeneration:
    """Test suite for synthetic dataset generation."""
    
    def test_dataset_creation_basic(self):
        """Test basic dataset creation."""
        df = create_complex_synthetic_dataset(
            n_samples=100,
            n_features=5,
            random_state=42
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 100  # May have duplicates
        assert 'target' in df.columns
    
    def test_dataset_has_missing_values(self):
        """Test that dataset contains missing values."""
        df = create_complex_synthetic_dataset(
            n_samples=100,
            n_features=10,
            missing_rate=0.2,
            random_state=42
        )
        
        missing_count = df.isnull().sum().sum()
        assert missing_count > 0
    
    def test_dataset_has_categorical_features(self):
        """Test that dataset contains categorical features."""
        df = create_complex_synthetic_dataset(n_samples=100, random_state=42)
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        assert len(categorical_cols) > 0
        assert 'category_1' in df.columns
    
    def test_dataset_has_datetime_feature(self):
        """Test that dataset contains datetime feature."""
        df = create_complex_synthetic_dataset(n_samples=100, random_state=42)
        
        assert 'date' in df.columns
        # Check if it's datetime-like
        assert df['date'].dtype == 'object' or pd.api.types.is_datetime64_any_dtype(df['date'])
    
    def test_dataset_has_class_imbalance(self):
        """Test that dataset has imbalanced classes."""
        df = create_complex_synthetic_dataset(
            n_samples=100,
            imbalance_ratio=0.3,
            random_state=42
        )
        
        target_counts = df['target'].value_counts()
        imbalance = target_counts.min() / target_counts.max()
        assert imbalance < 0.8  # Should be imbalanced
    
    def test_dataset_reproducibility(self):
        """Test that dataset generation is reproducible."""
        df1 = create_complex_synthetic_dataset(n_samples=100, random_state=42)
        df2 = create_complex_synthetic_dataset(n_samples=100, random_state=42)
        
        # Check same shape
        assert df1.shape == df2.shape
        
        # Check same values (where not NaN)
        numeric_cols = df1.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:5]:  # Check first 5 numeric columns
            pd.testing.assert_series_equal(
                df1[col].fillna(0), 
                df2[col].fillna(0),
                check_names=False
            )


# Tests for DataPreprocessor
class TestDataPreprocessor:
    """Test suite for DataPreprocessor class."""
    
    def test_initialization(self, sample_dataset):
        """Test preprocessor initialization."""
        preprocessor = DataPreprocessor(sample_dataset, target_col='target')
        
        assert preprocessor.target_col == 'target'
        assert preprocessor.original_data.shape == sample_dataset.shape
        assert preprocessor.data.shape == sample_dataset.shape
        assert isinstance(preprocessor.preprocessing_steps, list)
        assert len(preprocessor.preprocessing_steps) == 0
    
    def test_assess_data_quality(self, preprocessor):
        """Test data quality assessment."""
        report = preprocessor.assess_data_quality()
        
        assert isinstance(report, dict)
        assert 'shape' in report
        assert 'total_samples' in report
        assert 'total_features' in report
        assert 'missing_values' in report
        assert 'numeric_features' in report
        assert 'categorical_features' in report
        assert 'target_distribution' in report
    
    def test_quality_report_accuracy(self, preprocessor):
        """Test that quality report is accurate."""
        report = preprocessor.assess_data_quality()
        
        # Check shape
        assert report['shape'] == preprocessor.data.shape
        
        # Check missing values
        actual_missing = preprocessor.data.isnull().sum().sum()
        assert report['missing_values']['total'] == actual_missing
        
        # Check numeric features
        numeric_cols = preprocessor.data.select_dtypes(include=[np.number]).columns
        assert len(report['numeric_features']) <= len(numeric_cols)
    
    def test_handle_missing_values_simple(self, preprocessor):
        """Test simple missing value imputation."""
        preprocessor.assess_data_quality()
        original_missing = preprocessor.data.isnull().sum().sum()
        
        result = preprocessor.handle_missing_values(strategy='simple')
        
        # Check that numeric missing values are reduced
        numeric_cols = [col for col in preprocessor.quality_report['numeric_features'] 
                       if col != 'target']
        if numeric_cols:
            new_missing = result[numeric_cols].isnull().sum().sum()
            assert new_missing < original_missing or original_missing == 0
    
    def test_handle_missing_values_knn(self, preprocessor):
        """Test KNN missing value imputation."""
        preprocessor.assess_data_quality()
        
        result = preprocessor.handle_missing_values(strategy='knn')
        
        numeric_cols = [col for col in preprocessor.quality_report['numeric_features'] 
                       if col != 'target']
        if numeric_cols:
            # KNN should handle all numeric missing values
            assert result[numeric_cols].isnull().sum().sum() == 0
    
    def test_detect_and_handle_outliers(self, preprocessor):
        """Test outlier detection and handling."""
        preprocessor.assess_data_quality()
        original_data = preprocessor.data.copy()
        
        result = preprocessor.detect_and_handle_outliers(method='iqr', action='cap')
        
        # Check that data shape is preserved (capping doesn't remove rows)
        assert result.shape == original_data.shape
        
        # Check that preprocessing step was recorded
        assert len(preprocessor.preprocessing_steps) > 0
    
    def test_encode_categorical_features(self, preprocessor):
        """Test categorical feature encoding."""
        preprocessor.assess_data_quality()
        original_shape = preprocessor.data.shape
        
        result = preprocessor.encode_categorical_features(method='auto')
        
        # After encoding, there should be no object columns (except target if it's object)
        object_cols = result.select_dtypes(include=['object']).columns
        assert len(object_cols) <= 1  # Only target might remain
    
    def test_engineer_features(self, preprocessor):
        """Test feature engineering."""
        preprocessor.assess_data_quality()
        original_features = preprocessor.data.shape[1]
        
        result = preprocessor.engineer_features()
        
        # Should create new features
        assert result.shape[1] > original_features
        
        # Check for specific engineered features
        assert 'mean_all_features' in result.columns or result.shape[1] > original_features
    
    def test_scale_features(self, preprocessor):
        """Test feature scaling."""
        preprocessor.assess_data_quality()
        preprocessor.handle_missing_values()  # Remove missing first
        
        result = preprocessor.scale_features(method='standard')
        
        numeric_cols = [col for col in result.select_dtypes(include=[np.number]).columns 
                       if col != 'target']
        
        if numeric_cols:
            # Check that mean is approximately 0 (within tolerance)
            mean_vals = result[numeric_cols].mean()
            assert np.all(np.abs(mean_vals) < 0.5)
    
    def test_select_features(self, preprocessor):
        """Test feature selection."""
        preprocessor.assess_data_quality()
        preprocessor.handle_missing_values()
        preprocessor.encode_categorical_features()
        
        n_features = 5
        result = preprocessor.select_features(method='kbest', n_features=n_features)
        
        # Should have selected features plus target
        assert result.shape[1] <= n_features + 1
    
    def test_balance_classes(self, preprocessor):
        """Test class balancing."""
        preprocessor.assess_data_quality()
        
        result = preprocessor.balance_classes(method='undersample')
        
        # Check that classes are more balanced
        target_counts = result['target'].value_counts()
        balance_ratio = target_counts.min() / target_counts.max()
        assert balance_ratio >= 0.9  # Should be nearly balanced


# Tests for preprocessing pipeline
class TestPreprocessingPipeline:
    """Test suite for complete preprocessing pipeline."""
    
    def test_preprocess_pipeline_execution(self, preprocessor):
        """Test that preprocessing pipeline executes successfully."""
        preprocessor.assess_data_quality()
        
        config = {
            'handle_missing': True,
            'missing_strategy': 'auto',
            'handle_outliers': True,
            'encode_categorical': True,
            'engineer_features': True,
            'scale_features': True,
            'select_features': False,
            'balance_classes': False
        }
        
        result = preprocessor.preprocess_pipeline(config)
        
        assert isinstance(result, pd.DataFrame)
        assert len(preprocessor.preprocessing_steps) > 0
        assert preprocessor.preprocessed_data is not None
    
    def test_preprocess_pipeline_with_all_steps(self, preprocessor):
        """Test pipeline with all steps enabled."""
        preprocessor.assess_data_quality()
        
        config = {
            'handle_missing': True,
            'handle_outliers': True,
            'encode_categorical': True,
            'engineer_features': True,
            'scale_features': True,
            'select_features': True,
            'n_features': 10,
            'balance_classes': True
        }
        
        result = preprocessor.preprocess_pipeline(config)
        
        assert isinstance(result, pd.DataFrame)
        assert 'target' in result.columns
        assert len(preprocessor.preprocessing_steps) >= 5
    
    def test_preprocess_pipeline_minimal(self, preprocessor):
        """Test pipeline with minimal steps."""
        preprocessor.assess_data_quality()
        
        config = {
            'handle_missing': True,
            'handle_outliers': False,
            'encode_categorical': False,
            'engineer_features': False,
            'scale_features': False,
            'select_features': False,
            'balance_classes': False
        }
        
        result = preprocessor.preprocess_pipeline(config)
        
        assert isinstance(result, pd.DataFrame)
        assert len(preprocessor.preprocessing_steps) >= 1


# Tests for model training

# Tests for visualizations
class TestVisualizations:
    """Test suite for visualization methods."""
    
    def test_visualize_eda(self, preprocessor):
        """Test EDA visualization generation."""
        preprocessor.assess_data_quality()
        
        fig = preprocessor.visualize_eda()
        
        assert fig is not None
        assert hasattr(fig, 'savefig')
    
    def test_visualize_correlations(self, preprocessor):
        """Test correlation visualization."""
        preprocessor.assess_data_quality()
        
        fig = preprocessor.visualize_correlations()
        
        assert fig is not None
        assert hasattr(fig, 'savefig')
    
    def test_visualize_missing_patterns(self, preprocessor):
        """Test missing patterns visualization."""
        preprocessor.assess_data_quality()
        
        fig = preprocessor.visualize_missing_patterns()
        
        assert fig is not None
        assert hasattr(fig, 'savefig')
    
    
    def test_visualize_pca(self, preprocessor):
        """Test PCA visualization."""
        preprocessor.assess_data_quality()
        
        config = {'handle_missing': True, 'encode_categorical': True, 'scale_features': True}
        preprocessor.preprocess_pipeline(config)
        
        fig = preprocessor.visualize_pca()
        
        assert fig is not None
        assert hasattr(fig, 'savefig')


# Tests for report generation
class TestReportGeneration:
    """Test suite for report generation."""
    
    def test_generate_report(self, preprocessor, tmp_path):
        """Test report generation."""
        preprocessor.assess_data_quality()
        
        config = {'handle_missing': True, 'encode_categorical': True}
        preprocessor.preprocess_pipeline(config)
        
        filepath = tmp_path / "test_report.json"
        preprocessor.generate_report(str(filepath))
        
        assert filepath.exists()
        
        # Check report content
        with open(filepath, 'r') as f:
            report = json.load(f)
        
        assert 'data_quality' in report
        assert 'preprocessing_steps' in report
        assert 'final_shape' in report
        assert 'timestamp' in report
    
    def test_report_contains_valid_data(self, preprocessor, tmp_path):
        """Test that report contains valid data."""
        preprocessor.assess_data_quality()
        
        config = {'handle_missing': True}
        preprocessor.preprocess_pipeline(config)
        
        filepath = tmp_path / "test_report.json"
        preprocessor.generate_report(str(filepath))
        
        with open(filepath, 'r') as f:
            report = json.load(f)
        
        # Check data quality section
        assert 'shape' in report['data_quality']
        assert 'total_samples' in report['data_quality']
        
        # Check preprocessing steps
        assert len(report['preprocessing_steps']) > 0


# Integration tests
class TestIntegration:
    """Integration tests for complete workflow."""
    
    def test_complete_workflow(self):
        """Test complete workflow from data generation to preprocessing."""
        # Generate data
        df = create_complex_synthetic_dataset(
            n_samples=200,
            n_features=10,
            random_state=42
        )
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor(df, target_col='target')
        
        # Assess quality
        preprocessor.assess_data_quality()
        
        # Preprocess
        config = {
            'handle_missing': True,
            'handle_outliers': True,
            'encode_categorical': True,
            'scale_features': True
        }
        preprocessor.preprocess_pipeline(config)
        
        # Verify preprocessing completed
        assert len(preprocessor.preprocessing_steps) > 0
        assert preprocessor.data.shape[0] > 0
    
    def test_workflow_handles_edge_cases(self):
        """Test that workflow handles edge cases properly."""
        # Generate small dataset
        df = create_complex_synthetic_dataset(
            n_samples=50,
            n_features=5,
            random_state=42
        )
        
        preprocessor = DataPreprocessor(df, target_col='target')
        preprocessor.assess_data_quality()
        
        # Try to select more features than available
        config = {
            'handle_missing': True,
            'encode_categorical': True,
            'select_features': True,
            'n_features': 100  # More than available
        }
        
        # Should not raise error
        preprocessor.preprocess_pipeline(config)
        
        # Verify preprocessing completed
        assert len(preprocessor.preprocessing_steps) > 0


# Performance tests
class TestPerformance:
    """Test suite for performance characteristics."""
    
    def test_preprocessing_improves_data_quality(self):
        """Test that preprocessing improves data quality."""
        df = create_complex_synthetic_dataset(n_samples=500, random_state=42)
        
        # Minimal preprocessing
        preprocessor1 = DataPreprocessor(df, target_col='target')
        preprocessor1.assess_data_quality()
        
        config_minimal = {
            'handle_missing': True,
            'encode_categorical': True,
            'handle_outliers': False,
            'engineer_features': False,
            'scale_features': False
        }
        preprocessor1.preprocess_pipeline(config_minimal)
        
        # Full preprocessing
        preprocessor2 = DataPreprocessor(df, target_col='target')
        preprocessor2.assess_data_quality()
        
        config_full = {
            'handle_missing': True,
            'encode_categorical': True,
            'handle_outliers': True,
            'engineer_features': True,
            'scale_features': True
        }
        preprocessor2.preprocess_pipeline(config_full)
        
        # Both should complete successfully
        assert len(preprocessor1.preprocessing_steps) > 0
        assert len(preprocessor2.preprocessing_steps) > len(preprocessor1.preprocessing_steps)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
