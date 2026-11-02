"""
Unit Tests for Task 1: Data Quality Assessment and Missing Value Analysis

This module contains comprehensive unit tests for the DataQualityAnalyzer class,
testing all major functionality including imputation methods, visualization,
and missing data pattern identification.

Author: MLDS Course
Date: December 2025
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from task1_data_cleaning import (
    DataQualityAnalyzer,
    create_sample_dataset_with_missing
)


class TestDataQualityAnalyzer:
    """Test suite for DataQualityAnalyzer class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing"""
        np.random.seed(42)
        data = pd.DataFrame({
            'col_a': [1, 2, np.nan, 4, 5, 6, 7, 8, np.nan, 10],
            'col_b': [10, 20, 30, np.nan, 50, np.nan, 70, 80, 90, 100],
            'col_c': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            'col_d': ['a', 'b', np.nan, 'd', 'e', 'f', 'g', 'h', 'i', 'j']
        })
        return data
    
    @pytest.fixture
    def analyzer(self, sample_data):
        """Create analyzer instance"""
        return DataQualityAnalyzer(sample_data)
    
    def test_initialization(self, sample_data):
        """Test analyzer initialization"""
        analyzer = DataQualityAnalyzer(sample_data)
        
        assert analyzer.data is not None
        assert analyzer.original_data is not None
        assert len(analyzer.data) == len(sample_data)
        assert isinstance(analyzer.imputed_datasets, dict)
        assert len(analyzer.imputed_datasets) == 0
    
    def test_assess_data_quality(self, analyzer):
        """Test data quality assessment"""
        report = analyzer.assess_data_quality()
        
        assert 'total_rows' in report
        assert 'total_columns' in report
        assert 'duplicates' in report
        assert 'missing_values' in report
        assert 'missing_percentage' in report
        assert 'data_types' in report
        assert 'memory_usage' in report
        
        assert report['total_rows'] == 10
        assert report['total_columns'] == 4
        assert isinstance(report['duplicates'], (int, np.integer))
        assert isinstance(report['missing_values'], dict)
        assert isinstance(report['memory_usage'], (float, np.floating))
    
    def test_missing_values_count(self, analyzer):
        """Test correct counting of missing values"""
        report = analyzer.assess_data_quality()
        
        assert report['missing_values']['col_a'] == 2
        assert report['missing_values']['col_b'] == 2
        assert report['missing_values']['col_c'] == 0
        assert report['missing_values']['col_d'] == 1
    
    def test_missing_percentage_calculation(self, analyzer):
        """Test missing percentage calculation"""
        report = analyzer.assess_data_quality()
        
        assert report['missing_percentage']['col_a'] == 20.0
        assert report['missing_percentage']['col_b'] == 20.0
        assert report['missing_percentage']['col_c'] == 0.0
        assert report['missing_percentage']['col_d'] == 10.0
    
    def test_identify_missing_pattern(self, analyzer):
        """Test missing pattern identification"""
        pattern = analyzer.identify_missing_pattern('col_a')
        
        assert isinstance(pattern, str)
        assert pattern in [
            "Likely MCAR (Missing Completely at Random)",
            "Likely MAR (Missing at Random)",
            "Possibly MNAR (Missing Not at Random)",
            "No missing values"
        ]
    
    def test_identify_missing_pattern_no_missing(self, analyzer):
        """Test pattern identification for column with no missing values"""
        pattern = analyzer.identify_missing_pattern('col_c')
        assert pattern == "No missing values"
    
    def test_identify_missing_pattern_invalid_column(self, analyzer):
        """Test pattern identification for non-existent column"""
        pattern = analyzer.identify_missing_pattern('nonexistent')
        assert pattern == "Column not found"
    
    def test_create_missing_indicators(self, analyzer):
        """Test creation of missing value indicators"""
        indicators = analyzer.create_missing_indicators()
        
        assert isinstance(indicators, pd.DataFrame)
        assert len(indicators) == len(analyzer.data)
        assert 'col_a_missing' in indicators.columns
        assert 'col_b_missing' in indicators.columns
        assert 'col_d_missing' in indicators.columns
        assert 'col_c_missing' not in indicators.columns  # No missing in col_c
        
        # Check values are binary
        assert indicators['col_a_missing'].isin([0, 1]).all()
        assert indicators['col_a_missing'].sum() == 2  # 2 missing values
    
    def test_simple_imputation_mean(self, analyzer):
        """Test simple mean imputation"""
        imputed = analyzer.simple_imputation(strategy='mean')
        
        assert isinstance(imputed, pd.DataFrame)
        assert len(imputed) == len(analyzer.data)
        assert imputed['col_a'].isnull().sum() == 0
        assert imputed['col_b'].isnull().sum() == 0
        assert 'simple_mean' in analyzer.imputed_datasets
    
    def test_simple_imputation_median(self, analyzer):
        """Test simple median imputation"""
        imputed = analyzer.simple_imputation(strategy='median')
        
        assert isinstance(imputed, pd.DataFrame)
        assert imputed['col_a'].isnull().sum() == 0
        assert imputed['col_b'].isnull().sum() == 0
        assert 'simple_median' in analyzer.imputed_datasets
    
    def test_simple_imputation_mode(self, analyzer):
        """Test simple mode imputation"""
        imputed = analyzer.simple_imputation(strategy='most_frequent')
        
        assert isinstance(imputed, pd.DataFrame)
        assert imputed['col_d'].isnull().sum() == 0
        assert 'simple_most_frequent' in analyzer.imputed_datasets
    
    def test_knn_imputation(self, analyzer):
        """Test KNN imputation"""
        imputed = analyzer.knn_imputation(n_neighbors=3)
        
        assert isinstance(imputed, pd.DataFrame)
        assert imputed['col_a'].isnull().sum() == 0
        assert imputed['col_b'].isnull().sum() == 0
        assert 'knn' in analyzer.imputed_datasets
    
    def test_knn_imputation_custom_neighbors(self, analyzer):
        """Test KNN imputation with custom number of neighbors"""
        imputed = analyzer.knn_imputation(n_neighbors=5)
        
        assert isinstance(imputed, pd.DataFrame)
        assert imputed['col_a'].isnull().sum() == 0
    
    def test_iterative_imputation(self, analyzer):
        """Test iterative (MICE) imputation"""
        imputed = analyzer.iterative_imputation(max_iter=5)
        
        assert isinstance(imputed, pd.DataFrame)
        assert imputed['col_a'].isnull().sum() == 0
        assert imputed['col_b'].isnull().sum() == 0
        assert 'iterative' in analyzer.imputed_datasets
    
    def test_iterative_imputation_custom_iterations(self, analyzer):
        """Test iterative imputation with custom max iterations"""
        imputed = analyzer.iterative_imputation(max_iter=15)
        
        assert isinstance(imputed, pd.DataFrame)
        assert imputed['col_a'].isnull().sum() == 0
    
    def test_compare_imputation_methods(self, analyzer):
        """Test comparison of imputation methods"""
        # First apply multiple imputation methods
        analyzer.simple_imputation(strategy='mean')
        analyzer.simple_imputation(strategy='median')
        analyzer.knn_imputation()
        
        comparison = analyzer.compare_imputation_methods()
        
        assert isinstance(comparison, pd.DataFrame)
        assert 'original' in comparison.columns.get_level_values(0)
        assert 'simple_mean' in comparison.columns.get_level_values(0)
        assert 'simple_median' in comparison.columns.get_level_values(0)
        assert 'knn' in comparison.columns.get_level_values(0)
    
    def test_visualize_missing_data(self, analyzer):
        """Test missing data visualization"""
        with patch('matplotlib.pyplot.show'):
            fig = analyzer.visualize_missing_data(figsize=(12, 8))
            
            assert fig is not None
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
    
    def test_visualize_imputation_comparison(self, analyzer):
        """Test imputation comparison visualization"""
        # Apply some imputation methods first
        analyzer.simple_imputation(strategy='mean')
        analyzer.knn_imputation()
        
        with patch('matplotlib.pyplot.show'):
            fig = analyzer.visualize_imputation_comparison('col_a', figsize=(12, 8))
            
            assert fig is not None
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
    
    def test_visualize_imputation_comparison_invalid_column(self, analyzer, capsys):
        """Test visualization with invalid column"""
        analyzer.visualize_imputation_comparison('col_d')
        captured = capsys.readouterr()
        assert "not numeric" in captured.out or "doesn't exist" in captured.out
    
    def test_visualize_correlation_comparison(self, analyzer):
        """Test correlation comparison visualization"""
        # Apply some imputation methods first
        analyzer.simple_imputation(strategy='mean')
        analyzer.knn_imputation()
        
        with patch('matplotlib.pyplot.show'):
            fig = analyzer.visualize_correlation_comparison(figsize=(12, 8))
            
            assert fig is not None
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
    
    def test_imputation_preserves_shape(self, analyzer):
        """Test that imputation preserves DataFrame shape"""
        original_shape = analyzer.data.shape
        
        analyzer.simple_imputation(strategy='mean')
        imputed = analyzer.imputed_datasets['simple_mean']
        
        assert imputed.shape == original_shape
    
    def test_imputation_preserves_columns(self, analyzer):
        """Test that imputation preserves column names"""
        original_columns = list(analyzer.data.columns)
        
        analyzer.simple_imputation(strategy='mean')
        imputed = analyzer.imputed_datasets['simple_mean']
        
        assert list(imputed.columns) == original_columns
    
    def test_multiple_imputation_methods_storage(self, analyzer):
        """Test that multiple imputation methods are stored correctly"""
        analyzer.simple_imputation(strategy='mean')
        analyzer.simple_imputation(strategy='median')
        analyzer.knn_imputation()
        analyzer.iterative_imputation()
        
        assert len(analyzer.imputed_datasets) == 4
        assert 'simple_mean' in analyzer.imputed_datasets
        assert 'simple_median' in analyzer.imputed_datasets
        assert 'knn' in analyzer.imputed_datasets
        assert 'iterative' in analyzer.imputed_datasets
    
    def test_original_data_unchanged(self, analyzer):
        """Test that original data remains unchanged after operations"""
        original_copy = analyzer.original_data.copy()
        
        analyzer.simple_imputation(strategy='mean')
        analyzer.knn_imputation()
        analyzer.create_missing_indicators()
        
        pd.testing.assert_frame_equal(analyzer.original_data, original_copy)


class TestCreateSampleDataset:
    """Test suite for sample dataset creation"""
    
    def test_create_sample_dataset(self):
        """Test sample dataset creation"""
        df = create_sample_dataset_with_missing()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 200
        assert 'age' in df.columns
        assert 'income' in df.columns
        assert 'credit_score' in df.columns
        assert 'years_employed' in df.columns
        assert 'debt_ratio' in df.columns
    
    def test_sample_dataset_has_missing_values(self):
        """Test that sample dataset contains missing values"""
        df = create_sample_dataset_with_missing()
        
        assert df.isnull().sum().sum() > 0
        assert df['age'].isnull().sum() > 0
        assert df['income'].isnull().sum() > 0
        assert df['debt_ratio'].isnull().sum() > 0
    
    def test_sample_dataset_reproducibility(self):
        """Test that sample dataset creation is reproducible"""
        df1 = create_sample_dataset_with_missing()
        df2 = create_sample_dataset_with_missing()
        
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_sample_dataset_data_types(self):
        """Test that sample dataset has correct data types"""
        df = create_sample_dataset_with_missing()
        
        assert df['age'].dtype in [np.float64, np.float32]
        assert df['income'].dtype in [np.float64, np.float32]
        assert df['credit_score'].dtype in [np.float64, np.float32]
        assert df['years_employed'].dtype in [np.float64, np.float32]
        assert df['debt_ratio'].dtype in [np.float64, np.float32]


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_dataframe(self):
        """Test analyzer with empty DataFrame"""
        df = pd.DataFrame()
        analyzer = DataQualityAnalyzer(df)
        
        report = analyzer.assess_data_quality()
        assert report['total_rows'] == 0
        assert report['total_columns'] == 0
    
    def test_no_missing_values(self):
        """Test analyzer with data containing no missing values"""
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        analyzer = DataQualityAnalyzer(df)
        
        report = analyzer.assess_data_quality()
        assert all(count == 0 for count in report['missing_values'].values())
        
        indicators = analyzer.create_missing_indicators()
        assert len(indicators.columns) == 0
    
    def test_all_missing_column(self):
        """Test analyzer with column that is completely missing"""
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [np.nan, np.nan, np.nan, np.nan, np.nan]
        })
        analyzer = DataQualityAnalyzer(df)
        
        report = analyzer.assess_data_quality()
        assert report['missing_percentage']['b'] == 100.0
    
    def test_single_row_dataframe(self):
        """Test analyzer with single row DataFrame"""
        df = pd.DataFrame({
            'a': [1],
            'b': [np.nan]
        })
        analyzer = DataQualityAnalyzer(df)
        
        report = analyzer.assess_data_quality()
        assert report['total_rows'] == 1
        assert report['missing_percentage']['b'] == 100.0
    
    def test_categorical_only_dataframe(self):
        """Test analyzer with only categorical data"""
        df = pd.DataFrame({
            'cat1': ['a', 'b', np.nan, 'd'],
            'cat2': ['x', np.nan, 'z', 'w']
        })
        analyzer = DataQualityAnalyzer(df)
        
        imputed = analyzer.simple_imputation(strategy='most_frequent')
        assert imputed['cat1'].isnull().sum() == 0
        assert imputed['cat2'].isnull().sum() == 0


class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_complete_workflow(self):
        """Test complete analysis workflow"""
        # Create data
        df = create_sample_dataset_with_missing()
        analyzer = DataQualityAnalyzer(df)
        
        # Assess quality
        report = analyzer.assess_data_quality()
        assert report is not None
        
        # Create indicators
        indicators = analyzer.create_missing_indicators()
        assert indicators is not None
        
        # Apply imputations
        analyzer.simple_imputation(strategy='mean')
        analyzer.knn_imputation()
        analyzer.iterative_imputation()
        
        # Compare methods
        comparison = analyzer.compare_imputation_methods()
        assert comparison is not None
        assert len(analyzer.imputed_datasets) == 3
    
    def test_visualization_workflow(self):
        """Test complete visualization workflow"""
        df = create_sample_dataset_with_missing()
        analyzer = DataQualityAnalyzer(df)
        
        # Apply imputations
        analyzer.simple_imputation(strategy='mean')
        analyzer.knn_imputation()
        
        with patch('matplotlib.pyplot.show'):
            # Test all visualizations
            fig1 = analyzer.visualize_missing_data()
            assert fig1 is not None
            plt.close(fig1)
            
            fig2 = analyzer.visualize_imputation_comparison('income')
            assert fig2 is not None
            plt.close(fig2)
            
            fig3 = analyzer.visualize_correlation_comparison()
            assert fig3 is not None
            plt.close(fig3)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
