"""
Unit Tests for Task 2: Outlier Detection and Treatment

This module contains comprehensive unit tests for the OutlierAnalyzer class,
testing all major functionality including detection methods, treatment strategies,
and visualization capabilities.

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

from task2_outlier_detection import (
    OutlierAnalyzer,
    create_sample_dataset_with_outliers
)


class TestOutlierAnalyzer:
    """Test suite for OutlierAnalyzer class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing"""
        np.random.seed(42)
        data = pd.DataFrame({
            'col_a': np.concatenate([np.random.normal(50, 10, 95), [150, 160, 170, 180, 190]]),
            'col_b': np.concatenate([np.random.normal(100, 20, 97), [-50, -60, 200]]),
            'col_c': np.random.normal(200, 30, 100),
            'target': np.random.normal(75, 15, 100)
        })
        return data
    
    @pytest.fixture
    def analyzer(self, sample_data):
        """Create analyzer instance"""
        return OutlierAnalyzer(sample_data)
    
    def test_initialization(self, sample_data):
        """Test analyzer initialization"""
        analyzer = OutlierAnalyzer(sample_data)
        
        assert analyzer.data is not None
        assert analyzer.original_data is not None
        assert len(analyzer.data) == len(sample_data)
        assert isinstance(analyzer.outlier_masks, dict)
        assert isinstance(analyzer.treated_datasets, dict)
        assert len(analyzer.outlier_masks) == 0
        assert len(analyzer.treated_datasets) == 0
    
    def test_detect_outliers_zscore(self, analyzer):
        """Test Z-score outlier detection"""
        outliers = analyzer.detect_outliers_zscore(threshold=3.0)
        
        assert isinstance(outliers, pd.DataFrame)
        assert len(outliers) == len(analyzer.data)
        assert all(dtype == bool for dtype in outliers.dtypes)
        assert 'zscore' in analyzer.outlier_masks
        assert outliers.sum().sum() > 0  # Should detect some outliers
    
    def test_detect_outliers_zscore_custom_threshold(self, analyzer):
        """Test Z-score detection with custom threshold"""
        outliers_strict = analyzer.detect_outliers_zscore(threshold=2.0)
        outliers_lenient = analyzer.detect_outliers_zscore(threshold=4.0)
        
        # Stricter threshold should detect more outliers
        assert outliers_strict.sum().sum() >= outliers_lenient.sum().sum()
    
    def test_detect_outliers_iqr(self, analyzer):
        """Test IQR outlier detection"""
        outliers = analyzer.detect_outliers_iqr(multiplier=1.5)
        
        assert isinstance(outliers, pd.DataFrame)
        assert len(outliers) == len(analyzer.data)
        assert 'iqr' in analyzer.outlier_masks
        assert outliers.sum().sum() > 0
    
    def test_detect_outliers_iqr_custom_multiplier(self, analyzer):
        """Test IQR detection with custom multiplier"""
        outliers_strict = analyzer.detect_outliers_iqr(multiplier=1.0)
        outliers_lenient = analyzer.detect_outliers_iqr(multiplier=3.0)
        
        # Smaller multiplier should detect more outliers
        assert outliers_strict.sum().sum() >= outliers_lenient.sum().sum()
    
    def test_detect_outliers_isolation_forest(self, analyzer):
        """Test Isolation Forest outlier detection"""
        outliers = analyzer.detect_outliers_isolation_forest(contamination=0.1)
        
        assert isinstance(outliers, pd.DataFrame)
        assert len(outliers) == len(analyzer.data)
        assert 'isolation_forest' in analyzer.outlier_masks
    
    def test_detect_outliers_isolation_forest_contamination(self, analyzer):
        """Test Isolation Forest with different contamination levels"""
        outliers_low = analyzer.detect_outliers_isolation_forest(contamination=0.05)
        outliers_high = analyzer.detect_outliers_isolation_forest(contamination=0.15)
        
        assert isinstance(outliers_low, pd.DataFrame)
        assert isinstance(outliers_high, pd.DataFrame)
    
    def test_detect_outliers_lof(self, analyzer):
        """Test Local Outlier Factor detection"""
        outliers = analyzer.detect_outliers_lof(n_neighbors=20, contamination=0.1)
        
        assert isinstance(outliers, pd.DataFrame)
        assert len(outliers) == len(analyzer.data)
        assert 'lof' in analyzer.outlier_masks
    
    def test_detect_outliers_lof_neighbors(self, analyzer):
        """Test LOF with different neighbor counts"""
        outliers_few = analyzer.detect_outliers_lof(n_neighbors=5, contamination=0.1)
        outliers_many = analyzer.detect_outliers_lof(n_neighbors=30, contamination=0.1)
        
        assert isinstance(outliers_few, pd.DataFrame)
        assert isinstance(outliers_many, pd.DataFrame)
    
    def test_get_outlier_summary(self, analyzer):
        """Test outlier summary generation"""
        # Detect outliers using multiple methods
        analyzer.detect_outliers_zscore()
        analyzer.detect_outliers_iqr()
        analyzer.detect_outliers_isolation_forest()
        
        summary = analyzer.get_outlier_summary()
        
        assert isinstance(summary, pd.DataFrame)
        assert 'zscore' in summary.columns
        assert 'iqr' in summary.columns
        assert 'isolation_forest' in summary.columns
    
    def test_remove_outliers(self, analyzer):
        """Test outlier removal"""
        analyzer.detect_outliers_iqr()
        removed_data = analyzer.remove_outliers(method='iqr')
        
        assert isinstance(removed_data, pd.DataFrame)
        assert len(removed_data) <= len(analyzer.data)
        assert 'removed_iqr' in analyzer.treated_datasets
    
    def test_remove_outliers_reduces_size(self, analyzer):
        """Test that outlier removal reduces dataset size"""
        original_size = len(analyzer.data)
        analyzer.detect_outliers_iqr()
        removed_data = analyzer.remove_outliers(method='iqr')
        
        assert len(removed_data) < original_size
    
    def test_remove_outliers_invalid_method(self, analyzer):
        """Test error handling for invalid method"""
        with pytest.raises(ValueError):
            analyzer.remove_outliers(method='nonexistent')
    
    def test_cap_outliers(self, analyzer):
        """Test outlier capping (winsorization)"""
        capped_data = analyzer.cap_outliers(multiplier=1.5)
        
        assert isinstance(capped_data, pd.DataFrame)
        assert len(capped_data) == len(analyzer.data)
        assert 'capped' in analyzer.treated_datasets
    
    def test_cap_outliers_bounds(self, analyzer):
        """Test that capping keeps values within bounds"""
        original_max = analyzer.data['col_a'].max()
        capped_data = analyzer.cap_outliers(multiplier=1.5)
        capped_max = capped_data['col_a'].max()
        
        # Capped max should be less than or equal to original
        assert capped_max <= original_max
    
    def test_transform_outliers_log(self, analyzer):
        """Test log transformation"""
        log_data = analyzer.transform_outliers_log()
        
        assert isinstance(log_data, pd.DataFrame)
        assert len(log_data) == len(analyzer.data)
        assert 'log_transform' in analyzer.treated_datasets
    
    def test_transform_outliers_log_positive(self, analyzer):
        """Test that log transformation handles negative values"""
        # Add negative values
        analyzer.data.loc[0, 'col_a'] = -10
        log_data = analyzer.transform_outliers_log()
        
        # Should not have NaN or inf values
        assert not log_data['col_a'].isnull().any()
        assert not np.isinf(log_data['col_a']).any()
    
    def test_transform_outliers_boxcox(self, analyzer):
        """Test Box-Cox/Yeo-Johnson transformation"""
        boxcox_data = analyzer.transform_outliers_boxcox()
        
        assert isinstance(boxcox_data, pd.DataFrame)
        assert len(boxcox_data) == len(analyzer.data)
        assert 'boxcox_transform' in analyzer.treated_datasets
    
    def test_evaluate_model_performance_original(self, analyzer):
        """Test model performance evaluation on original data"""
        performance = analyzer.evaluate_model_performance('target', 'original')
        
        assert isinstance(performance, dict)
        assert 'r2_score' in performance
        assert 'r2_std' in performance
        assert 'mean_cv_score' in performance
    
    def test_evaluate_model_performance_treated(self, analyzer):
        """Test model performance evaluation on treated data"""
        analyzer.detect_outliers_iqr()
        analyzer.remove_outliers(method='iqr')
        
        performance = analyzer.evaluate_model_performance('target', 'removed_iqr')
        
        assert isinstance(performance, dict)
        assert 'r2_score' in performance
    
    def test_evaluate_model_performance_invalid_dataset(self, analyzer):
        """Test error handling for invalid dataset name"""
        with pytest.raises(ValueError):
            analyzer.evaluate_model_performance('target', 'nonexistent')
    
    def test_visualize_outliers_boxplot(self, analyzer):
        """Test box plot visualization"""
        with patch('matplotlib.pyplot.show'):
            fig = analyzer.visualize_outliers_boxplot(figsize=(12, 8))
            
            assert fig is not None
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
    
    def test_visualize_outliers_violin(self, analyzer):
        """Test violin plot visualization"""
        with patch('matplotlib.pyplot.show'):
            fig = analyzer.visualize_outliers_violin(figsize=(12, 8))
            
            assert fig is not None
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
    
    def test_visualize_outliers_scatter(self, analyzer):
        """Test scatter plot visualization"""
        analyzer.detect_outliers_zscore()
        analyzer.detect_outliers_iqr()
        
        with patch('matplotlib.pyplot.show'):
            fig = analyzer.visualize_outliers_scatter('col_a', 'col_b', figsize=(12, 6))
            
            assert fig is not None
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
    
    def test_visualize_outliers_scatter_invalid_columns(self, analyzer, capsys):
        """Test scatter plot with invalid columns"""
        analyzer.visualize_outliers_scatter('invalid', 'col_b')
        captured = capsys.readouterr()
        assert "not found" in captured.out
    
    def test_visualize_3d_outliers(self, analyzer):
        """Test 3D scatter plot visualization"""
        analyzer.detect_outliers_isolation_forest()
        
        with patch('matplotlib.pyplot.show'):
            fig = analyzer.visualize_3d_outliers('col_a', 'col_b', 'col_c', 
                                                 method='isolation_forest')
            
            assert fig is not None
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
    
    def test_visualize_3d_outliers_no_detection(self, analyzer, capsys):
        """Test 3D visualization without prior detection"""
        analyzer.visualize_3d_outliers('col_a', 'col_b', 'col_c', method='zscore')
        captured = capsys.readouterr()
        assert "not found" in captured.out
    
    def test_visualize_treatment_comparison(self, analyzer):
        """Test treatment comparison visualization"""
        analyzer.cap_outliers()
        analyzer.transform_outliers_log()
        
        with patch('matplotlib.pyplot.show'):
            fig = analyzer.visualize_treatment_comparison('col_a', figsize=(12, 8))
            
            assert fig is not None
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
    
    def test_visualize_treatment_comparison_invalid_column(self, analyzer, capsys):
        """Test treatment comparison with invalid column"""
        analyzer.visualize_treatment_comparison('invalid')
        captured = capsys.readouterr()
        assert "not found" in captured.out
    
    def test_visualize_qq_plots(self, analyzer):
        """Test Q-Q plot visualization"""
        with patch('matplotlib.pyplot.show'):
            fig = analyzer.visualize_qq_plots(figsize=(12, 8))
            
            assert fig is not None
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
    
    def test_multiple_detection_methods(self, analyzer):
        """Test running multiple detection methods"""
        analyzer.detect_outliers_zscore()
        analyzer.detect_outliers_iqr()
        analyzer.detect_outliers_isolation_forest()
        analyzer.detect_outliers_lof()
        
        assert len(analyzer.outlier_masks) == 4
        assert 'zscore' in analyzer.outlier_masks
        assert 'iqr' in analyzer.outlier_masks
        assert 'isolation_forest' in analyzer.outlier_masks
        assert 'lof' in analyzer.outlier_masks
    
    def test_multiple_treatment_methods(self, analyzer):
        """Test applying multiple treatment methods"""
        analyzer.detect_outliers_iqr()
        analyzer.remove_outliers(method='iqr')
        analyzer.cap_outliers()
        analyzer.transform_outliers_log()
        analyzer.transform_outliers_boxcox()
        
        assert len(analyzer.treated_datasets) == 4
    
    def test_original_data_unchanged(self, analyzer):
        """Test that original data remains unchanged after operations"""
        original_copy = analyzer.original_data.copy()
        
        analyzer.detect_outliers_zscore()
        analyzer.detect_outliers_iqr()
        analyzer.cap_outliers()
        analyzer.transform_outliers_log()
        
        pd.testing.assert_frame_equal(analyzer.original_data, original_copy)
    
    def test_data_shape_preservation(self, analyzer):
        """Test that treatments preserve shape (except removal)"""
        original_shape = analyzer.data.shape
        
        analyzer.cap_outliers()
        assert analyzer.treated_datasets['capped'].shape == original_shape
        
        analyzer.transform_outliers_log()
        assert analyzer.treated_datasets['log_transform'].shape == original_shape
        
        analyzer.transform_outliers_boxcox()
        assert analyzer.treated_datasets['boxcox_transform'].shape == original_shape


class TestCreateSampleDataset:
    """Test suite for sample dataset creation"""
    
    def test_create_sample_dataset(self):
        """Test sample dataset creation"""
        df = create_sample_dataset_with_outliers()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 300
        assert 'feature1' in df.columns
        assert 'feature2' in df.columns
        assert 'feature3' in df.columns
        assert 'feature4' in df.columns
        assert 'target' in df.columns
    
    def test_sample_dataset_has_outliers(self):
        """Test that sample dataset contains outliers"""
        df = create_sample_dataset_with_outliers()
        analyzer = OutlierAnalyzer(df)
        
        outliers = analyzer.detect_outliers_zscore(threshold=3.0)
        assert outliers.sum().sum() > 0
    
    def test_sample_dataset_reproducibility(self):
        """Test that sample dataset creation is reproducible"""
        df1 = create_sample_dataset_with_outliers()
        df2 = create_sample_dataset_with_outliers()
        
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_sample_dataset_data_types(self):
        """Test that sample dataset has correct data types"""
        df = create_sample_dataset_with_outliers()
        
        for col in df.columns:
            assert df[col].dtype in [np.float64, np.float32, np.int64, np.int32]


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_dataframe(self):
        """Test analyzer with empty DataFrame"""
        df = pd.DataFrame()
        analyzer = OutlierAnalyzer(df)
        
        assert len(analyzer.data) == 0
    
    def test_single_column_dataframe(self):
        """Test analyzer with single column"""
        df = pd.DataFrame({'col': [1, 2, 3, 4, 100]})
        analyzer = OutlierAnalyzer(df)
        
        outliers = analyzer.detect_outliers_zscore()
        assert isinstance(outliers, pd.DataFrame)
    
    def test_no_outliers(self):
        """Test analyzer with data containing no outliers"""
        np.random.seed(42)
        df = pd.DataFrame({
            'a': np.random.normal(50, 5, 100),
            'b': np.random.normal(100, 10, 100)
        })
        analyzer = OutlierAnalyzer(df)
        
        outliers = analyzer.detect_outliers_zscore(threshold=5.0)  # Very lenient
        # Should detect very few or no outliers
        assert outliers.sum().sum() < 5
    
    def test_all_same_values(self):
        """Test analyzer with column of identical values"""
        df = pd.DataFrame({
            'a': [5.0] * 100,
            'b': np.random.normal(50, 10, 100)
        })
        analyzer = OutlierAnalyzer(df)
        
        outliers = analyzer.detect_outliers_zscore()
        # Column with no variance should have no outliers
        assert outliers['a'].sum() == 0
    
    def test_single_row_dataframe(self):
        """Test analyzer with single row"""
        df = pd.DataFrame({'a': [1], 'b': [2]})
        analyzer = OutlierAnalyzer(df)
        
        outliers = analyzer.detect_outliers_zscore()
        assert len(outliers) == 1
    
    def test_categorical_columns_ignored(self):
        """Test that categorical columns are handled properly"""
        df = pd.DataFrame({
            'numeric': [1, 2, 3, 100],
            'categorical': ['a', 'b', 'c', 'd']
        })
        analyzer = OutlierAnalyzer(df)
        
        outliers = analyzer.detect_outliers_zscore()
        assert 'numeric' in outliers.columns
        assert 'categorical' not in outliers.columns


class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_complete_workflow(self):
        """Test complete analysis workflow"""
        # Create data
        df = create_sample_dataset_with_outliers()
        analyzer = OutlierAnalyzer(df)
        
        # Detect outliers
        analyzer.detect_outliers_zscore()
        analyzer.detect_outliers_iqr()
        analyzer.detect_outliers_isolation_forest()
        
        # Get summary
        summary = analyzer.get_outlier_summary()
        assert summary is not None
        
        # Apply treatments
        analyzer.remove_outliers(method='iqr')
        analyzer.cap_outliers()
        analyzer.transform_outliers_log()
        
        assert len(analyzer.treated_datasets) == 3
    
    def test_visualization_workflow(self):
        """Test complete visualization workflow"""
        df = create_sample_dataset_with_outliers()
        analyzer = OutlierAnalyzer(df)
        
        # Detect outliers
        analyzer.detect_outliers_zscore()
        analyzer.detect_outliers_iqr()
        analyzer.detect_outliers_isolation_forest()
        
        # Apply treatments
        analyzer.cap_outliers()
        analyzer.transform_outliers_log()
        
        with patch('matplotlib.pyplot.show'):
            # Test all visualizations
            fig1 = analyzer.visualize_outliers_boxplot()
            assert fig1 is not None
            plt.close(fig1)
            
            fig2 = analyzer.visualize_outliers_violin()
            assert fig2 is not None
            plt.close(fig2)
            
            fig3 = analyzer.visualize_outliers_scatter('feature1', 'feature2')
            assert fig3 is not None
            plt.close(fig3)
            
            fig4 = analyzer.visualize_3d_outliers('feature1', 'feature2', 'feature3',
                                                  method='isolation_forest')
            assert fig4 is not None
            plt.close(fig4)
            
            fig5 = analyzer.visualize_treatment_comparison('feature1')
            assert fig5 is not None
            plt.close(fig5)
            
            fig6 = analyzer.visualize_qq_plots()
            assert fig6 is not None
            plt.close(fig6)
    
    def test_performance_evaluation_workflow(self):
        """Test model performance evaluation workflow"""
        df = create_sample_dataset_with_outliers()
        analyzer = OutlierAnalyzer(df)
        
        # Apply different treatments
        analyzer.detect_outliers_iqr()
        analyzer.remove_outliers(method='iqr')
        analyzer.cap_outliers()
        
        # Evaluate performance
        perf_original = analyzer.evaluate_model_performance('target', 'original')
        perf_removed = analyzer.evaluate_model_performance('target', 'removed_iqr')
        perf_capped = analyzer.evaluate_model_performance('target', 'capped')
        
        assert perf_original is not None
        assert perf_removed is not None
        assert perf_capped is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
