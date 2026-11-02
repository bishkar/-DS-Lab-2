"""
Unit Tests for Task 3: Comprehensive Data Scaling and Normalization Study

This module contains comprehensive unit tests for the ScalingAnalyzer class,
testing all scaling methods, model evaluation, and visualization capabilities.

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

from task3_scaling import (
    ScalingAnalyzer,
    create_sample_dataset
)


class TestScalingAnalyzer:
    """Test suite for ScalingAnalyzer class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing"""
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.random.uniform(0, 100, 100),
            'feature2': np.random.normal(500, 100, 100),
            'feature3': np.random.exponential(10, 100),
            'target': np.random.choice([0, 1], 100)
        })
        return data
    
    @pytest.fixture
    def analyzer(self, sample_data):
        """Create analyzer instance"""
        return ScalingAnalyzer(sample_data, target_col='target')
    
    def test_initialization(self, sample_data):
        """Test analyzer initialization"""
        analyzer = ScalingAnalyzer(sample_data, target_col='target')
        
        assert analyzer.data is not None
        assert analyzer.original_data is not None
        assert len(analyzer.data) == len(sample_data)
        assert analyzer.target_col == 'target'
        assert isinstance(analyzer.scaled_datasets, dict)
        assert isinstance(analyzer.scalers, dict)
        assert len(analyzer.scaled_datasets) == 0
        assert len(analyzer.scalers) == 0
    
    def test_initialization_no_target(self, sample_data):
        """Test initialization without target column"""
        analyzer = ScalingAnalyzer(sample_data)
        
        assert analyzer.target_col is None
    
    def test_apply_minmax_scaling(self, analyzer):
        """Test Min-Max scaling"""
        scaled_data = analyzer.apply_minmax_scaling(feature_range=(0, 1))
        
        assert isinstance(scaled_data, pd.DataFrame)
        assert len(scaled_data) == len(analyzer.data)
        assert 'minmax' in analyzer.scaled_datasets
        assert 'minmax' in analyzer.scalers
        
        # Check that values are in range [0, 1]
        for col in ['feature1', 'feature2', 'feature3']:
            assert scaled_data[col].min() >= 0
            assert scaled_data[col].max() <= 1
    
    def test_apply_minmax_scaling_custom_range(self, analyzer):
        """Test Min-Max scaling with custom range"""
        scaled_data = analyzer.apply_minmax_scaling(feature_range=(-1, 1))
        
        # Check that values are in range [-1, 1] (with small tolerance for floating point)
        for col in ['feature1', 'feature2', 'feature3']:
            assert scaled_data[col].min() >= -1 - 1e-10
            assert scaled_data[col].max() <= 1 + 1e-10
    
    def test_apply_standard_scaling(self, analyzer):
        """Test Standard (Z-score) scaling"""
        scaled_data = analyzer.apply_standard_scaling()
        
        assert isinstance(scaled_data, pd.DataFrame)
        assert len(scaled_data) == len(analyzer.data)
        assert 'standard' in analyzer.scaled_datasets
        assert 'standard' in analyzer.scalers
        
        # Check that mean is close to 0 and std is close to 1
        for col in ['feature1', 'feature2', 'feature3']:
            assert abs(scaled_data[col].mean()) < 0.1
            assert abs(scaled_data[col].std() - 1.0) < 0.1
    
    def test_apply_robust_scaling(self, analyzer):
        """Test Robust scaling"""
        scaled_data = analyzer.apply_robust_scaling()
        
        assert isinstance(scaled_data, pd.DataFrame)
        assert len(scaled_data) == len(analyzer.data)
        assert 'robust' in analyzer.scaled_datasets
        assert 'robust' in analyzer.scalers
    
    def test_apply_maxabs_scaling(self, analyzer):
        """Test MaxAbs scaling"""
        scaled_data = analyzer.apply_maxabs_scaling()
        
        assert isinstance(scaled_data, pd.DataFrame)
        assert len(scaled_data) == len(analyzer.data)
        assert 'maxabs' in analyzer.scaled_datasets
        assert 'maxabs' in analyzer.scalers
        
        # Check that values are in range [-1, 1]
        for col in ['feature1', 'feature2', 'feature3']:
            assert scaled_data[col].min() >= -1
            assert scaled_data[col].max() <= 1
    
    def test_apply_power_transform(self, analyzer):
        """Test Power transformation"""
        scaled_data = analyzer.apply_power_transform(method='yeo-johnson')
        
        assert isinstance(scaled_data, pd.DataFrame)
        assert len(scaled_data) == len(analyzer.data)
        assert 'power' in analyzer.scaled_datasets
        assert 'power' in analyzer.scalers
    
    def test_target_column_not_scaled(self, analyzer):
        """Test that target column is not scaled"""
        original_target = analyzer.data['target'].copy()
        
        analyzer.apply_minmax_scaling()
        scaled_data = analyzer.scaled_datasets['minmax']
        
        # Target should remain unchanged
        pd.testing.assert_series_equal(scaled_data['target'], original_target, check_names=False)
    
    def test_get_scaling_summary(self, analyzer):
        """Test scaling summary generation"""
        analyzer.apply_minmax_scaling()
        analyzer.apply_standard_scaling()
        analyzer.apply_robust_scaling()
        
        summary = analyzer.get_scaling_summary()
        
        assert isinstance(summary, pd.DataFrame)
        assert 'original' in summary.columns.get_level_values(0)
        assert 'minmax' in summary.columns.get_level_values(0)
        assert 'standard' in summary.columns.get_level_values(0)
        assert 'robust' in summary.columns.get_level_values(0)
    
    def test_evaluate_model_performance_original(self, analyzer):
        """Test model performance evaluation on original data"""
        performance = analyzer.evaluate_model_performance('linear_regression', 'original')
        
        assert isinstance(performance, dict)
        assert 'cv_score' in performance
        assert 'cv_std' in performance
    
    def test_evaluate_model_performance_scaled(self, analyzer):
        """Test model performance evaluation on scaled data"""
        analyzer.apply_standard_scaling()
        performance = analyzer.evaluate_model_performance('linear_regression', 'standard')
        
        assert isinstance(performance, dict)
        assert 'cv_score' in performance
        assert 'cv_std' in performance
    
    def test_evaluate_model_performance_knn(self, analyzer):
        """Test KNN model evaluation"""
        performance = analyzer.evaluate_model_performance('knn', 'original')
        
        assert isinstance(performance, dict)
        assert 'cv_score' in performance
    
    def test_evaluate_model_performance_invalid_dataset(self, analyzer):
        """Test error handling for invalid dataset"""
        with pytest.raises(ValueError):
            analyzer.evaluate_model_performance('linear_regression', 'nonexistent')
    
    def test_compare_model_performance(self, analyzer):
        """Test model performance comparison"""
        analyzer.apply_minmax_scaling()
        analyzer.apply_standard_scaling()
        
        comparison = analyzer.compare_model_performance(model_types=['linear_regression'])
        
        assert isinstance(comparison, pd.DataFrame)
        assert 'model' in comparison.columns
        assert 'scaling' in comparison.columns
        assert 'score' in comparison.columns
        assert 'std' in comparison.columns
        assert len(comparison) > 0
    
    def test_compare_model_performance_multiple_models(self, analyzer):
        """Test comparison with multiple models"""
        analyzer.apply_minmax_scaling()
        
        comparison = analyzer.compare_model_performance(model_types=['linear_regression', 'knn'])
        
        assert len(comparison['model'].unique()) == 2
    
    def test_visualize_distributions(self, analyzer):
        """Test distribution visualization"""
        analyzer.apply_minmax_scaling()
        analyzer.apply_standard_scaling()
        
        with patch('matplotlib.pyplot.show'):
            fig = analyzer.visualize_distributions(figsize=(12, 8))
            
            assert fig is not None
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
    
    def test_visualize_boxplots(self, analyzer):
        """Test box plot visualization"""
        analyzer.apply_minmax_scaling()
        analyzer.apply_standard_scaling()
        
        with patch('matplotlib.pyplot.show'):
            fig = analyzer.visualize_boxplots(figsize=(12, 8))
            
            assert fig is not None
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
    
    def test_visualize_correlation_comparison(self, analyzer):
        """Test correlation comparison visualization"""
        analyzer.apply_minmax_scaling()
        analyzer.apply_standard_scaling()
        
        with patch('matplotlib.pyplot.show'):
            fig = analyzer.visualize_correlation_comparison(figsize=(12, 8))
            
            assert fig is not None
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
    
    def test_visualize_performance_comparison(self, analyzer):
        """Test performance comparison visualization"""
        analyzer.apply_minmax_scaling()
        comparison = analyzer.compare_model_performance(model_types=['linear_regression'])
        
        with patch('matplotlib.pyplot.show'):
            fig = analyzer.visualize_performance_comparison(comparison, figsize=(12, 6))
            
            assert fig is not None
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
    
    def test_visualize_feature_space_2d(self, analyzer):
        """Test 2D feature space visualization"""
        analyzer.apply_minmax_scaling()
        analyzer.apply_standard_scaling()
        
        with patch('matplotlib.pyplot.show'):
            fig = analyzer.visualize_feature_space_2d('feature1', 'feature2', figsize=(12, 4))
            
            assert fig is not None
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
    
    def test_visualize_feature_space_2d_invalid_features(self, analyzer, capsys):
        """Test 2D visualization with invalid features"""
        analyzer.visualize_feature_space_2d('invalid1', 'feature2')
        captured = capsys.readouterr()
        assert "not found" in captured.out
    
    def test_visualize_feature_space_3d(self, analyzer):
        """Test 3D feature space visualization"""
        analyzer.apply_minmax_scaling()
        analyzer.apply_standard_scaling()
        
        with patch('matplotlib.pyplot.show'):
            fig = analyzer.visualize_feature_space_3d('feature1', 'feature2', 'feature3', 
                                                     figsize=(12, 4))
            
            assert fig is not None
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
    
    def test_multiple_scaling_methods(self, analyzer):
        """Test applying multiple scaling methods"""
        analyzer.apply_minmax_scaling()
        analyzer.apply_standard_scaling()
        analyzer.apply_robust_scaling()
        analyzer.apply_maxabs_scaling()
        analyzer.apply_power_transform()
        
        assert len(analyzer.scaled_datasets) == 5
        assert len(analyzer.scalers) == 5
    
    def test_original_data_unchanged(self, analyzer):
        """Test that original data remains unchanged"""
        original_copy = analyzer.original_data.copy()
        
        analyzer.apply_minmax_scaling()
        analyzer.apply_standard_scaling()
        analyzer.apply_robust_scaling()
        
        pd.testing.assert_frame_equal(analyzer.original_data, original_copy)
    
    def test_data_shape_preservation(self, analyzer):
        """Test that scaling preserves data shape"""
        original_shape = analyzer.data.shape
        
        analyzer.apply_minmax_scaling()
        assert analyzer.scaled_datasets['minmax'].shape == original_shape
        
        analyzer.apply_standard_scaling()
        assert analyzer.scaled_datasets['standard'].shape == original_shape
    
    def test_column_names_preservation(self, analyzer):
        """Test that scaling preserves column names"""
        original_columns = list(analyzer.data.columns)
        
        analyzer.apply_minmax_scaling()
        assert list(analyzer.scaled_datasets['minmax'].columns) == original_columns


class TestCreateSampleDataset:
    """Test suite for sample dataset creation"""
    
    def test_create_sample_dataset(self):
        """Test sample dataset creation"""
        df, target_col = create_sample_dataset()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 200
        assert target_col == 'approved'
        assert target_col in df.columns
        assert 'age' in df.columns
        assert 'income' in df.columns
        assert 'credit_score' in df.columns
    
    def test_sample_dataset_target_binary(self):
        """Test that target is binary"""
        df, target_col = create_sample_dataset()
        
        unique_values = df[target_col].unique()
        assert len(unique_values) == 2
        assert set(unique_values) == {0, 1}
    
    def test_sample_dataset_reproducibility(self):
        """Test reproducibility of dataset creation"""
        df1, target1 = create_sample_dataset()
        df2, target2 = create_sample_dataset()
        
        pd.testing.assert_frame_equal(df1, df2)
        assert target1 == target2
    
    def test_sample_dataset_different_scales(self):
        """Test that features have different scales"""
        df, target_col = create_sample_dataset()
        
        # Check that features have different ranges
        feature_cols = [col for col in df.columns if col != target_col]
        ranges = {col: df[col].max() - df[col].min() for col in feature_cols}
        
        # Should have at least 2 different orders of magnitude
        assert max(ranges.values()) / min(ranges.values()) > 10


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_dataframe(self):
        """Test analyzer with empty DataFrame"""
        df = pd.DataFrame()
        analyzer = ScalingAnalyzer(df)
        
        assert len(analyzer.data) == 0
    
    def test_single_column_dataframe(self):
        """Test analyzer with single numeric column"""
        df = pd.DataFrame({'col': [1, 2, 3, 4, 5]})
        analyzer = ScalingAnalyzer(df)
        
        scaled = analyzer.apply_minmax_scaling()
        assert isinstance(scaled, pd.DataFrame)
        assert len(scaled) == 5
    
    def test_single_row_dataframe(self):
        """Test analyzer with single row"""
        df = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
        analyzer = ScalingAnalyzer(df)
        
        scaled = analyzer.apply_minmax_scaling()
        assert len(scaled) == 1
    
    def test_constant_feature(self):
        """Test scaling with constant feature"""
        df = pd.DataFrame({
            'constant': [5.0] * 100,
            'varying': np.random.normal(50, 10, 100),
            'target': np.random.choice([0, 1], 100)
        })
        analyzer = ScalingAnalyzer(df, target_col='target')
        
        # Should not raise error
        scaled = analyzer.apply_standard_scaling()
        assert isinstance(scaled, pd.DataFrame)
    
    def test_no_numeric_features(self):
        """Test analyzer with no numeric features except target"""
        df = pd.DataFrame({
            'category': ['a', 'b', 'c'] * 33 + ['a'],
            'target': np.random.choice([0, 1], 100)
        })
        analyzer = ScalingAnalyzer(df, target_col='target')
        
        scaled = analyzer.apply_minmax_scaling()
        # Should return data unchanged (or with only target)
        assert isinstance(scaled, pd.DataFrame)
    
    def test_missing_values(self):
        """Test handling of missing values"""
        df = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': [10, np.nan, 30, 40, 50],
            'target': [0, 1, 0, 1, 0]
        })
        analyzer = ScalingAnalyzer(df, target_col='target')
        
        # Should handle NaN values gracefully
        scaled = analyzer.apply_standard_scaling()
        assert isinstance(scaled, pd.DataFrame)
    
    def test_negative_values(self):
        """Test scaling with negative values"""
        df = pd.DataFrame({
            'feature1': np.random.normal(0, 10, 100),
            'feature2': np.random.uniform(-100, 100, 100),
            'target': np.random.choice([0, 1], 100)
        })
        analyzer = ScalingAnalyzer(df, target_col='target')
        
        scaled = analyzer.apply_minmax_scaling()
        assert isinstance(scaled, pd.DataFrame)
        # MinMax should still work with negative values
        assert scaled['feature1'].min() >= 0
        assert scaled['feature1'].max() <= 1


class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_complete_workflow(self):
        """Test complete analysis workflow"""
        df, target_col = create_sample_dataset()
        analyzer = ScalingAnalyzer(df, target_col=target_col)
        
        # Apply all scaling methods
        analyzer.apply_minmax_scaling()
        analyzer.apply_standard_scaling()
        analyzer.apply_robust_scaling()
        analyzer.apply_maxabs_scaling()
        analyzer.apply_power_transform()
        
        # Get summary
        summary = analyzer.get_scaling_summary()
        assert summary is not None
        
        # Compare performance
        comparison = analyzer.compare_model_performance(model_types=['linear_regression'])
        assert comparison is not None
        assert len(comparison) > 0
    
    def test_visualization_workflow(self):
        """Test complete visualization workflow"""
        df, target_col = create_sample_dataset()
        analyzer = ScalingAnalyzer(df, target_col=target_col)
        
        # Apply scaling
        analyzer.apply_minmax_scaling()
        analyzer.apply_standard_scaling()
        
        with patch('matplotlib.pyplot.show'):
            # Test all visualizations
            fig1 = analyzer.visualize_distributions()
            assert fig1 is not None
            plt.close(fig1)
            
            fig2 = analyzer.visualize_boxplots()
            assert fig2 is not None
            plt.close(fig2)
            
            fig3 = analyzer.visualize_correlation_comparison()
            assert fig3 is not None
            plt.close(fig3)
            
            comparison = analyzer.compare_model_performance()
            fig4 = analyzer.visualize_performance_comparison(comparison)
            assert fig4 is not None
            plt.close(fig4)
            
            fig5 = analyzer.visualize_feature_space_2d('age', 'income')
            assert fig5 is not None
            plt.close(fig5)
            
            fig6 = analyzer.visualize_feature_space_3d('age', 'income', 'credit_score')
            assert fig6 is not None
            plt.close(fig6)
    
    def test_performance_evaluation_workflow(self):
        """Test model performance evaluation workflow"""
        df, target_col = create_sample_dataset()
        analyzer = ScalingAnalyzer(df, target_col=target_col)
        
        # Apply different scaling methods
        analyzer.apply_minmax_scaling()
        analyzer.apply_standard_scaling()
        
        # Evaluate with different models
        comparison = analyzer.compare_model_performance(
            model_types=['linear_regression', 'knn']
        )
        
        assert len(comparison) >= 6  # 2 models * 3 datasets (original + 2 scaled)
        assert set(comparison['model'].unique()) == {'linear_regression', 'knn'}
    
    def test_scaler_reusability(self):
        """Test that scalers can be reused"""
        df, target_col = create_sample_dataset()
        analyzer = ScalingAnalyzer(df, target_col=target_col)
        
        # Apply scaling
        analyzer.apply_minmax_scaling()
        scaler = analyzer.scalers['minmax']
        
        # Scaler should be fitted and reusable
        assert hasattr(scaler, 'data_min_')
        assert hasattr(scaler, 'data_max_')
        
        # Test transform on new data
        new_data = df.iloc[:10].copy()
        numeric_cols = [col for col in new_data.select_dtypes(include=[np.number]).columns 
                       if col != target_col]
        transformed = scaler.transform(new_data[numeric_cols])
        assert transformed.shape[0] == 10


class TestScalingProperties:
    """Test mathematical properties of scaling methods"""
    
    def test_minmax_range(self):
        """Test that MinMax scaling produces correct range"""
        df = pd.DataFrame({
            'feature': [0, 25, 50, 75, 100],
            'target': [0, 0, 1, 1, 0]
        })
        analyzer = ScalingAnalyzer(df, target_col='target')
        
        scaled = analyzer.apply_minmax_scaling(feature_range=(0, 1))
        
        assert scaled['feature'].min() == 0
        assert scaled['feature'].max() == 1
    
    def test_standard_scaling_properties(self):
        """Test that Standard scaling produces mean=0, std=1"""
        df = pd.DataFrame({
            'feature': np.random.normal(100, 20, 1000),
            'target': np.random.choice([0, 1], 1000)
        })
        analyzer = ScalingAnalyzer(df, target_col='target')
        
        scaled = analyzer.apply_standard_scaling()
        
        assert abs(scaled['feature'].mean()) < 0.01
        assert abs(scaled['feature'].std() - 1.0) < 0.01
    
    def test_scaling_reversibility(self):
        """Test that scaling can be reversed"""
        df = pd.DataFrame({
            'feature': [1, 2, 3, 4, 5],
            'target': [0, 1, 0, 1, 0]
        })
        analyzer = ScalingAnalyzer(df, target_col='target')
        
        analyzer.apply_minmax_scaling()
        scaler = analyzer.scalers['minmax']
        scaled_data = analyzer.scaled_datasets['minmax']
        
        # Inverse transform
        original_values = scaler.inverse_transform(scaled_data[['feature']])
        np.testing.assert_array_almost_equal(original_values.flatten(), df['feature'].values)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
