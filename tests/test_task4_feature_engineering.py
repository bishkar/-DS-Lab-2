"""
Unit tests for Task 4: Advanced Feature Engineering Pipeline
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from task4_feature_engineering import (
    FeatureEngineeringPipeline,
    create_sample_dataset,
    visualize_feature_importance,
    visualize_feature_correlations,
    visualize_pairplot,
    visualize_polynomial_impact,
    visualize_encoding_comparison,
    visualize_feature_network
)


class TestFeatureEngineeringPipeline:
    """Test cases for FeatureEngineeringPipeline class"""
    
    def test_initialization(self):
        """Test pipeline initialization"""
        pipeline = FeatureEngineeringPipeline(random_state=42)
        assert pipeline.random_state == 42
        assert isinstance(pipeline.feature_catalog, dict)
        assert len(pipeline.feature_catalog) == 0
        assert isinstance(pipeline.encoders, dict)
        assert isinstance(pipeline.engineered_features, list)
    
    def test_create_mathematical_features_log(self):
        """Test logarithmic transformation"""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        
        pipeline = FeatureEngineeringPipeline()
        result = pipeline.create_mathematical_features(
            df, columns=['A'], operations=['log']
        )
        
        assert 'A_log' in result.columns
        assert len(result) == len(df)
        assert 'A_log' in pipeline.engineered_features
        assert 'A_log' in pipeline.feature_catalog
    
    def test_create_mathematical_features_sqrt(self):
        """Test square root transformation"""
        df = pd.DataFrame({
            'A': [1, 4, 9, 16, 25],
            'B': [10, 20, 30, 40, 50]
        })
        
        pipeline = FeatureEngineeringPipeline()
        result = pipeline.create_mathematical_features(
            df, columns=['A'], operations=['sqrt']
        )
        
        assert 'A_sqrt' in result.columns
        np.testing.assert_array_almost_equal(
            result['A_sqrt'].values,
            np.array([1, 2, 3, 4, 5])
        )
    
    def test_create_mathematical_features_square(self):
        """Test square transformation"""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        
        pipeline = FeatureEngineeringPipeline()
        result = pipeline.create_mathematical_features(
            df, columns=['A'], operations=['square']
        )
        
        assert 'A_square' in result.columns
        np.testing.assert_array_equal(
            result['A_square'].values,
            np.array([1, 4, 9, 16, 25])
        )
    
    def test_create_mathematical_features_cube(self):
        """Test cube transformation"""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5]
        })
        
        pipeline = FeatureEngineeringPipeline()
        result = pipeline.create_mathematical_features(
            df, columns=['A'], operations=['cube']
        )
        
        assert 'A_cube' in result.columns
        np.testing.assert_array_equal(
            result['A_cube'].values,
            np.array([1, 8, 27, 64, 125])
        )
    
    def test_create_mathematical_features_negative_values(self):
        """Test mathematical transformations with negative values"""
        df = pd.DataFrame({
            'A': [-5, -2, 0, 2, 5]
        })
        
        pipeline = FeatureEngineeringPipeline()
        result = pipeline.create_mathematical_features(
            df, columns=['A'], operations=['log', 'sqrt']
        )
        
        # Should handle negative values by adding offset
        assert 'A_log' in result.columns
        assert 'A_sqrt' in result.columns
        assert not result['A_log'].isna().any()
        assert not result['A_sqrt'].isna().any()
    
    def test_create_mathematical_features_multiple_operations(self):
        """Test multiple operations at once"""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5]
        })
        
        pipeline = FeatureEngineeringPipeline()
        result = pipeline.create_mathematical_features(
            df, columns=['A'], operations=['log', 'sqrt', 'square', 'cube']
        )
        
        assert 'A_log' in result.columns
        assert 'A_sqrt' in result.columns
        assert 'A_square' in result.columns
        assert 'A_cube' in result.columns
        assert len(pipeline.engineered_features) == 4
    
    def test_create_interaction_features_multiply(self):
        """Test multiplication interaction"""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [2, 3, 4, 5, 6]
        })
        
        pipeline = FeatureEngineeringPipeline()
        result = pipeline.create_interaction_features(
            df, column_pairs=[('A', 'B')], operations=['multiply']
        )
        
        assert 'A_x_B' in result.columns
        np.testing.assert_array_equal(
            result['A_x_B'].values,
            df['A'].values * df['B'].values
        )
    
    def test_create_interaction_features_add(self):
        """Test addition interaction"""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [2, 3, 4, 5, 6]
        })
        
        pipeline = FeatureEngineeringPipeline()
        result = pipeline.create_interaction_features(
            df, column_pairs=[('A', 'B')], operations=['add']
        )
        
        assert 'A_plus_B' in result.columns
        np.testing.assert_array_equal(
            result['A_plus_B'].values,
            df['A'].values + df['B'].values
        )
    
    def test_create_interaction_features_subtract(self):
        """Test subtraction interaction"""
        df = pd.DataFrame({
            'A': [5, 6, 7, 8, 9],
            'B': [2, 3, 4, 5, 6]
        })
        
        pipeline = FeatureEngineeringPipeline()
        result = pipeline.create_interaction_features(
            df, column_pairs=[('A', 'B')], operations=['subtract']
        )
        
        assert 'A_minus_B' in result.columns
        np.testing.assert_array_equal(
            result['A_minus_B'].values,
            df['A'].values - df['B'].values
        )
    
    def test_create_interaction_features_divide(self):
        """Test division interaction"""
        df = pd.DataFrame({
            'A': [10, 20, 30, 40, 50],
            'B': [2, 4, 5, 8, 10]
        })
        
        pipeline = FeatureEngineeringPipeline()
        result = pipeline.create_interaction_features(
            df, column_pairs=[('A', 'B')], operations=['divide']
        )
        
        assert 'A_div_B' in result.columns
        assert not result['A_div_B'].isna().any()
    
    def test_create_interaction_features_multiple_pairs(self):
        """Test interactions with multiple column pairs"""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [2, 3, 4, 5, 6],
            'C': [3, 4, 5, 6, 7]
        })
        
        pipeline = FeatureEngineeringPipeline()
        result = pipeline.create_interaction_features(
            df,
            column_pairs=[('A', 'B'), ('A', 'C'), ('B', 'C')],
            operations=['multiply']
        )
        
        assert 'A_x_B' in result.columns
        assert 'A_x_C' in result.columns
        assert 'B_x_C' in result.columns
    
    def test_create_aggregation_features_mean(self):
        """Test mean aggregation"""
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'C'],
            'value': [10, 20, 30, 40, 50]
        })
        
        pipeline = FeatureEngineeringPipeline()
        result = pipeline.create_aggregation_features(
            df,
            group_by='group',
            agg_columns=['value'],
            agg_functions=['mean']
        )
        
        assert 'value_mean_by_group' in result.columns
        assert result.loc[0, 'value_mean_by_group'] == 15.0  # Mean of A group
        assert result.loc[2, 'value_mean_by_group'] == 35.0  # Mean of B group
    
    def test_create_aggregation_features_multiple_functions(self):
        """Test multiple aggregation functions"""
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [10, 20, 30, 40]
        })
        
        pipeline = FeatureEngineeringPipeline()
        result = pipeline.create_aggregation_features(
            df,
            group_by='group',
            agg_columns=['value'],
            agg_functions=['mean', 'std', 'min', 'max', 'count']
        )
        
        assert 'value_mean_by_group' in result.columns
        assert 'value_std_by_group' in result.columns
        assert 'value_min_by_group' in result.columns
        assert 'value_max_by_group' in result.columns
        assert 'value_count_by_group' in result.columns
    
    def test_encode_onehot_basic(self):
        """Test basic one-hot encoding"""
        df = pd.DataFrame({
            'cat': ['A', 'B', 'C', 'A', 'B'],
            'num': [1, 2, 3, 4, 5]
        })
        
        pipeline = FeatureEngineeringPipeline()
        result = pipeline.encode_onehot(df, columns=['cat'])
        
        assert 'cat' not in result.columns
        assert 'cat_A' in result.columns or 'cat_B' in result.columns
        assert 'num' in result.columns
    
    def test_encode_onehot_drop_first(self):
        """Test one-hot encoding with drop_first"""
        df = pd.DataFrame({
            'cat': ['A', 'B', 'C', 'A', 'B'],
            'num': [1, 2, 3, 4, 5]
        })
        
        pipeline = FeatureEngineeringPipeline()
        result = pipeline.encode_onehot(df, columns=['cat'], drop_first=True)
        
        # Should have n-1 columns for n categories
        cat_cols = [col for col in result.columns if col.startswith('cat_')]
        assert len(cat_cols) == 2  # 3 categories - 1
    
    def test_encode_onehot_max_categories(self):
        """Test one-hot encoding with max_categories limit"""
        df = pd.DataFrame({
            'cat': ['A'] * 50 + ['B'] * 30 + ['C'] * 10 + ['D'] * 5 + ['E'] * 5,
            'num': range(100)
        })
        
        pipeline = FeatureEngineeringPipeline()
        result = pipeline.encode_onehot(df, columns=['cat'], max_categories=3)
        
        cat_cols = [col for col in result.columns if col.startswith('cat_')]
        # Should have max 3 categories + 1 for 'Other'
        assert len(cat_cols) <= 4
    
    def test_encode_label_basic(self):
        """Test basic label encoding"""
        df = pd.DataFrame({
            'cat': ['A', 'B', 'C', 'A', 'B'],
            'num': [1, 2, 3, 4, 5]
        })
        
        pipeline = FeatureEngineeringPipeline()
        result = pipeline.encode_label(df, columns=['cat'])
        
        assert 'cat_label' in result.columns
        assert result['cat_label'].dtype in [np.int32, np.int64]
        assert 'cat' in result.columns  # Original preserved
    
    def test_encode_label_consistent_mapping(self):
        """Test label encoding produces consistent mappings"""
        df = pd.DataFrame({
            'cat': ['A', 'B', 'C', 'A', 'B', 'C']
        })
        
        pipeline = FeatureEngineeringPipeline()
        result = pipeline.encode_label(df, columns=['cat'])
        
        # Same category should get same label
        a_values = result.loc[df['cat'] == 'A', 'cat_label'].unique()
        assert len(a_values) == 1
    
    def test_encode_target_basic(self):
        """Test basic target encoding"""
        df = pd.DataFrame({
            'cat': ['A', 'A', 'B', 'B', 'C'],
            'target': [1, 1, 0, 0, 1]
        })
        
        pipeline = FeatureEngineeringPipeline()
        result = pipeline.encode_target(df, columns=['cat'], target='target')
        
        assert 'cat_target' in result.columns
        assert result['cat_target'].dtype == np.float64
    
    def test_encode_target_with_smoothing(self):
        """Test target encoding with smoothing"""
        df = pd.DataFrame({
            'cat': ['A', 'A', 'B', 'B', 'C'],
            'target': [1, 1, 0, 0, 1]
        })
        
        pipeline = FeatureEngineeringPipeline()
        result = pipeline.encode_target(
            df, columns=['cat'], target='target', smoothing=1.0
        )
        
        assert 'cat_target' in result.columns
        # Values should be smoothed towards global mean
        global_mean = df['target'].mean()
        assert not (result['cat_target'] == 1.0).all()
        assert not (result['cat_target'] == 0.0).all()
    
    def test_encode_frequency_basic(self):
        """Test basic frequency encoding"""
        df = pd.DataFrame({
            'cat': ['A', 'A', 'A', 'B', 'B', 'C']
        })
        
        pipeline = FeatureEngineeringPipeline()
        result = pipeline.encode_frequency(df, columns=['cat'])
        
        assert 'cat_freq' in result.columns
        assert np.isclose(result.loc[0, 'cat_freq'], 3/6)  # A appears 3 times
        assert np.isclose(result.loc[3, 'cat_freq'], 2/6)  # B appears 2 times
        assert np.isclose(result.loc[5, 'cat_freq'], 1/6)  # C appears 1 time
    
    def test_create_polynomial_features_degree2(self):
        """Test polynomial features with degree 2"""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [2, 3, 4, 5, 6]
        })
        
        pipeline = FeatureEngineeringPipeline()
        result = pipeline.create_polynomial_features(
            df, columns=['A', 'B'], degree=2
        )
        
        # Should have A*B and A^2, B^2
        poly_cols = [col for col in result.columns if col.startswith('poly_')]
        assert len(poly_cols) > 0
    
    def test_create_polynomial_features_degree3(self):
        """Test polynomial features with degree 3"""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5]
        })
        
        pipeline = FeatureEngineeringPipeline()
        result = pipeline.create_polynomial_features(
            df, columns=['A'], degree=3
        )
        
        poly_cols = [col for col in result.columns if col.startswith('poly_')]
        # Should include A^2 and A^3
        assert len(poly_cols) >= 2
    
    def test_select_features_univariate(self):
        """Test univariate feature selection"""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 10))
        y = pd.Series(np.random.randint(0, 2, 100))
        
        pipeline = FeatureEngineeringPipeline()
        selected, scores = pipeline.select_features_univariate(X, y, k=5)
        
        assert len(selected) == 5
        assert len(scores) == 10
        assert 'univariate' in pipeline.feature_importance_scores
    
    def test_select_features_rfe(self):
        """Test RFE feature selection"""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 10))
        y = pd.Series(np.random.randint(0, 2, 100))
        
        pipeline = FeatureEngineeringPipeline()
        selected = pipeline.select_features_rfe(X, y, n_features=5)
        
        assert len(selected) == 5
        assert 'rfe' in pipeline.feature_importance_scores
    
    def test_select_features_model_based(self):
        """Test model-based feature selection"""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 10))
        y = pd.Series(np.random.randint(0, 2, 100))
        
        pipeline = FeatureEngineeringPipeline()
        selected, importances = pipeline.select_features_model_based(X, y, n_features=5)
        
        assert len(selected) == 5
        assert len(importances) == 10
        assert 'random_forest' in pipeline.feature_importance_scores
    
    def test_get_feature_catalog(self):
        """Test feature catalog generation"""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [2, 3, 4, 5, 6]
        })
        
        pipeline = FeatureEngineeringPipeline()
        pipeline.create_mathematical_features(df, columns=['A'], operations=['log'])
        
        catalog = pipeline.get_feature_catalog()
        
        assert isinstance(catalog, pd.DataFrame)
        assert 'Feature' in catalog.columns
        assert 'Description' in catalog.columns
        assert len(catalog) > 0
    
    def test_evaluate_feature_impact(self):
        """Test feature impact evaluation"""
        np.random.seed(42)
        X_original = pd.DataFrame(np.random.randn(100, 5))
        X_engineered = pd.DataFrame(np.random.randn(100, 10))
        y = pd.Series(np.random.randint(0, 2, 100))
        
        pipeline = FeatureEngineeringPipeline()
        impact = pipeline.evaluate_feature_impact(
            X_original, X_engineered, y, cv=3
        )
        
        assert 'original_mean' in impact
        assert 'original_std' in impact
        assert 'engineered_mean' in impact
        assert 'engineered_std' in impact
        assert 'improvement' in impact


class TestCreateSampleDataset:
    """Test cases for sample dataset creation"""
    
    def test_basic_dataset_creation(self):
        """Test basic dataset creation"""
        df, y = create_sample_dataset(n_samples=100, n_features=5, n_categorical=2)
        
        assert df.shape[0] == 100
        assert len(y) == 100
        assert df.shape[1] >= 5 + 2 + 1  # numerical + categorical + group
    
    def test_dataset_target_distribution(self):
        """Test target variable distribution"""
        df, y = create_sample_dataset(n_samples=1000, random_state=42)
        
        # Should be roughly balanced for binary classification
        value_counts = y.value_counts()
        assert len(value_counts) == 2
        assert value_counts.min() / value_counts.max() > 0.3  # Not too imbalanced
    
    def test_dataset_reproducibility(self):
        """Test dataset creation is reproducible"""
        df1, y1 = create_sample_dataset(n_samples=100, random_state=42)
        df2, y2 = create_sample_dataset(n_samples=100, random_state=42)
        
        pd.testing.assert_frame_equal(df1, df2)
        pd.testing.assert_series_equal(y1, y2)
    
    def test_dataset_feature_types(self):
        """Test dataset has correct feature types"""
        df, y = create_sample_dataset(n_samples=100, n_features=3, n_categorical=2)
        
        # Check for numerical features
        num_features = [col for col in df.columns if col.startswith('num_')]
        assert len(num_features) == 3
        
        # Check for categorical features
        cat_features = [col for col in df.columns if col.startswith('cat_')]
        assert len(cat_features) == 2


class TestVisualizationFunctions:
    """Test cases for visualization functions"""
    
    def test_visualize_feature_importance(self, tmp_path):
        """Test feature importance visualization"""
        importance_scores = {
            'feature1': 0.5,
            'feature2': 0.3,
            'feature3': 0.2
        }
        
        output_path = tmp_path / "test_importance.png"
        visualize_feature_importance(
            importance_scores,
            top_n=3,
            output_path=str(output_path)
        )
        
        assert output_path.exists()
    
    def test_visualize_feature_correlations(self, tmp_path):
        """Test feature correlation visualization"""
        df = pd.DataFrame(np.random.randn(100, 5), columns=['A', 'B', 'C', 'D', 'E'])
        
        output_path = tmp_path / "test_corr.png"
        visualize_feature_correlations(
            df,
            features=['A', 'B', 'C'],
            output_path=str(output_path)
        )
        
        assert output_path.exists()
    
    def test_visualize_pairplot(self, tmp_path):
        """Test pair plot visualization"""
        df = pd.DataFrame(np.random.randn(100, 3), columns=['A', 'B', 'C'])
        y = pd.Series(np.random.randint(0, 2, 100))
        
        output_path = tmp_path / "test_pairplot.png"
        visualize_pairplot(
            df,
            features=['A', 'B', 'C'],
            target=y,
            output_path=str(output_path)
        )
        
        assert output_path.exists()
    
    def test_visualize_polynomial_impact(self, tmp_path):
        """Test polynomial impact visualization"""
        X = np.random.randn(100)
        X_poly = np.column_stack([X, X**2])
        y = 2 * X + 0.5 * X**2 + np.random.randn(100) * 0.1
        
        output_path = tmp_path / "test_poly.png"
        visualize_polynomial_impact(
            X, X_poly, y,
            output_path=str(output_path)
        )
        
        assert output_path.exists()
    
    def test_visualize_encoding_comparison(self, tmp_path):
        """Test encoding comparison visualization"""
        performance_dict = {
            'Label Encoding': 0.85,
            'One-Hot Encoding': 0.87,
            'Target Encoding': 0.88
        }
        
        output_path = tmp_path / "test_encoding.png"
        visualize_encoding_comparison(
            performance_dict,
            output_path=str(output_path)
        )
        
        assert output_path.exists()


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_mathematical_features_empty_columns(self):
        """Test mathematical features with empty column list"""
        df = pd.DataFrame({'A': [1, 2, 3]})
        pipeline = FeatureEngineeringPipeline()
        
        result = pipeline.create_mathematical_features(df, columns=[])
        pd.testing.assert_frame_equal(result, df)
    
    def test_mathematical_features_nonexistent_columns(self):
        """Test mathematical features with non-existent columns"""
        df = pd.DataFrame({'A': [1, 2, 3]})
        pipeline = FeatureEngineeringPipeline()
        
        result = pipeline.create_mathematical_features(df, columns=['B', 'C'])
        pd.testing.assert_frame_equal(result, df)
    
    def test_interaction_features_invalid_pairs(self):
        """Test interaction features with invalid column pairs"""
        df = pd.DataFrame({'A': [1, 2, 3]})
        pipeline = FeatureEngineeringPipeline()
        
        result = pipeline.create_interaction_features(
            df, column_pairs=[('A', 'B'), ('C', 'D')]
        )
        # Should only have original columns
        assert list(result.columns) == ['A']
    
    def test_aggregation_features_nonexistent_group(self):
        """Test aggregation with non-existent group column"""
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        pipeline = FeatureEngineeringPipeline()
        
        result = pipeline.create_aggregation_features(
            df, group_by='nonexistent', agg_columns=['A']
        )
        pd.testing.assert_frame_equal(result, df)
    
    def test_onehot_encoding_single_category(self):
        """Test one-hot encoding with single category"""
        df = pd.DataFrame({'cat': ['A', 'A', 'A'], 'num': [1, 2, 3]})
        pipeline = FeatureEngineeringPipeline()
        
        result = pipeline.encode_onehot(df, columns=['cat'])
        # Should have at least one encoded column
        assert 'num' in result.columns
        assert 'cat' not in result.columns
    
    def test_feature_selection_k_larger_than_features(self):
        """Test feature selection when k > n_features"""
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.randint(0, 2, 100))
        
        pipeline = FeatureEngineeringPipeline()
        selected, scores = pipeline.select_features_univariate(X, y, k=10)
        
        # Should select all available features
        assert len(selected) == 5
    
    def test_polynomial_features_empty_columns(self):
        """Test polynomial features with empty column list"""
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        pipeline = FeatureEngineeringPipeline()
        
        result = pipeline.create_polynomial_features(df, columns=[])
        pd.testing.assert_frame_equal(result, df)
    
    def test_target_encoding_missing_target(self):
        """Test target encoding with missing target column"""
        df = pd.DataFrame({'cat': ['A', 'B', 'C']})
        pipeline = FeatureEngineeringPipeline()
        
        result = pipeline.encode_target(df, columns=['cat'], target='nonexistent')
        pd.testing.assert_frame_equal(result, df)


class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_full_pipeline_workflow(self):
        """Test complete feature engineering pipeline"""
        # Create dataset
        df, y = create_sample_dataset(n_samples=200, n_features=3, n_categorical=2)
        
        # Initialize pipeline
        pipeline = FeatureEngineeringPipeline(random_state=42)
        
        # Apply transformations
        df = pipeline.create_mathematical_features(
            df, columns=['num_feature_1', 'num_feature_2'], operations=['log', 'square']
        )
        
        df = pipeline.create_interaction_features(
            df, column_pairs=[('num_feature_1', 'num_feature_2')], operations=['multiply']
        )
        
        df = pipeline.encode_label(df, columns=['cat_feature_1'])
        df = pipeline.encode_frequency(df, columns=['cat_feature_2'])
        
        # Feature selection
        numerical_cols = [col for col in df.columns if df[col].dtype in [np.float64, np.int64]]
        if len(numerical_cols) > 0:
            X = df[numerical_cols]
            selected, scores = pipeline.select_features_univariate(X, y, k=min(10, len(numerical_cols)))
            
            assert len(selected) > 0
            assert len(pipeline.feature_catalog) > 0
    
    def test_encoding_pipeline(self):
        """Test different encoding methods in sequence"""
        df = pd.DataFrame({
            'cat1': ['A', 'B', 'C'] * 30,
            'cat2': ['X', 'Y', 'Z'] * 30,
            'num': np.random.randn(90),
            'target': np.random.randint(0, 2, 90)
        })
        
        pipeline = FeatureEngineeringPipeline(random_state=42)
        
        # Label encoding
        df = pipeline.encode_label(df, columns=['cat1'])
        assert 'cat1_label' in df.columns
        
        # Frequency encoding
        df = pipeline.encode_frequency(df, columns=['cat2'])
        assert 'cat2_freq' in df.columns
        
        # Target encoding
        df = pipeline.encode_target(df, columns=['cat1'], target='target')
        assert 'cat1_target' in df.columns
        
        assert len(pipeline.encoders) >= 3
    
    def test_feature_importance_comparison(self):
        """Test comparing feature importance from different methods"""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(150, 8))
        y = pd.Series(np.random.randint(0, 2, 150))
        
        pipeline = FeatureEngineeringPipeline(random_state=42)
        
        # Get importance from different methods
        selected_univariate, scores_univariate = pipeline.select_features_univariate(X, y, k=5)
        selected_rf, scores_rf = pipeline.select_features_model_based(X, y, n_features=5)
        selected_rfe = pipeline.select_features_rfe(X, y, n_features=5)
        
        # Check all methods ran successfully
        assert 'univariate' in pipeline.feature_importance_scores
        assert 'random_forest' in pipeline.feature_importance_scores
        assert 'rfe' in pipeline.feature_importance_scores
        
        # All should select the requested number of features
        assert len(selected_univariate) == 5
        assert len(selected_rf) == 5
        assert len(selected_rfe) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
