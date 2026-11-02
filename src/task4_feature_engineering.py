"""
Task 4: Advanced Feature Engineering Pipeline
Implements comprehensive feature engineering techniques including mathematical transformations,
categorical encoding, interaction features, and polynomial features.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import (
    OneHotEncoder, LabelEncoder, PolynomialFeatures, StandardScaler
)
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, RFE
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from typing import List, Dict, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineeringPipeline:
    """
    Comprehensive feature engineering pipeline for creating, transforming,
    and selecting features to improve model performance.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the feature engineering pipeline.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.feature_catalog = {}
        self.encoders = {}
        self.original_features = []
        self.engineered_features = []
        self.feature_importance_scores = {}
        
    def create_mathematical_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        operations: List[str] = ['log', 'sqrt', 'square', 'cube']
    ) -> pd.DataFrame:
        """
        Create new features using mathematical transformations.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        columns : List[str]
            Columns to transform
        operations : List[str]
            Mathematical operations to apply
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with added mathematical features
        """
        pass
    
    def create_interaction_features(
        self,
        df: pd.DataFrame,
        column_pairs: List[Tuple[str, str]],
        operations: List[str] = ['multiply', 'add', 'subtract', 'divide']
    ) -> pd.DataFrame:
        """
        Create interaction features between pairs of columns.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        column_pairs : List[Tuple[str, str]]
            Pairs of columns to create interactions
        operations : List[str]
            Operations to apply
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with interaction features
        """
        pass
    
    def create_aggregation_features(
        self,
        df: pd.DataFrame,
        group_by: str,
        agg_columns: List[str],
        agg_functions: List[str] = ['mean', 'std', 'min', 'max', 'count']
    ) -> pd.DataFrame:
        """
        Create aggregation features based on groupby operations.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        group_by : str
            Column to group by
        agg_columns : List[str]
            Columns to aggregate
        agg_functions : List[str]
            Aggregation functions
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with aggregation features
        """
        pass
    
    def encode_onehot(
        self,
        df: pd.DataFrame,
        columns: List[str],
        drop_first: bool = False,
        max_categories: int = 10
    ) -> pd.DataFrame:
        """
        Apply one-hot encoding to categorical columns.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        columns : List[str]
            Categorical columns to encode
        drop_first : bool
            Whether to drop first category to avoid multicollinearity
        max_categories : int
            Maximum number of categories (others grouped as 'Other')
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with one-hot encoded features
        """
        pass
    
    def encode_label(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> pd.DataFrame:
        """
        Apply label encoding to categorical columns.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        columns : List[str]
            Categorical columns to encode
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with label encoded features
        """
        pass
    
    def encode_target(
        self,
        df: pd.DataFrame,
        columns: List[str],
        target: str,
        smoothing: float = 0.0
    ) -> pd.DataFrame:
        """
        Apply target encoding to categorical columns.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        columns : List[str]
            Categorical columns to encode
        target : str
            Target column name
        smoothing : float
            Smoothing parameter to avoid overfitting
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with target encoded features
        """
        pass
    
    def encode_frequency(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> pd.DataFrame:
        """
        Apply frequency encoding to categorical columns.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        columns : List[str]
            Categorical columns to encode
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with frequency encoded features
        """
        pass
    
    def create_polynomial_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        degree: int = 2,
        include_bias: bool = False
    ) -> pd.DataFrame:
        """
        Create polynomial features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        columns : List[str]
            Columns to create polynomial features from
        degree : int
            Polynomial degree
        include_bias : bool
            Whether to include bias column
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with polynomial features
        """
        pass
    
    def select_features_univariate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        k: int = 10,
        score_func=f_classif
    ) -> Tuple[List[str], np.ndarray]:
        """
        Select top k features using univariate statistical tests.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        k : int
            Number of features to select
        score_func : callable
            Scoring function (f_classif, mutual_info_classif, etc.)
            
        Returns:
        --------
        Tuple[List[str], np.ndarray]
            Selected feature names and their scores
        """
        selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
        selector.fit(X, y)
        
        # Get selected features
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        # Get scores for all features
        scores = selector.scores_
        
        self.feature_importance_scores['univariate'] = dict(zip(X.columns, scores))
        
        return selected_features, scores
    
    def select_features_rfe(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int = 10,
        step: int = 1
    ) -> List[str]:
        """
        Select features using Recursive Feature Elimination.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        n_features : int
            Number of features to select
        step : int
            Number of features to remove at each iteration
            
        Returns:
        --------
        List[str]
            Selected feature names
        """
        estimator = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            solver='lbfgs'
        )
        
        selector = RFE(
            estimator=estimator,
            n_features_to_select=min(n_features, X.shape[1]),
            step=step
        )
        selector.fit(X, y)
        
        # Get selected features
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        # Get rankings
        rankings = dict(zip(X.columns, selector.ranking_))
        self.feature_importance_scores['rfe'] = rankings
        
        return selected_features
    
    def select_features_model_based(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int = 10
    ) -> Tuple[List[str], np.ndarray]:
        """
        Select features using tree-based feature importance.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        n_features : int
            Number of top features to select
            
        Returns:
        --------
        Tuple[List[str], np.ndarray]
            Selected feature names and their importance scores
        """
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        
        # Sort and select top features
        indices = np.argsort(importances)[::-1]
        top_indices = indices[:n_features]
        selected_features = X.columns[top_indices].tolist()
        
        self.feature_importance_scores['random_forest'] = dict(
            zip(X.columns, importances)
        )
        
        return selected_features, importances
    
    def get_feature_catalog(self) -> pd.DataFrame:
        """
        Get catalog of all features with descriptions.
        
        Returns:
        --------
        pd.DataFrame
            Feature catalog
        """
        catalog_df = pd.DataFrame([
            {'Feature': k, 'Description': v, 'Type': 'Engineered'}
            for k, v in self.feature_catalog.items()
        ])
        
        return catalog_df
    
    def evaluate_feature_impact(
        self,
        X_original: pd.DataFrame,
        X_engineered: pd.DataFrame,
        y: pd.Series,
        cv: int = 5
    ) -> Dict[str, float]:
        """
        Evaluate the impact of feature engineering on model performance.
        
        Parameters:
        -----------
        X_original : pd.DataFrame
            Original features
        X_engineered : pd.DataFrame
            Features after engineering
        y : pd.Series
            Target variable
        cv : int
            Number of cross-validation folds
            
        Returns:
        --------
        Dict[str, float]
            Performance scores for original and engineered features
        """
        model = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            solver='lbfgs'
        )
        
        # Evaluate original features
        scores_original = cross_val_score(
            model, X_original, y, cv=cv, scoring='accuracy'
        )
        
        # Evaluate engineered features
        scores_engineered = cross_val_score(
            model, X_engineered, y, cv=cv, scoring='accuracy'
        )
        
        return {
            'original_mean': scores_original.mean(),
            'original_std': scores_original.std(),
            'engineered_mean': scores_engineered.mean(),
            'engineered_std': scores_engineered.std(),
            'improvement': scores_engineered.mean() - scores_original.mean()
        }


def create_sample_dataset(
    n_samples: int = 1000,
    n_features: int = 5,
    n_categorical: int = 2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create a sample dataset for feature engineering demonstration.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    n_features : int
        Number of numerical features
    n_categorical : int
        Number of categorical features
    random_state : int
        Random seed
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.Series]
        Feature dataframe and target series
    """
    np.random.seed(random_state)
    
    # Create numerical features
    data = {}
    for i in range(n_features):
        data[f'num_feature_{i+1}'] = np.random.randn(n_samples) * 10 + 50
    
    # Create categorical features
    categories_list = [
        ['A', 'B', 'C', 'D'],
        ['X', 'Y', 'Z'],
        ['Low', 'Medium', 'High'],
        ['Type1', 'Type2', 'Type3', 'Type4']
    ]
    
    for i in range(n_categorical):
        categories = categories_list[i % len(categories_list)]
        data[f'cat_feature_{i+1}'] = np.random.choice(
            categories,
            size=n_samples
        )
    
    # Create group feature for aggregation
    data['group'] = np.random.choice(['Group1', 'Group2', 'Group3'], size=n_samples)
    
    df = pd.DataFrame(data)
    
    # Create target with some relationship to features
    target = (
        0.5 * df['num_feature_1'] +
        0.3 * df['num_feature_2'] +
        0.2 * df['num_feature_1'] * df['num_feature_2'] +
        np.random.randn(n_samples) * 5
    )
    
    # Convert to binary classification
    y = (target > target.median()).astype(int)
    
    return df, y


def visualize_feature_importance(
    importance_scores: Dict[str, float],
    top_n: int = 15,
    title: str = 'Feature Importance',
    output_path: str = 'feature_importance.png'
):
    """
    Visualize feature importance scores.
    
    Parameters:
    -----------
    importance_scores : Dict[str, float]
        Feature names and their importance scores
    top_n : int
        Number of top features to display
    title : str
        Plot title
    output_path : str
        Path to save the plot
    """
    # Sort by importance
    sorted_features = sorted(
        importance_scores.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_n]
    
    features, scores = zip(*sorted_features)
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
    bars = plt.barh(range(len(features)), scores, color=colors)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance Score')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Feature importance plot saved to {output_path}")


def visualize_feature_correlations(
    df: pd.DataFrame,
    features: List[str],
    title: str = 'Feature Correlation Matrix',
    output_path: str = 'feature_correlations.png'
):
    """
    Visualize correlation between features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with features
    features : List[str]
        Features to include in correlation matrix
    title : str
        Plot title
    output_path : str
        Path to save the plot
    """
    # Get valid features
    valid_features = [f for f in features if f in df.columns]
    
    if not valid_features:
        print("No valid features found for correlation plot")
        return
    
    # Calculate correlation
    corr = df[valid_features].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr,
        annot=False,
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Correlation heatmap saved to {output_path}")


def visualize_pairplot(
    df: pd.DataFrame,
    features: List[str],
    target: Optional[pd.Series] = None,
    output_path: str = 'feature_pairplot.png'
):
    """
    Create pair plot for interaction features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with features
    features : List[str]
        Features to include in pair plot
    target : Optional[pd.Series]
        Target variable for coloring
    output_path : str
        Path to save the plot
    """
    # Limit number of features
    features = features[:5]
    valid_features = [f for f in features if f in df.columns]
    
    if not valid_features:
        print("No valid features found for pair plot")
        return
    
    plot_df = df[valid_features].copy()
    
    if target is not None:
        plot_df['Target'] = target.values
        hue = 'Target'
    else:
        hue = None
    
    pair_plot = sns.pairplot(
        plot_df,
        hue=hue,
        diag_kind='kde',
        plot_kws={'alpha': 0.6},
        height=2.5
    )
    
    pair_plot.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Pair plot saved to {output_path}")


def visualize_polynomial_impact(
    X_original: np.ndarray,
    X_poly: np.ndarray,
    y: np.ndarray,
    feature_name: str = 'Feature',
    output_path: str = 'polynomial_features.png'
):
    """
    Visualize the impact of polynomial features on predictions.
    
    Parameters:
    -----------
    X_original : np.ndarray
        Original feature values
    X_poly : np.ndarray
        Polynomial feature values
    y : np.ndarray
        Target values
    feature_name : str
        Name of the feature
    output_path : str
        Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Original feature
    axes[0].scatter(X_original, y, alpha=0.5, s=20)
    axes[0].set_xlabel(feature_name)
    axes[0].set_ylabel('Target')
    axes[0].set_title('Original Feature vs Target')
    axes[0].grid(True, alpha=0.3)
    
    # With polynomial fit
    axes[1].scatter(X_original, y, alpha=0.5, s=20, label='Data')
    
    # Fit polynomial
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    
    poly = PolynomialFeatures(degree=2)
    X_poly_fit = poly.fit_transform(X_original.reshape(-1, 1))
    model = LinearRegression()
    model.fit(X_poly_fit, y)
    
    # Plot fit
    X_plot = np.linspace(X_original.min(), X_original.max(), 100).reshape(-1, 1)
    X_plot_poly = poly.transform(X_plot)
    y_pred = model.predict(X_plot_poly)
    
    axes[1].plot(X_plot, y_pred, 'r-', linewidth=2, label='Polynomial Fit')
    axes[1].set_xlabel(feature_name)
    axes[1].set_ylabel('Target')
    axes[1].set_title('Polynomial Features (Degree 2)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Polynomial feature visualization saved to {output_path}")


def visualize_encoding_comparison(
    performance_dict: Dict[str, float],
    output_path: str = 'encoding_comparison.png'
):
    """
    Compare model performance with different encoding techniques.
    
    Parameters:
    -----------
    performance_dict : Dict[str, float]
        Encoding method names and performance scores
    output_path : str
        Path to save the plot
    """
    methods = list(performance_dict.keys())
    scores = list(performance_dict.values())
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    bars = plt.bar(range(len(methods)), scores, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f'{score:.4f}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
    plt.ylabel('Model Accuracy')
    plt.title('Performance Comparison: Different Encoding Techniques')
    plt.ylim(0, max(scores) * 1.15)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Encoding comparison plot saved to {output_path}")


def visualize_feature_network(
    interaction_pairs: List[Tuple[str, str]],
    importance_scores: Dict[str, float],
    output_path: str = 'feature_network.png'
):
    """
    Visualize feature interactions as a network graph.
    
    Parameters:
    -----------
    interaction_pairs : List[Tuple[str, str]]
        Pairs of interacting features
    importance_scores : Dict[str, float]
        Importance scores for features
    output_path : str
        Path to save the plot
    """
    try:
        import networkx as nx
    except ImportError:
        print("NetworkX not installed. Skipping network visualization.")
        return
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    all_features = set()
    for f1, f2 in interaction_pairs:
        all_features.add(f1)
        all_features.add(f2)
    
    for feature in all_features:
        importance = importance_scores.get(feature, 0.5)
        G.add_node(feature, importance=importance)
    
    # Add edges
    for f1, f2 in interaction_pairs:
        G.add_edge(f1, f2)
    
    # Draw graph
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Node sizes based on importance
    node_sizes = [
        G.nodes[node].get('importance', 0.5) * 2000
        for node in G.nodes()
    ]
    
    # Draw
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color='lightblue',
        edgecolors='black',
        linewidths=2
    )
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=2)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    plt.title('Feature Interaction Network', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Feature network visualization saved to {output_path}")


if __name__ == "__main__":
    print("=" * 80)
    print("Task 4: Advanced Feature Engineering Pipeline")
    print("=" * 80)
    
    # Create sample dataset
    print("\n1. Creating sample dataset...")
    df, y = create_sample_dataset(n_samples=1000, n_features=5, n_categorical=2)
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(df.columns)}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Initialize pipeline
    print("\n2. Initializing Feature Engineering Pipeline...")
    pipeline = FeatureEngineeringPipeline(random_state=42)
    
    # Store original features
    original_numerical = ['num_feature_1', 'num_feature_2', 'num_feature_3']
    original_categorical = ['cat_feature_1', 'cat_feature_2']
    
    # Mathematical transformations
    print("\n3. Creating mathematical features...")
    df = pipeline.create_mathematical_features(
        df,
        columns=original_numerical[:2],
        operations=['log', 'sqrt', 'square']
    )
    print(f"Features after mathematical transformations: {df.shape[1]}")
    
    # Interaction features
    print("\n4. Creating interaction features...")
    interaction_pairs = [
        ('num_feature_1', 'num_feature_2'),
        ('num_feature_1', 'num_feature_3'),
        ('num_feature_2', 'num_feature_3')
    ]
    df = pipeline.create_interaction_features(
        df,
        column_pairs=interaction_pairs,
        operations=['multiply', 'add', 'divide']
    )
    print(f"Features after interactions: {df.shape[1]}")
    
    # Aggregation features
    print("\n5. Creating aggregation features...")
    df = pipeline.create_aggregation_features(
        df,
        group_by='group',
        agg_columns=original_numerical[:2],
        agg_functions=['mean', 'std', 'max']
    )
    print(f"Features after aggregations: {df.shape[1]}")
    
    # Categorical encoding
    print("\n6. Encoding categorical features...")
    
    # Target encoding
    df = pipeline.encode_target(
        df,
        columns=['cat_feature_1'],
        target='target_temp',
        smoothing=10.0
    )
    
    # Add temporary target for encoding
    df['target_temp'] = y
    df = pipeline.encode_target(
        df,
        columns=['cat_feature_2'],
        target='target_temp',
        smoothing=10.0
    )
    df = df.drop(columns=['target_temp'])
    
    # Frequency encoding
    df = pipeline.encode_frequency(df, columns=['group'])
    
    # Label encoding (keep original for one-hot)
    df = pipeline.encode_label(df, columns=original_categorical)
    
    # One-hot encoding (this will drop original categorical columns)
    df = pipeline.encode_onehot(
        df,
        columns=original_categorical + ['group'],
        drop_first=True
    )
    
    print(f"Features after encoding: {df.shape[1]}")
    
    # Polynomial features
    print("\n7. Creating polynomial features...")
    df = pipeline.create_polynomial_features(
        df,
        columns=original_numerical[:2],
        degree=2,
        include_bias=False
    )
    print(f"Total features after all engineering: {df.shape[1]}")
    
    # Feature selection
    print("\n8. Performing feature selection...")
    
    # Univariate selection
    selected_univariate, scores_univariate = pipeline.select_features_univariate(
        df, y, k=20, score_func=f_classif
    )
    print(f"Top 20 features (univariate): {len(selected_univariate)}")
    
    # Model-based selection
    selected_rf, scores_rf = pipeline.select_features_model_based(
        df, y, n_features=20
    )
    print(f"Top 20 features (Random Forest): {len(selected_rf)}")
    
    # RFE selection
    selected_rfe = pipeline.select_features_rfe(
        df, y, n_features=20, step=5
    )
    print(f"Top 20 features (RFE): {len(selected_rfe)}")
    
    # Feature catalog
    print("\n9. Generating feature catalog...")
    catalog = pipeline.get_feature_catalog()
    print(f"Total engineered features cataloged: {len(catalog)}")
    print("\nSample catalog entries:")
    print(catalog.head(10).to_string(index=False))
    
    # Evaluate impact
    print("\n10. Evaluating feature engineering impact...")
    X_original = df[original_numerical].copy()
    X_engineered = df[selected_rf].copy()
    
    impact = pipeline.evaluate_feature_impact(X_original, X_engineered, y, cv=5)
    print(f"\nOriginal features accuracy: {impact['original_mean']:.4f} (+/- {impact['original_std']:.4f})")
    print(f"Engineered features accuracy: {impact['engineered_mean']:.4f} (+/- {impact['engineered_std']:.4f})")
    print(f"Improvement: {impact['improvement']:.4f}")
    
    # Visualizations
    print("\n11. Generating visualizations...")
    
    # Feature importance (Random Forest)
    visualize_feature_importance(
        pipeline.feature_importance_scores['random_forest'],
        top_n=15,
        title='Top 15 Features by Random Forest Importance',
        output_path='task4_feature_importance.png'
    )
    
    # Feature importance (Univariate)
    visualize_feature_importance(
        pipeline.feature_importance_scores['univariate'],
        top_n=15,
        title='Top 15 Features by Univariate F-Score',
        output_path='task4_univariate_importance.png'
    )
    
    # Correlation heatmap
    visualize_feature_correlations(
        df,
        features=selected_rf[:20],
        title='Correlation Matrix of Top 20 Features',
        output_path='task4_feature_correlations.png'
    )
    
    # Pair plot for interaction features
    interaction_feature_names = [
        f'{col1}_x_{col2}' for col1, col2 in interaction_pairs
    ]
    visualize_pairplot(
        df,
        features=interaction_feature_names,
        target=y,
        output_path='task4_interaction_pairplot.png'
    )
    
    # Polynomial feature visualization
    visualize_polynomial_impact(
        df['num_feature_1'].values,
        df[['num_feature_1', 'poly_num_feature_1^2']].values,
        y.values,
        feature_name='num_feature_1',
        output_path='task4_polynomial_impact.png'
    )
    
    # Encoding comparison
    print("\n12. Comparing encoding techniques...")
    encoding_performance = {}
    
    # Prepare different encoded versions
    df_temp, y_temp = create_sample_dataset(n_samples=1000, random_state=42)
    
    # Label encoding
    df_label = df_temp.copy()
    le1 = LabelEncoder()
    le2 = LabelEncoder()
    df_label['cat_feature_1_encoded'] = le1.fit_transform(df_label['cat_feature_1'])
    df_label['cat_feature_2_encoded'] = le2.fit_transform(df_label['cat_feature_2'])
    X_label = df_label[['num_feature_1', 'num_feature_2', 'num_feature_3',
                        'cat_feature_1_encoded', 'cat_feature_2_encoded']]
    model = LogisticRegression(max_iter=1000, random_state=42)
    encoding_performance['Label Encoding'] = cross_val_score(
        model, X_label, y_temp, cv=5
    ).mean()
    
    # One-hot encoding
    df_onehot = pd.get_dummies(df_temp, columns=['cat_feature_1', 'cat_feature_2'], drop_first=True)
    X_onehot = df_onehot.drop(columns=['group'])
    encoding_performance['One-Hot Encoding'] = cross_val_score(
        model, X_onehot, y_temp, cv=5
    ).mean()
    
    # Frequency encoding
    freq1 = df_temp['cat_feature_1'].value_counts(normalize=True).to_dict()
    freq2 = df_temp['cat_feature_2'].value_counts(normalize=True).to_dict()
    df_freq = df_temp.copy()
    df_freq['cat_feature_1_freq'] = df_freq['cat_feature_1'].map(freq1)
    df_freq['cat_feature_2_freq'] = df_freq['cat_feature_2'].map(freq2)
    X_freq = df_freq[['num_feature_1', 'num_feature_2', 'num_feature_3',
                      'cat_feature_1_freq', 'cat_feature_2_freq']]
    encoding_performance['Frequency Encoding'] = cross_val_score(
        model, X_freq, y_temp, cv=5
    ).mean()
    
    print("\nEncoding performance comparison:")
    for method, score in encoding_performance.items():
        print(f"  {method}: {score:.4f}")
    
    visualize_encoding_comparison(
        encoding_performance,
        output_path='task4_encoding_comparison.png'
    )
    
    # Feature network
    visualize_feature_network(
        interaction_pairs,
        pipeline.feature_importance_scores['random_forest'],
        output_path='task4_feature_network.png'
    )
    
    print("\n" + "=" * 80)
    print("Feature Engineering Pipeline Complete!")
    print("=" * 80)
    print(f"\nOriginal features: {len(original_numerical) + len(original_categorical)}")
    print(f"Engineered features: {df.shape[1]}")
    print(f"Top selected features: {len(selected_rf)}")
    print(f"Performance improvement: {impact['improvement']:.4f}")
    print("\nGenerated visualizations:")
    print("  - task4_feature_importance.png")
    print("  - task4_univariate_importance.png")
    print("  - task4_feature_correlations.png")
    print("  - task4_interaction_pairplot.png")
    print("  - task4_polynomial_impact.png")
    print("  - task4_encoding_comparison.png")
    print("  - task4_feature_network.png")
