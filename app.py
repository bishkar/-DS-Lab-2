"""
Streamlit App for MLDS Week 2 - Data Preprocessing Visualization

This app provides interactive visualizations for all preprocessing tasks.
Run with: streamlit run app.py

Author: MLDS Course
Date: December 2025
"""

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Page configuration
st.set_page_config(
    page_title="MLDS Week 2 - Data Preprocessing",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .task-header {
        font-size: 1.8rem;
        color: #2ca02c;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


def main():
    """Main application function."""
    
    # Sidebar
    st.sidebar.title("📊 MLDS Week 2")
    st.sidebar.markdown("### Data Preprocessing & Visualization")
    st.sidebar.markdown("---")
    
    # Task selection
    task_options = {
        "Home": "🏠 Overview",
        "Task 1": "📋 Data Quality & Missing Values",
        "Task 2": "🎯 Outlier Detection & Treatment",
        "Task 3": "⚖️ Data Scaling & Normalization",
        "Task 4": "🔧 Feature Engineering",
        "Task 5": "🔄 Integrated Pipeline",
        "Task 6": "🌍 Real-World Case Study"
    }
    
    selected_task = st.sidebar.selectbox(
        "Select Task",
        options=list(task_options.keys()),
        format_func=lambda x: task_options[x]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This app visualizes preprocessing techniques including:\n"
        "- Data quality assessment\n"
        "- Missing value handling\n"
        "- Outlier detection\n"
        "- Feature scaling\n"
        "- Feature engineering\n"
        "- Complete pipelines"
    )
    
    # Main content area
    if selected_task == "Home":
        show_home()
    elif selected_task == "Task 1":
        show_task1()
    elif selected_task == "Task 2":
        show_task2()
    elif selected_task == "Task 3":
        show_task3()
    elif selected_task == "Task 4":
        show_task4()
    elif selected_task == "Task 5":
        show_task5()
    elif selected_task == "Task 6":
        show_task6()


def show_home():
    """Display home page with overview."""
    st.markdown('<h1 class="main-header">📊 MLDS Week 2: Data Preprocessing</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the **Data Preprocessing and Visualization** interactive dashboard!
    
    This application provides comprehensive visualizations for all preprocessing tasks 
    covered in Week 2 of the Machine Learning and Data Science course.
    """)
    
    # Task overview cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📋 Tasks 1-2
        **Data Quality & Outliers**
        - Missing value analysis
        - Imputation strategies
        - Outlier detection
        - Treatment methods
        """)
    
    with col2:
        st.markdown("""
        ### ⚖️ Tasks 3-4
        **Scaling & Engineering**
        - Feature scaling
        - Normalization methods
        - Feature creation
        - Interaction features
        """)
    
    with col3:
        st.markdown("""
        ### 🔄 Tasks 5-6
        **Complete Pipelines**
        - Integrated workflows
        - Cross-validation
        - Real-world cases
        - Production-ready code
        """)
    
    st.markdown("---")
    
    # Implementation status
    st.markdown("### 📈 Implementation Status")
    
    status_df = pd.DataFrame({
        'Task': ['Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5', 'Task 6'],
        'Status': ['📝 To Implement', '📝 To Implement', '📝 To Implement', 
                   '📝 To Implement', '✅ Complete', '✅ Complete'],
        'Tests': ['N/A', 'N/A', 'N/A', 'N/A', '34/34', '29/29'],
        'Lines of Code': ['TBD', 'TBD', 'TBD', 'TBD', '910', '911']
    })
    
    st.dataframe(status_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("""
    ### 🚀 Getting Started
    
    1. **Select a task** from the sidebar dropdown
    2. **Explore visualizations** in the main panel
    3. **Interact with controls** to see different perspectives
    4. **Download results** for your reports
    
    **Note**: Tasks 5 and 6 are fully implemented and can be explored.
    Tasks 1-4 are templates for student implementation.
    """)


def show_task1():
    """Display Task 1: Data Quality & Missing Values."""
    st.markdown('<h1 class="task-header">📋 Task 1: Data Quality & Missing Values</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This task focuses on **data quality assessment** and **missing value handling**.
    
    ### Objectives:
    - Assess data quality metrics
    - Identify missing data patterns
    - Implement imputation strategies
    - Compare imputation methods
    """)
    
    st.info("🚧 **Status**: To be implemented by students")
    
    # Demo visualization placeholder
    st.markdown("### Expected Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Missing Data Heatmap")
        # Create sample missing data pattern
        np.random.seed(42)
        sample_data = np.random.rand(20, 8)
        sample_data[np.random.rand(20, 8) < 0.2] = np.nan
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(pd.DataFrame(sample_data).isnull(), cbar=True, 
                   yticklabels=False, cmap='viridis', ax=ax)
        ax.set_title('Missing Data Pattern (Example)')
        ax.set_xlabel('Features')
        ax.set_ylabel('Samples')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("#### Missing Value Percentages")
        # Sample missing percentages
        features = [f'Feature_{i}' for i in range(8)]
        missing_pct = np.random.uniform(0, 30, 8)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(features, missing_pct, color='steelblue', edgecolor='black')
        ax.set_xlabel('Missing Percentage (%)')
        ax.set_title('Missing Values by Feature (Example)')
        ax.grid(alpha=0.3, axis='x')
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    st.markdown("### Implementation Checklist")
    st.markdown("""
    - [ ] DataQualityAnalyzer class
    - [ ] Missing pattern identification
    - [ ] Simple imputation (mean/median/mode)
    - [ ] KNN imputation
    - [ ] Iterative imputation (MICE)
    - [ ] Comparison visualizations
    - [ ] Statistical summaries
    """)


def show_task2():
    """Display Task 2: Outlier Detection & Treatment."""
    st.markdown('<h1 class="task-header">🎯 Task 2: Outlier Detection & Treatment</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This task covers **outlier detection** methods and **treatment strategies**.
    
    ### Objectives:
    - Implement multiple detection methods
    - Apply treatment strategies
    - Compare effectiveness
    - Evaluate impact on models
    """)
    
    st.info("🚧 **Status**: To be implemented by students")
    
    # Demo visualization
    st.markdown("### Expected Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Box Plot with Outliers")
        # Generate sample data with outliers
        np.random.seed(42)
        data = np.concatenate([
            np.random.normal(50, 10, 100),
            np.random.uniform(100, 120, 5)
        ])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        bp = ax.boxplot(data, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        ax.set_ylabel('Values')
        ax.set_title('Box Plot with Outliers (Example)')
        ax.grid(alpha=0.3, axis='y')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("#### Outlier Detection Methods")
        # Sample detection comparison
        methods = ['Z-Score', 'IQR', 'Isolation\nForest', 'LOF']
        outliers_detected = [5, 6, 8, 7]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(methods, outliers_detected, color='coral', edgecolor='black')
        ax.set_ylabel('Number of Outliers Detected')
        ax.set_title('Outliers Detected by Method (Example)')
        ax.grid(alpha=0.3, axis='y')
        
        for bar, val in zip(bars, outliers_detected):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val}', ha='center', va='bottom', fontweight='bold')
        
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    st.markdown("### Implementation Checklist")
    st.markdown("""
    - [ ] OutlierAnalyzer class
    - [ ] Z-score detection
    - [ ] IQR detection
    - [ ] Isolation Forest
    - [ ] Local Outlier Factor
    - [ ] Treatment methods (cap/remove/transform)
    - [ ] Performance evaluation
    """)


def show_task3():
    """Display Task 3: Data Scaling & Normalization."""
    st.markdown('<h1 class="task-header">⚖️ Task 3: Data Scaling & Normalization</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This task explores **feature scaling** and **normalization** techniques.
    
    ### Objectives:
    - Implement scaling methods
    - Compare transformations
    - Evaluate impact on models
    - Provide recommendations
    """)
    
    st.info("🚧 **Status**: To be implemented by students")
    
    # Demo visualization
    st.markdown("### Expected Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Distribution Comparison")
        # Generate sample data
        np.random.seed(42)
        original = np.random.exponential(scale=20, size=200)
        standardized = (original - original.mean()) / original.std()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.hist(original, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.set_title('Original Data')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Frequency')
        
        ax2.hist(standardized, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
        ax2.set_title('Standardized Data')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Frequency')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("#### Scaling Methods Comparison")
        # Sample performance data
        methods = ['Original', 'MinMax', 'Standard', 'Robust', 'MaxAbs']
        scores = [0.72, 0.85, 0.88, 0.86, 0.83]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['gray', 'skyblue', 'gold', 'lightgreen', 'lightcoral']
        bars = ax.bar(methods, scores, color=colors, edgecolor='black')
        ax.set_ylabel('Model Performance (R²)')
        ax.set_title('Scaling Method Impact (Example)')
        ax.set_ylim([0.6, 1.0])
        ax.grid(alpha=0.3, axis='y')
        
        # Highlight best
        bars[2].set_edgecolor('darkgoldenrod')
        bars[2].set_linewidth(3)
        
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    st.markdown("### Implementation Checklist")
    st.markdown("""
    - [ ] ScalingAnalyzer class
    - [ ] Min-Max scaling
    - [ ] Standard scaling
    - [ ] Robust scaling
    - [ ] MaxAbs scaling
    - [ ] Power transformation
    - [ ] Model comparison
    """)


def show_task4():
    """Display Task 4: Feature Engineering."""
    st.markdown('<h1 class="task-header">🔧 Task 4: Feature Engineering</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This task focuses on **feature creation** and **transformation**.
    
    ### Objectives:
    - Create new features
    - Handle categorical variables
    - Generate interactions
    - Select best features
    """)
    
    st.info("🚧 **Status**: To be implemented by students")
    
    # Demo visualization
    st.markdown("### Expected Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Feature Importance")
        # Sample feature importance
        features = [f'Feature_{i}' for i in range(1, 11)]
        importance = np.random.uniform(0.02, 0.15, 10)
        importance = np.sort(importance)[::-1]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['gold' if i < 3 else 'steelblue' for i in range(10)]
        ax.barh(features[::-1], importance[::-1], color=colors[::-1], edgecolor='black')
        ax.set_xlabel('Importance Score')
        ax.set_title('Top 10 Feature Importance (Example)')
        ax.grid(alpha=0.3, axis='x')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("#### Encoding Comparison")
        # Sample encoding performance
        encodings = ['Label', 'One-Hot', 'Target', 'Frequency']
        accuracy = [0.78, 0.82, 0.85, 0.80]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(encodings, accuracy, color=['skyblue', 'lightgreen', 'gold', 'lightcoral'],
                     edgecolor='black')
        ax.set_ylabel('Model Accuracy')
        ax.set_title('Categorical Encoding Comparison (Example)')
        ax.set_ylim([0.7, 0.9])
        ax.grid(alpha=0.3, axis='y')
        
        # Highlight best
        bars[2].set_edgecolor('darkgoldenrod')
        bars[2].set_linewidth(3)
        
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    st.markdown("### Implementation Checklist")
    st.markdown("""
    - [ ] FeatureEngineeringPipeline class
    - [ ] Mathematical transformations
    - [ ] Interaction features
    - [ ] Categorical encoding methods
    - [ ] Polynomial features
    - [ ] Feature selection (3 methods)
    - [ ] Performance evaluation
    """)


def show_task5():
    """Display Task 5: Integrated Pipeline."""
    st.markdown('<h1 class="task-header">🔄 Task 5: Integrated Pipeline</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This task demonstrates an **end-to-end preprocessing pipeline** with proper cross-validation.
    
    ### Key Features:
    - Custom sklearn transformers
    - Data leakage prevention
    - Pipeline comparison
    - Hyperparameter optimization
    """)
    
    st.success("✅ **Status**: Fully implemented with 34/34 tests passing")
    
    try:
        from task5_pipeline import PreprocessingPipeline, create_sample_dataset_with_issues
        
        # Generate sample data
        with st.spinner("Generating sample dataset..."):
            X, y = create_sample_dataset_with_issues(
                n_samples=500,
                n_features=15,
                missing_rate=0.1,
                outlier_rate=0.05,
                random_state=42
            )
        
        st.markdown(f"**Dataset**: {X.shape[0]} samples × {X.shape[1]} features")
        
        # Controls
        st.markdown("### Pipeline Configuration")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            imputer_strategy = st.selectbox("Imputation", ['mean', 'median', 'knn'])
            outlier_method = st.selectbox("Outlier Detection", ['iqr', 'zscore'])
        
        with col2:
            scaler_type = st.selectbox("Scaling", ['standard', 'robust', 'minmax'])
            feature_engineering = st.checkbox("Feature Engineering", value=True)
        
        with col3:
            feature_selection = st.checkbox("Feature Selection", value=True)
            n_features = st.slider("Features to Select", 5, 20, 10)
        
        if st.button("Run Pipeline", type="primary"):
            with st.spinner("Running preprocessing pipeline..."):
                # Create and run pipeline
                pipeline = PreprocessingPipeline(random_state=42)
                pipeline.create_pipeline(
                    imputer_strategy=imputer_strategy,
                    outlier_method=outlier_method,
                    scaler_type=scaler_type,
                    feature_engineering=feature_engineering,
                    feature_selection=feature_selection,
                    n_features=n_features if feature_selection else None
                )
                
                X_train, X_test, y_train, y_test = pipeline.prepare_data(X, y, test_size=0.2)
                pipeline.fit_pipeline(X_train, y_train)
                metrics = pipeline.evaluate_pipeline(X_test, y_test)
                
                # Calculate additional metrics for display
                from sklearn.metrics import precision_score, recall_score, f1_score
                y_pred = pipeline.pipeline.predict(X_test)
                metrics['precision'] = precision_score(y_test, y_pred)
                metrics['recall'] = recall_score(y_test, y_pred)
                metrics['f1_score'] = f1_score(y_test, y_pred)
                
                # Display metrics
                st.markdown("### Results")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                with col2:
                    st.metric("Precision", f"{metrics['precision']:.3f}")
                with col3:
                    st.metric("Recall", f"{metrics['recall']:.3f}")
                with col4:
                    st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
                
                # Visualizations
                st.markdown("### Visualizations")
                
                tab1, tab2 = st.tabs(["Confusion Matrix", "Cross-Validation"])
                
                with tab1:
                    y_pred = pipeline.pipeline.predict(X_test)
                    fig = pipeline.visualize_confusion_matrix(y_test, y_pred)
                    st.pyplot(fig)
                    plt.close()
                
                with tab2:
                    cv_results = pipeline.cross_validate(X, y, cv=5)
                    fig = pipeline.visualize_cv_scores(cv_results['scores'])
                    st.pyplot(fig)
                    plt.close()
                
                st.success("✅ Pipeline completed successfully!")
    
    except ImportError as e:
        st.error(f"Could not import Task 5 module: {e}")
        st.info("Make sure task5_pipeline.py is in the src/ directory")


def show_task6():
    """Display Task 6: Real-World Case Study."""
    st.markdown('<h1 class="task-header">🌍 Task 6: Real-World Case Study</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This task applies preprocessing techniques to a **complex synthetic dataset** 
    with multiple quality issues.
    
    ### Key Features:
    - Comprehensive quality assessment
    - Multiple preprocessing strategies
    - Complete visualization suite
    - JSON report generation
    """)
    
    st.success("✅ **Status**: Fully implemented with 29/29 tests passing")
    
    try:
        from task6_case_study import DataPreprocessor, create_complex_synthetic_dataset
        
        # Controls
        st.markdown("### Dataset Configuration")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_samples = st.slider("Samples", 100, 2000, 500, step=100)
            n_features = st.slider("Features", 5, 30, 15)
        
        with col2:
            missing_rate = st.slider("Missing Rate", 0.0, 0.3, 0.15, step=0.05)
            outlier_rate = st.slider("Outlier Rate", 0.0, 0.2, 0.08, step=0.02)
        
        with col3:
            imbalance_ratio = st.slider("Class Imbalance", 0.1, 0.5, 0.3, step=0.05)
        
        if st.button("Generate & Preprocess Dataset", type="primary"):
            with st.spinner("Generating complex dataset..."):
                # Generate dataset
                df = create_complex_synthetic_dataset(
                    n_samples=n_samples,
                    n_features=n_features,
                    missing_rate=missing_rate,
                    outlier_rate=outlier_rate,
                    imbalance_ratio=imbalance_ratio,
                    random_state=42
                )
                
                preprocessor = DataPreprocessor(df, target_col='target')
                
                # Assess quality
                quality_report = preprocessor.assess_data_quality()
                
                # Display quality metrics
                st.markdown("### Data Quality Assessment")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Samples", quality_report['total_samples'])
                with col2:
                    st.metric("Total Features", quality_report['total_features'])
                with col3:
                    st.metric("Missing Values", quality_report['missing_values']['total'])
                with col4:
                    st.metric("Duplicates", quality_report['duplicates'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Numeric Features", len(quality_report['numeric_features']))
                with col2:
                    st.metric("Categorical Features", len(quality_report['categorical_features']))
                with col3:
                    st.metric("Class Balance", f"{quality_report['class_balance']:.3f}")
                
                # Preprocessing configuration
                st.markdown("### Preprocessing Configuration")
                config_col1, config_col2 = st.columns(2)
                
                with config_col1:
                    handle_missing = st.checkbox("Handle Missing Values", value=True)
                    missing_strategy = st.selectbox("Strategy", ['simple', 'knn', 'iterative'])
                    handle_outliers = st.checkbox("Handle Outliers", value=True)
                    encode_categorical = st.checkbox("Encode Categorical", value=True)
                
                with config_col2:
                    engineer_features = st.checkbox("Engineer Features", value=True)
                    scale_features = st.checkbox("Scale Features", value=True)
                    select_features = st.checkbox("Select Features", value=False)
                    balance_classes = st.checkbox("Balance Classes", value=False)
                
                # Run preprocessing
                if st.button("Run Preprocessing"):
                    with st.spinner("Running preprocessing pipeline..."):
                        config = {
                            'handle_missing': handle_missing,
                            'missing_strategy': missing_strategy,
                            'handle_outliers': handle_outliers,
                            'encode_categorical': encode_categorical,
                            'engineer_features': engineer_features,
                            'scale_features': scale_features,
                            'select_features': select_features,
                            'n_features': 15,
                            'balance_classes': balance_classes
                        }
                        
                        preprocessor.preprocess_pipeline(config)
                        
                        st.success(f"✅ Preprocessing complete! Final shape: {preprocessor.data.shape}")
                        
                        # Show preprocessing steps
                        st.markdown("### Preprocessing Steps")
                        for i, step in enumerate(preprocessor.preprocessing_steps, 1):
                            st.write(f"{i}. {step}")
                        
                        # Visualizations
                        st.markdown("### Visualizations")
                        
                        tab1, tab2, tab3, tab4 = st.tabs([
                            "EDA", "Correlations", "Missing Patterns", "PCA"
                        ])
                        
                        with tab1:
                            with st.spinner("Generating EDA visualizations..."):
                                fig = preprocessor.visualize_eda()
                                st.pyplot(fig)
                                plt.close()
                        
                        with tab2:
                            with st.spinner("Generating correlation heatmap..."):
                                fig = preprocessor.visualize_correlations()
                                st.pyplot(fig)
                                plt.close()
                        
                        with tab3:
                            with st.spinner("Analyzing missing patterns..."):
                                # Use original data for missing patterns
                                temp_preprocessor = DataPreprocessor(df, target_col='target')
                                fig = temp_preprocessor.visualize_missing_patterns()
                                st.pyplot(fig)
                                plt.close()
                        
                        with tab4:
                            with st.spinner("Generating PCA visualization..."):
                                fig = preprocessor.visualize_pca()
                                st.pyplot(fig)
                                plt.close()
    
    except ImportError as e:
        st.error(f"Could not import Task 6 module: {e}")
        st.info("Make sure task6_case_study.py is in the src/ directory")


if __name__ == "__main__":
    main()
