# Practical Labs 2
## Data Preprocessing. Visualization and Data Exploration

## Setup Instructions

### Creating and Activating Virtual Environment

1. **Create a virtual environment:**
   ```powershell
   python -m venv venv
   ```

2. **Activate the virtual environment:**
   
   **Windows (PowerShell):**
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
   
   **Windows (Command Prompt):**
   ```cmd
   venv\Scripts\activate.bat
   ```
   
   **Linux/Mac:**
   ```bash
   source venv/bin/activate
   ```

3. **Upgrade pip and install dependencies:**
   ```powershell
   python -m pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```powershell
   python -c "import numpy, pandas, matplotlib, seaborn, sklearn; print('All packages installed successfully!')"
   ```

### Running the Interactive Streamlit App

This repository includes an **interactive visualization dashboard** built with Streamlit:

```powershell
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

**Features:**
- Interactive visualizations for all tasks
- Configurable parameters with sliders and dropdowns
- Real-time preprocessing and visualization
- Task selection from sidebar
- Full integration with Tasks 5 & 6 implementations

**App Structure:**
- **Home**: Overview and implementation status
- **Task 1-4**: Template visualizations and checklists
- **Task 5**: Interactive pipeline with live configuration
- **Task 6**: Complete preprocessing workflow with quality assessment

---

## Course Overview

This week focuses on **Data Preprocessing, Visualization, and Data Exploration** - fundamental skills for any data scientist or machine learning practitioner. The course consists of 6 progressive tasks:

1. **Task 1**: Data Quality Assessment & Missing Value Handling
2. **Task 2**: Outlier Detection & Treatment
3. **Task 3**: Data Scaling & Normalization
4. **Task 4**: Feature Engineering Pipeline
5. **Task 5**: Integrated Preprocessing Pipeline (with models)
6. **Task 6**: Real-World Case Study (preprocessing only)

---

## Course Tasks: Data Cleaning, Scaling, and Feature Engineering

### Task 1: Comprehensive Data Quality Assessment and Missing Value Analysis

**Objective**: Analyze a dataset to identify and visualize data quality issues, then implement and compare multiple missing value handling strategies.

**Requirements**:
- Generate or load a dataset with various data quality issues (duplicates, missing values)
- Assess overall data quality (total rows/columns, duplicates, missing value counts and percentages, memory usage)
- Identify and classify missing data patterns (MCAR, MAR, MNAR) using statistical correlation analysis
- Implement multiple imputation techniques:
  - Simple imputation (mean, median, mode)
  - KNN imputation (with configurable neighbors)
  - Multivariate imputation using IterativeImputer (MICE algorithm)
- Create binary indicator variables to flag missing data patterns
- Compare the statistical properties (mean, std, min, max) before and after each imputation method

**Data Visualization Requirements**:
- **Missing data matrix heatmap**: Visualize missing data patterns across all features using missingno library
- **Bar plot**: Show the percentage of missing values per column
- **Missing data dendrogram**: Show hierarchical clustering of missing value patterns
- **Correlation heatmap of missing indicators**: Identify relationships between missingness patterns
- **Distribution plots (histograms)**: Compare distributions before and after imputation for each method
- **Correlation matrix heatmaps**: Compare correlation structures before and after each imputation method

**Deliverables**:
- Python module (`task1_data_cleaning.py`) with complete implementation
- `DataQualityAnalyzer` class with reusable methods
- Generated visualization files (PNG format)
- Statistical comparison tables

---

### Task 2: Outlier Detection and Treatment with Advanced Visualization

**Objective**: Detect, analyze, and handle outliers using multiple statistical and machine learning-based techniques, then evaluate their impact on model performance.

**Requirements**:
- Implement and compare 4 outlier detection methods:
  - Statistical Z-score method (configurable threshold, default: 3.0)
  - Interquartile Range (IQR) method (configurable multiplier, default: 1.5)
  - Isolation Forest (ML-based, configurable contamination rate)
  - Local Outlier Factor (LOF) with configurable neighbors
- Implement multiple outlier handling strategies:
  - Complete removal of outlier rows
  - Capping/Winsorization (clipping at IQR bounds)
  - Log transformation (log1p with automatic shift for negative values)
  - Power transformation (Yeo-Johnson method)
- Generate outlier summary statistics across all detection methods
- Evaluate model performance (Linear Regression with cross-validation) on original vs. treated datasets
- Compare the effectiveness of each treatment strategy

**Data Visualization Requirements**:
- **Box plots**: Show outliers in up to 6 numerical features with original data
- **Violin plots**: Display distribution density with means and medians for multiple features
- **2D scatter plots**: Color-coded outliers from Z-score, IQR, and Isolation Forest methods
- **3D scatter plots**: Visualize multivariate outliers in 3D feature space
- **Treatment comparison histograms**: Grid showing distributions before and after each treatment (original, removed, capped, log, Box-Cox)
- **Q-Q plots**: Assess normality for up to 6 features using probability plots

**Deliverables**:
- Python module (`task2_outlier_detection.py`) with complete implementation
- `OutlierAnalyzer` class with reusable detection and treatment methods
- Generated visualization files (6 PNG files)
- Performance comparison report showing R² scores across treatments

---

### Task 3: Comprehensive Data Scaling and Normalization Study

**Objective**: Implement and systematically compare different scaling techniques, analyzing their mathematical properties and impact on machine learning algorithms.

**Requirements**:
- Implement 5 scaling techniques with proper sklearn integration:
  - Min-Max Scaling (customizable feature range, default: [0, 1])
  - Z-Score Normalization/Standardization (mean=0, std=1)
  - Robust Scaling (using median and IQR, resistant to outliers)
  - MaxAbs Scaling (scale by maximum absolute value to [-1, 1])
  - Power Transformation (Yeo-Johnson method for non-Gaussian data)
- Preserve target column (no scaling applied to target)
- Store fitted scalers for reuse and inverse transformation
- Generate comprehensive scaling summary statistics (mean, std, min, max)
- Evaluate multiple model types with cross-validation:
  - Linear Regression (for regression tasks)
  - K-Nearest Neighbors (distance-based, sensitive to scaling)
  - Support Vector Machines (optional)
  - Neural Networks (optional)
- Compare performance across all scaling methods and original data

**Data Visualization Requirements**:
- **Distribution histograms**: Compare distributions of one feature across all scaling methods (original + 5 scaled)
- **Box plots**: Show range and spread of up to 6 features across different scaling methods
- **Correlation heatmaps**: Visualize correlation structure preservation under different scaling methods
- **Performance bar charts**: Compare average model scores by scaling method and grouped by model type
- **2D feature space plots**: Visualize how scaling transforms the feature space (up to 3 methods shown)
- **3D feature space plots**: Show feature space transformation in 3D with color-coded target classes

**Deliverables**:
- Python module (`task3_scaling.py`) with complete implementation
- `ScalingAnalyzer` class with all scaling methods and evaluation capabilities
- Generated visualization files (6 PNG files)
- Performance comparison DataFrame with scores and standard deviations
- Decision guide with recommendations:
  * Min-Max: Neural networks, algorithms sensitive to feature ranges
  * Standard: SVM, Linear Regression, algorithms assuming normal distribution
  * Robust: Data with outliers
  * Power Transform: Making data more Gaussian-like

---

### Task 4: Advanced Feature Engineering Pipeline

**Objective**: Design and implement a comprehensive feature engineering pipeline that creates new features, handles categorical variables, and captures feature interactions.

**Requirements**:
- Implement `FeatureEngineeringPipeline` class with comprehensive transformation methods:
  - **Mathematical transformations**: log, sqrt, square, cube, reciprocal (with automatic handling of negative values)
  - **Interaction features**: multiply, add, subtract, divide, ratio operations between feature pairs
  - **Aggregation features**: groupby operations with mean, std, min, max, count functions
  - **Categorical encoding**:
    * One-Hot Encoding (with max_categories limit and drop_first option)
    * Label Encoding (preserving original columns)
    * Target Encoding (with smoothing parameter to prevent overfitting)
    * Frequency Encoding (normalized category frequencies)
  - **Polynomial features**: degree 2 or 3 with sklearn PolynomialFeatures integration
- Three feature selection methods implemented:
  - Univariate statistical tests (f_classif, mutual_info_classif)
  - Recursive Feature Elimination (RFE with Logistic Regression)
  - Model-based (Random Forest feature importance)
- Automatic feature catalog generation documenting all engineered features
- Performance evaluation comparing original vs. engineered features with cross-validation

**Data Visualization Requirements**:
- **Feature importance bar charts**: Top 15 features ranked by Random Forest and univariate F-scores
- **Pair plots**: Visualize relationships between interaction features, colored by target class
- **Correlation heatmaps**: Show correlation structure of top 20 selected features
- **Polynomial impact plots**: Before/after comparison showing polynomial fit capturing non-linear patterns
- **Encoding comparison bar charts**: Model accuracy comparison across Label, One-Hot, and Frequency encoding
- **Feature network graph**: Visualize feature interaction dependencies (requires NetworkX)

**Deliverables**:
- Python module (`task4_feature_engineering.py`) with complete implementation
- `FeatureEngineeringPipeline` class with all transformation and selection methods
- Generated visualization files (6-7 PNG files including network graph if NetworkX available)
- Feature catalog DataFrame with descriptions for all engineered features
- Performance improvement analysis showing accuracy gains from feature engineering

---

### Task 5: Integrated Data Preprocessing Pipeline with Cross-Validation

**Objective**: Build an end-to-end data preprocessing pipeline that combines cleaning, scaling, and feature engineering with proper cross-validation to prevent data leakage.

**Implementation Overview**:
The solution implements a production-ready `PreprocessingPipeline` class that orchestrates multiple preprocessing steps in a single sklearn-compatible pipeline. The implementation includes custom transformers and comprehensive evaluation methods.

**Key Components**:

1. **Custom Transformers** (sklearn-compatible BaseEstimator, TransformerMixin):
   - `OutlierHandler`: Detects and handles outliers using IQR or Z-score methods
     * Strategies: capping (winsorization) or removal (mark as NaN for imputation)
     * Configurable thresholds for detection sensitivity
   - `FeatureEngineer`: Creates new features through transformations
     * Mathematical transformations: log, square root
     * Interaction features: pairwise feature products (limited to avoid explosion)
     * Optional polynomial features for non-linear patterns

2. **PreprocessingPipeline Class** with comprehensive methods:
   - `create_pipeline()`: Configurable pipeline builder with 5 preprocessing steps
     * Step 1: Imputation (mean, median, or KNN)
     * Step 2: Outlier handling (IQR or Z-score based)
     * Step 3: Feature engineering (optional, with interactions/transformations)
     * Step 4: Scaling (Standard, Robust, or MinMax)
     * Step 5: Feature selection (SelectKBest with f_classif)
   - `prepare_data()`: Stratified train-test splitting
   - `fit_pipeline()`: Fits pipeline on training data only (prevents data leakage)
   - `cross_validate()`: 5-fold stratified cross-validation with score statistics
   - `compare_configurations()`: Evaluates multiple pipeline configurations in parallel
   - `optimize_hyperparameters()`: GridSearchCV for preprocessing and model parameters
   - `get_learning_curves()`: Generates learning curves with multiple training set sizes
   - `save_pipeline()` / `load_pipeline()`: Serialization using joblib for deployment

3. **Data Leakage Prevention**:
   - All transformers fitted exclusively on training data
   - Cross-validation properly isolates each fold
   - Test set never used for fitting any preprocessing step
   - Custom transformers implement fit() and transform() separately

4. **Sample Data Generation**:
   - `create_sample_dataset_with_issues()`: Generates synthetic classification data
   - Controllable injection of missing values (default: 10%)
   - Controllable injection of outliers (default: 5%)
   - Binary classification task with configurable feature count

**Data Visualization Requirements**:
- **Confusion matrices**: Heatmaps showing classification performance before/after preprocessing
- **ROC curves**: ROC-AUC visualization with area under curve calculation
- **Learning curves**: Training vs validation scores across different training set sizes
  * Includes confidence intervals (±1 std) to assess variance
  * Helps identify overfitting/underfitting patterns
- **Cross-validation box plots**: Distribution of CV scores across folds with individual points
- **Pipeline comparison bar charts**: Side-by-side comparison of different configurations
  * Best configuration highlighted in gold
  * Error bars showing standard deviation across CV folds
  * Value labels on each bar for precise comparison

**Delivered Features**:
- Python module (`task5_pipeline.py`) with 910 lines of production-ready code
- `PreprocessingPipeline` class with 15+ methods for complete preprocessing workflow
- 2 custom sklearn-compatible transformers (OutlierHandler, FeatureEngineer)
- 5 visualization methods generating publication-quality plots
- Saved pipeline object (`.pkl` file) ready for deployment
- Performance comparison showing improvement from baseline to optimized pipeline

**Usage Example**:
```python
# Initialize and configure pipeline
pipeline = PreprocessingPipeline(random_state=42)
pipeline.create_pipeline(
    imputer_strategy='knn',
    outlier_method='iqr',
    scaler_type='robust',
    feature_engineering=True,
    feature_selection=True,
    n_features=20
)

# Split data with stratification
X_train, X_test, y_train, y_test = pipeline.prepare_data(X, y, test_size=0.2)

# Fit pipeline (no data leakage)
pipeline.fit_pipeline(X_train, y_train)

# Evaluate and visualize
metrics = pipeline.evaluate_pipeline(X_test, y_test)
pipeline.visualize_confusion_matrix(y_test, y_pred)
pipeline.visualize_roc_curve(y_test, y_pred_proba)

# Save for deployment
pipeline.save_pipeline('preprocessing_pipeline.pkl')
```

**Key Results**:
- Successfully demonstrates prevention of data leakage through proper workflow
- Configuration comparison identifies optimal preprocessing combination
- Learning curves validate model generalization
- Cross-validation provides robust performance estimates
- Production-ready pipeline achieves consistent 80%+ accuracy on synthetic data
- All preprocessing steps properly integrated in single reusable pipeline

**Deliverables**:
- Python module (`src/task5_pipeline.py`) - 910 lines of production code
- `PreprocessingPipeline` class with 15+ methods
- Custom sklearn transformers: `OutlierHandler`, `FeatureEngineer`
- Comprehensive test suite (`tests/test_task5_pipeline.py`) with 34 unit tests (100% passing)
- 5 visualization methods with generated plots:
  * Confusion matrix heatmap
  * ROC curve with AUC
  * Learning curves with confidence intervals
  * Cross-validation score distributions
  * Pipeline configuration comparison
- Serialized pipeline object (`preprocessing_pipeline.pkl`)
- Sample dataset generation function
- Complete documentation with usage examples

---

### Task 6: Real-World Case Study - Complete Data Preprocessing Challenge

**Objective**: Apply all learned techniques to a complex, real-world dataset with multiple data quality issues, requiring strategic decision-making throughout the preprocessing workflow.

#### Implementation Overview

This task implements a complete real-world data preprocessing workflow through the `DataPreprocessor` class, which provides comprehensive data quality assessment, preprocessing methods, and visualization capabilities.

#### Key Components

**1. DataPreprocessor Class**

The main class that orchestrates the entire preprocessing workflow:

```python
from task6_case_study import DataPreprocessor, create_complex_synthetic_dataset

# Generate complex synthetic dataset
df = create_complex_synthetic_dataset(
    n_samples=2000,
    n_features=25,
    missing_rate=0.15,
    outlier_rate=0.08,
    imbalance_ratio=0.3
)

# Initialize preprocessor
preprocessor = DataPreprocessor(df, target_col='target')
```

**2. Data Quality Assessment**

Comprehensive analysis of data quality issues:

```python
# Assess data quality
quality_report = preprocessor.assess_data_quality()

# Print summary
preprocessor.print_quality_summary()
```

**3. Preprocessing Pipeline Configuration**

Configure and execute the complete preprocessing workflow:

```python
config = {
    'handle_missing': True,
    'missing_strategy': 'iterative',  # 'simple', 'knn', 'iterative'
    'handle_outliers': True,
    'outlier_method': 'iqr',  # 'iqr', 'zscore'
    'outlier_action': 'cap',  # 'cap', 'remove'
    'encode_categorical': True,
    'encoding_method': 'auto',  # 'auto', 'onehot', 'label'
    'engineer_features': True,
    'scale_features': True,
    'scaling_method': 'standard',  # 'standard', 'robust'
    'select_features': True,
    'n_features': 15,
    'feature_selection_method': 'kbest',  # 'kbest', 'mutual_info'
    'balance_classes': True,
    'balance_method': 'undersample'  # 'undersample', 'oversample'
}

result = preprocessor.preprocess_pipeline(config)
```

**4. Visualization Suite**

Five comprehensive visualization methods:

```python
# 1. Exploratory Data Analysis
preprocessor.visualize_eda()

# 2. Correlation Heatmap
preprocessor.visualize_correlations()

# 3. Missing Value Patterns
preprocessor.visualize_missing_patterns()

# 4. PCA Dimensionality Reduction
preprocessor.visualize_pca(n_components=2)
```

**5. Report Generation**

Generate comprehensive JSON report:

```python
# Generate report
report = preprocessor.generate_report(filepath='preprocessing_report.json')
```

#### Preprocessing Methods

**Individual preprocessing steps** (all methods update `self.data` automatically):

```python
# 1. Missing value handling
preprocessor.handle_missing_values(strategy='iterative')

# 2. Outlier detection and treatment
preprocessor.detect_and_handle_outliers(method='iqr', action='cap')

# 3. Categorical encoding (also handles datetime features)
preprocessor.encode_categorical_features(method='auto')

# 4. Feature engineering
preprocessor.engineer_features(domain_features=True)

# 5. Feature scaling
preprocessor.scale_features(method='standard')

# 6. Feature selection
preprocessor.select_features(method='kbest', n_features=15)

# 7. Class balancing
preprocessor.balance_classes(method='undersample')
```

#### Data Generation

Create complex synthetic datasets with realistic quality issues:

```python
df = create_complex_synthetic_dataset(
    n_samples=2000,
    n_features=25,
    missing_rate=0.15,      # 15% missing values
    outlier_rate=0.08,      # 8% outliers
    imbalance_ratio=0.3,    # 30/70 class split
    random_state=42
)
```

The generated dataset includes:
- Numeric features (normal, exponential, uniform distributions)
- Categorical features (multiple cardinality levels)
- Datetime features
- Target variable with class imbalance
- Injected missing values
- Injected outliers
- Duplicate rows

#### Delivered Features

**Complete preprocessing workflow**
- Modular preprocessing methods
- Configurable pipeline execution
- Automatic data quality assessment

**Advanced imputation strategies**
- Simple imputation (mean/median/mode)
- KNN imputation
- Iterative imputation (MICE algorithm)

**Comprehensive outlier handling**
- IQR method
- Z-score method
- Cap or remove actions

**Intelligent categorical encoding**
- Automatic method selection based on cardinality
- One-hot encoding for low cardinality
- Label encoding for high cardinality
- Datetime feature extraction (year, month, day, dayofweek)

**Feature engineering**
- Statistical features (mean, std, min, max across features)
- Interaction features (products and ratios)

**Multiple scaling methods**
- Standard scaling (z-score normalization)
- Robust scaling (using median and IQR)

**Feature selection**
- SelectKBest with f_classif
- Mutual information-based selection

**Class balancing**
- Random undersampling
- Random oversampling

**Visualization suite**
- EDA dashboard (distributions, value counts)
- Correlation heatmaps
- Missing value analysis
- PCA visualization

**Report generation**
- JSON format with all metrics and statistics
- Quality assessment details
- Preprocessing steps log

#### Usage Example

```python
from task6_case_study import DataPreprocessor, create_complex_synthetic_dataset

# 1. Generate data
df = create_complex_synthetic_dataset(n_samples=2000, random_state=42)

# 2. Initialize preprocessor
preprocessor = DataPreprocessor(df, target_col='target')

# 3. Assess quality
preprocessor.assess_data_quality()
preprocessor.print_quality_summary()

# 4. Configure and run pipeline
config = {
    'handle_missing': True,
    'missing_strategy': 'iterative',
    'handle_outliers': True,
    'encode_categorical': True,
    'engineer_features': True,
    'scale_features': True,
    'select_features': True,
    'n_features': 15,
    'balance_classes': True
}
preprocessor.preprocess_pipeline(config)

# 5. Visualize results
preprocessor.visualize_eda()
preprocessor.visualize_correlations()
preprocessor.visualize_missing_patterns()
preprocessor.visualize_pca()

# 6. Generate report
preprocessor.generate_report('results/preprocessing_report.json')
```

#### Key Design Decisions

1. **Automatic datetime handling**: Datetime columns are automatically detected and converted to numeric features (year, month, day, dayofweek) during categorical encoding
2. **Stateful preprocessing**: All preprocessing methods update `self.data` automatically, allowing method chaining
3. **Quality-first approach**: Data quality assessment is performed before preprocessing to inform strategy selection
4. **Comprehensive testing**: Unit tests covering all functionality
5. **Modular design**: Each preprocessing step is independent and can be called individually or as part of the pipeline
6. **No model training**: Focuses purely on preprocessing without ML models (suitable for early-stage students)

**Deliverables**:
- Python module (`src/task6_case_study.py`) - 911 lines of complete implementation
- `DataPreprocessor` class with 15+ preprocessing and visualization methods
- Complex synthetic dataset generator with realistic quality issues
- 4 visualization methods with generated plots:
  * EDA dashboard (distributions and value counts)
  * Correlation heatmap
  * Missing value patterns analysis
  * PCA dimensionality reduction
- JSON report generation with all metrics and preprocessing steps
- Complete documentation and usage examples
- `main()` function demonstrating complete workflow

**Preprocessing Methods Implemented**:
1. `assess_data_quality()` - Comprehensive quality assessment
2. `handle_missing_values()` - Simple, KNN, and iterative imputation
3. `detect_and_handle_outliers()` - IQR and Z-score methods
4. `encode_categorical_features()` - Auto, one-hot, and label encoding
5. `engineer_features()` - Statistical and interaction features
6. `scale_features()` - Standard and robust scaling
7. `select_features()` - K-best and mutual information
8. `balance_classes()` - Undersample and oversample
9. `preprocess_pipeline()` - Complete workflow orchestration
10. `generate_report()` - JSON report with all details

**Visualization Methods**:
1. `visualize_eda()` - Distributions and categorical counts
2. `visualize_correlations()` - Feature correlation heatmap
3. `visualize_missing_patterns()` - Missing data analysis
4. `visualize_pca()` - PCA with variance explained

---

## Recommended Tools & Libraries

- **Data Manipulation**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly, missingno, streamlit
- **Preprocessing**: scikit-learn, category_encoders
- **Imputation**: scikit-learn (SimpleImputer, KNNImputer, IterativeImputer), fancyimpute
- **Outlier Detection**: scikit-learn (IsolationForest, LocalOutlierFactor), PyOD
- **Feature Engineering**: feature-engine, featuretools
- **Model Interpretation**: SHAP, eli5
- **Interactive Dashboards**: streamlit (included in this repo)

### Streamlit App Features

The included **interactive dashboard** (`app.py`) provides:
- 🏠 **Home**: Overview of all tasks with implementation status
- 📋 **Task 1-4**: Template visualizations and implementation checklists
- 🔄 **Task 5**: Interactive pipeline builder with real-time configuration
  - Configure imputation, outlier handling, scaling, and feature engineering
  - Run preprocessing and see results instantly
  - View confusion matrices and cross-validation scores
- 🌍 **Task 6**: Complete preprocessing workflow
  - Generate synthetic datasets with configurable parameters
  - Real-time data quality assessment
  - Interactive preprocessing configuration
  - Multiple visualization tabs (EDA, correlations, missing patterns, PCA)

**Usage Tips**:
- Use sidebar dropdown to navigate between tasks
- Adjust sliders and checkboxes to configure preprocessing
- Click "Run" buttons to execute pipelines
- Visualizations update in real-time
- Perfect for exploring preprocessing impact on data

---

## Learning Resources

### Official Documentation:
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)

### Recommended Reading:
- "Feature Engineering for Machine Learning" by Alice Zheng & Amanda Casari
- "Hands-On Machine Learning" by Aurélien Géron (Chapters on Data Preprocessing)
- Scikit-learn Preprocessing Guide
- Towards Data Science articles on feature engineering

### Useful Tools:
- Jupyter Notebook for interactive development
- VS Code with Python extension
- pytest for testing
- black for code formatting
- flake8 for linting

---

## Tips for Success

1. **Start Simple**: Implement basic functionality first, then add complexity
2. **Test Frequently**: Write and run tests as you develop
3. **Visualize Often**: Create plots to understand your data and results
4. **Document As You Go**: Write docstrings and comments while coding
5. **Handle Edge Cases**: Consider missing values, outliers, empty datasets
6. **Compare Methods**: Always compare multiple approaches
7. **Validate Results**: Ensure preprocessing improves model performance
8. **Ask Questions**: Clarify requirements before starting implementation
9. **Review Examples**: Study Tasks 5 and 6 implementations as references
10. **Use Version Control**: Commit your work regularly

Good luck with your advanced ML and DS journey!
