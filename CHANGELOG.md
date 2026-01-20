# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-01-18

### Added

- **Core Framework**
  - `AutoFeatureTransformer` - Main entry point for automatic feature engineering
  - `ForgePipeline` - Custom pipeline for chaining feature transformations
  - Full scikit-learn compatibility (BaseEstimator, TransformerMixin)

- **Data Analysis**
  - `DataAnalyzer` - Automatic data profiling and analysis
  - Automatic column type inference (numeric, categorical, temporal, text)
  - Statistical profiling and data quality assessment
  - Analysis reports with actionable insights

- **Feature Generators**
  - Numeric: Aggregations, interactions, transformations, polynomials
  - Categorical: One-hot, target, frequency, ordinal encoding
  - Temporal: Components extraction, lags, rolling windows, differences
  - Text: Basic stats (length, word count), TF-IDF vectorization

- **Feature Selectors**
  - Statistical: Chi-square, ANOVA F-test, mutual information
  - Importance-based: Tree-based feature importance
  - Correlation: Remove highly correlated features
  - Variance: Remove low-variance features
  - SHAP: SHAP value-based selection (optional)

- **Missing Value Handling**
  - Multiple imputation strategies (mean, median, mode, constant, knn)
  - Missing value indicators

- **Visualization** (optional)
  - Feature importance plots
  - Correlation heatmaps
  - SHAP visualizations
  - Distribution plots

- **Utilities**
  - Input validation
  - Parallel processing support
  - Pipeline serialization (save/load)
  - Configurable logging

### Dependencies

- Core: numpy, pandas, scikit-learn, scipy
- Optional: matplotlib, seaborn (viz), shap, xgboost, lightgbm (boosting)

[Unreleased]: https://github.com/forge-features/forge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/forge-features/forge/releases/tag/v0.1.0
