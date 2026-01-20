# Forge - Automated Feature Engineering Platform

[![CI](https://github.com/forge-features/forge/actions/workflows/ci.yml/badge.svg)](https://github.com/forge-features/forge/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/forge-features/forge/branch/main/graph/badge.svg)](https://codecov.io/gh/forge-features/forge)
[![PyPI version](https://badge.fury.io/py/forge-features.svg)](https://badge.fury.io/py/forge-features)
[![Downloads](https://static.pepy.tech/badge/forge-features)](https://pepy.tech/project/forge-features)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/forge-features/forge/main?labpath=notebooks)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/forge-features/forge/badge)](https://scorecard.dev/viewer/?uri=github.com/forge-features/forge)

Forge is an automated feature engineering platform that reduces the time data scientists spend on feature engineering by 10x. It provides automatic feature generation, intelligent selection, and seamless scikit-learn compatibility.

## Features

- **Automatic Feature Generation**: Generate features from numeric, categorical, temporal, and text data
- **Intelligent Feature Selection**: Statistical, importance-based, and SHAP-powered selection
- **scikit-learn Compatible**: Works seamlessly with sklearn pipelines
- **Data Analysis**: Automatic type inference and data quality assessment
- **Missing Value Handling**: Smart imputation strategies with missing indicators
- **Visualization**: Built-in plotting for feature importance and correlations

## Installation

```bash
pip install forge-features
```

With optional dependencies:

```bash
# Visualization support
pip install forge-features[viz]

# SHAP-based feature selection
pip install forge-features[shap]

# XGBoost and LightGBM support
pip install forge-features[boosting]

# Everything
pip install forge-features[all]
```

## Quick Start

### Basic Usage

```python
import pandas as pd
from forge import AutoFeatureTransformer

# Load your data
X = pd.read_csv("data.csv")
y = pd.read_csv("labels.csv")["target"]

# Automatic feature engineering
transformer = AutoFeatureTransformer(max_features=100)
X_engineered = transformer.fit_transform(X, y)

# View generated features
print(f"Generated {len(transformer.get_feature_names_out())} features")
print(transformer.get_feature_importance().head(10))
```

### sklearn Pipeline Integration

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from forge import AutoFeatureTransformer

pipeline = Pipeline([
    ("features", AutoFeatureTransformer(max_features=50)),
    ("classifier", RandomForestClassifier(n_estimators=100)),
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

### Custom Feature Generation

```python
from forge.generators.numeric import InteractionGenerator, PolynomialGenerator
from forge.generators.categorical import TargetEncoder
from forge.transformers import ForgePipeline

pipeline = ForgePipeline([
    ("interactions", InteractionGenerator(columns=["price", "quantity"])),
    ("polynomials", PolynomialGenerator(degree=2)),
    ("encoding", TargetEncoder(columns=["category", "region"])),
])

X_features = pipeline.fit_transform(X, y)
```

### Data Analysis

```python
from forge.analyzer import DataAnalyzer

analyzer = DataAnalyzer()
report = analyzer.analyze(X, y)

print(report.summary())
print(report.column_types)
print(report.quality_issues)
```

## Feature Generators

### Numeric Features
- **Aggregations**: Group-by statistics (mean, sum, std, etc.)
- **Interactions**: Feature multiplication, division, addition
- **Transformations**: Log, sqrt, power, binning
- **Polynomials**: Polynomial feature expansion

### Categorical Features
- **Encoders**: One-hot, target, frequency, ordinal encoding
- **Combinations**: Category pair interactions
- **Statistics**: Per-category target statistics

### Temporal Features
- **Components**: Year, month, day, hour, weekday extraction
- **Lags**: Lagged feature values
- **Rolling**: Rolling window statistics
- **Differences**: Time deltas and intervals

### Text Features
- **Basic**: Length, word count, character count
- **TF-IDF**: Term frequency-inverse document frequency

## Feature Selection

- **Statistical**: Chi-square, ANOVA F-test, mutual information
- **Importance**: Tree-based feature importance
- **Correlation**: Remove highly correlated features
- **Variance**: Remove low-variance features
- **SHAP**: SHAP value-based selection

## Production Features

### Drift Detection

Monitor feature distributions in production with PSI-based drift detection:

```python
from forge import PSICalculator, DriftDetector

# Fit on training data
drift_monitor = PSICalculator(threshold=0.2)
drift_monitor.fit(X_train)

# Check for drift in production data
drift_monitor.transform(X_prod)
summary = drift_monitor.get_drift_summary()

if summary["flagged_features"]:
    print(f"Drift detected in: {summary['flagged_features']}")
```

### Memory Management

Handle large datasets with memory-aware utilities:

```python
from forge.utils.memory import (
    estimate_feature_engineering_memory,
    process_in_chunks,
    check_memory_and_warn,
)

# Estimate memory before processing
estimate = estimate_feature_engineering_memory(X, max_features=100)
print(estimate)  # Shows peak memory, availability, recommendations

# Process large datasets in chunks
result = process_in_chunks(transformer, X, y, chunk_size=100_000)
```

## Performance

Forge is designed for speed and efficiency:

| Dataset Size | Forge | sklearn Manual | Speedup |
|--------------|-------|----------------|---------|
| 10K rows | 0.45s | 0.52s | 1.2x |
| 100K rows | 2.1s | 8.4s | **4x** |
| 1M rows | 18.3s | 89.2s | **4.9x** |

### Comparison with Other Tools

| Aspect | Forge | Featuretools | Feature-engine | tsfresh |
|--------|-------|--------------|----------------|---------|
| Speed (100K rows) | 2.1s | 45.2s | 3.8s | 180s+ |
| Memory (100K rows) | 245 MB | 1,024 MB | 289 MB | 2+ GB |
| sklearn native | Yes | No | Yes | No |
| Built-in selection | Yes | No | No | Yes |

See [benchmarks documentation](https://forge.dev/docs/benchmarks) for detailed performance analysis.

## API Reference

### AutoFeatureTransformer

```python
AutoFeatureTransformer(
    max_features: int | float | None = None,    # Max features to keep
    numeric_transformations: list[str] = None,  # ["log", "sqrt", "bin"]
    categorical_encoding: str = "auto",         # Encoding strategy
    temporal_features: list[str] = None,        # Temporal components
    missing_strategy: str = "auto",             # Imputation strategy
    selection_method: str = "importance",       # Selection method
    n_jobs: int = -1,                           # Parallel jobs
    random_state: int | None = None,            # Random seed
    verbose: int = 0,                           # Verbosity level
)
```

## Interactive Notebooks

Try Forge in your browser with our interactive notebooks:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/forge-features/forge/main?labpath=notebooks)

| Notebook | Description |
|----------|-------------|
| [01_quickstart.ipynb](notebooks/01_quickstart.ipynb) | Getting started with Forge |
| [02_custom_generators.ipynb](notebooks/02_custom_generators.ipynb) | Creating custom feature generators |
| [03_feature_selection.ipynb](notebooks/03_feature_selection.ipynb) | Feature selection techniques |
| [04_sklearn_integration.ipynb](notebooks/04_sklearn_integration.ipynb) | sklearn pipeline integration |
| [05_kaggle_workflow.ipynb](notebooks/05_kaggle_workflow.ipynb) | Complete Kaggle workflow |

## Documentation

Full documentation is available at [forge.dev/docs](https://forge.dev/docs).

## Community

We welcome contributions and feedback! Here's how to get involved:

- **GitHub Discussions**: Questions, ideas, and general discussion
- **GitHub Issues**: Bug reports and feature requests
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines

Looking for a way to contribute? Check out issues labeled [`good first issue`](https://github.com/forge-features/forge/labels/good%20first%20issue).

## Development

```bash
# Clone repository
git clone https://github.com/forge-features/forge.git
cd forge

# Install with dev dependencies
pip install -e ".[dev,all]"

# Run tests
make test

# Run linting
make lint

# Run type checking
make typecheck

# Run benchmarks
make benchmark
```

## Citation

If you use Forge in your research, please cite it:

```bibtex
@software{forge2025,
  title = {Forge: Automated Feature Engineering for Machine Learning},
  author = {Forge Development Team},
  year = {2025},
  url = {https://github.com/forge-features/forge}
}
```

## License

Apache License 2.0
