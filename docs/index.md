# Forge

**Automated Feature Engineering Platform for Machine Learning**

[![CI](https://github.com/forge-features/forge/actions/workflows/ci.yml/badge.svg)](https://github.com/forge-features/forge/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/forge-features.svg)](https://badge.fury.io/py/forge-features)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Forge reduces the time data scientists spend on feature engineering by 10x. It provides automatic feature generation, intelligent selection, and seamless scikit-learn compatibility.

## Features

- **Automatic Feature Generation**: Generate features from numeric, categorical, temporal, and text data
- **Intelligent Feature Selection**: Statistical, importance-based, and SHAP-powered selection
- **scikit-learn Compatible**: Works seamlessly with sklearn pipelines
- **Data Analysis**: Automatic type inference and data quality assessment
- **Missing Value Handling**: Smart imputation strategies with missing indicators
- **Visualization**: Built-in plotting for feature importance and correlations

## Quick Start

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

## Installation

```bash
pip install forge-features
```

With optional dependencies:

```bash
# All optional features
pip install forge-features[all]
```

## Next Steps

- [Installation Guide](getting-started/installation.md) - Detailed installation instructions
- [Quick Start](getting-started/quickstart.md) - Get up and running in minutes
- [User Guide](guide/generation.md) - Learn about feature generation
- [API Reference](api/auto_transformer.md) - Detailed API documentation
