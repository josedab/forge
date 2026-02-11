# Installation

## Requirements

- Python 3.9 or higher
- pip

## Basic Installation

Install Forge from PyPI:

```bash
pip install forge-features
```

## Optional Dependencies

Forge has optional dependencies for additional features:

### Visualization

For plotting feature importance, correlations, and distributions:

```bash
pip install forge-features[viz]
```

### SHAP-based Selection

For SHAP value-based feature selection:

```bash
pip install forge-features[shap]
```

### Gradient Boosting

For XGBoost and LightGBM-based importance:

```bash
pip install forge-features[boosting]
```

### Recommended (Most Users)

Install visualization, SHAP, and gradient boosting — everything most users need without heavy dependencies like LLMs or distributed compute:

```bash
pip install forge-features[recommended]
```

### All Features

Install everything (including LLM, distributed, and dashboard dependencies):

```bash
pip install forge-features[all]
```

## Development Installation

For contributing to Forge:

```bash
# Clone the repository
git clone https://github.com/forge-features/forge.git
cd forge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev,all]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

## Verifying Installation

Verify your installation works:

```python
from forge import __version__
print(f"Forge version: {__version__}")
```

Or run the smoke tests:

```bash
pytest tests/smoke_test.py -v
```

## Troubleshooting

### ImportError: No module named 'forge'

Make sure you installed with `pip install forge-features` (not just `forge`).

### Missing Optional Dependencies

If you get import errors for visualization or SHAP features, install the relevant optional dependencies:

```bash
pip install forge-features[viz]  # For matplotlib/seaborn
pip install forge-features[shap]  # For SHAP
```
