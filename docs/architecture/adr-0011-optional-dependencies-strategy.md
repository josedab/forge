# ADR-0011: Optional Dependencies Strategy

## Status

Accepted

## Context

Forge has functionality that requires heavy dependencies:

| Feature | Dependencies | Size |
|---------|--------------|------|
| Visualization | matplotlib, seaborn | ~50MB |
| SHAP selection | shap | ~30MB |
| Gradient boosting | xgboost, lightgbm | ~100MB |
| Documentation | mkdocs, mkdocs-material | ~40MB |

Installing all dependencies:
- Increases install time significantly
- Uses more disk space
- May cause dependency conflicts
- Pulls in transitive dependencies users don't need

We needed a strategy to provide these features without burdening all users.

## Decision

Use **optional dependency groups** via pip extras:

```toml
# pyproject.toml
[project.optional-dependencies]
viz = [
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
]
shap = [
    "shap>=0.42.0",
]
boosting = [
    "xgboost>=2.0.0",
    "lightgbm>=4.0.0",
]
all = [
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "shap>=0.42.0",
    "xgboost>=2.0.0",
    "lightgbm>=4.0.0",
]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-benchmark>=4.0.0",
    "hypothesis>=6.100.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
]
docs = [
    "mkdocs>=1.5.0",
    "mkdocs-material>=9.5.0",
    "mkdocstrings[python]>=0.24.0",
]
```

### Installation Patterns

```bash
# Base installation - core functionality only
pip install forge-features

# With visualization
pip install forge-features[viz]

# With SHAP-based selection
pip install forge-features[shap]

# With boosting models for importance
pip install forge-features[boosting]

# Everything for full functionality
pip install forge-features[all]

# For development
pip install forge-features[dev,all]

# Multiple extras
pip install forge-features[viz,shap]
```

### Graceful Degradation

Features check for dependencies at runtime:

```python
# forge/visualization/importance.py

def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 20):
    """Plot feature importance bar chart.

    Requires: matplotlib, seaborn (install with `pip install forge-features[viz]`)
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as e:
        raise MissingDependencyError(
            "Visualization requires matplotlib and seaborn. "
            "Install with: pip install forge-features[viz]"
        ) from e

    # Plotting logic...
```

```python
# forge/selectors/shap_selector.py

class ShapSelector(BaseFeatureSelector):
    """SHAP-based feature selection.

    Requires: shap (install with `pip install forge-features[shap]`)
    """

    def __init__(self, max_features: int = 100, ...):
        try:
            import shap
        except ImportError as e:
            raise MissingDependencyError(
                "ShapSelector requires the shap package. "
                "Install with: pip install forge-features[shap]"
            ) from e
        self.max_features = max_features
        ...
```

### Import Guards

Lazy imports prevent errors at module load time:

```python
# forge/__init__.py

def __getattr__(name: str) -> object:
    if name == "ShapSelector":
        # Only import when accessed
        from forge.selectors.shap_selector import ShapSelector
        return ShapSelector

    if name == "plot_feature_importance":
        from forge.visualization.importance import plot_feature_importance
        return plot_feature_importance
```

## Consequences

### Positive

1. **Minimal base install**: Core functionality installs quickly:
   ```bash
   # ~5 seconds, ~20MB
   pip install forge-features

   # vs ~30 seconds, ~200MB for everything
   pip install forge-features[all]
   ```

2. **User choice**: Users install what they need:
   ```bash
   # Data scientist who doesn't need visualization
   pip install forge-features

   # Data scientist who needs plots
   pip install forge-features[viz]

   # ML engineer who needs SHAP explanations
   pip install forge-features[shap]
   ```

3. **Fewer conflicts**: Reduced dependency surface:
   ```bash
   # User has matplotlib 3.5, forge needs 3.7
   # No conflict if user doesn't install [viz]
   ```

4. **Clear error messages**: Users know exactly what to install:
   ```python
   >>> from forge import ShapSelector
   MissingDependencyError: ShapSelector requires the shap package.
   Install with: pip install forge-features[shap]
   ```

5. **Documentation clarity**: Features document their requirements:
   ```python
   def plot_correlation_heatmap(df):
       """Plot correlation heatmap.

       Requires: matplotlib, seaborn
       Install: pip install forge-features[viz]
       """
   ```

### Negative

1. **User confusion**: Multiple install options can confuse:
   ```bash
   # Which do I need?
   pip install forge-features
   pip install forge-features[viz]
   pip install forge-features[all]
   ```

2. **Runtime errors**: Missing dependencies discovered at runtime:
   ```python
   # Works
   from forge import AutoFeatureTransformer

   # Fails if shap not installed
   from forge import ShapSelector  # MissingDependencyError
   ```

3. **Testing complexity**: Must test with and without optional deps:
   ```yaml
   # CI matrix
   - python: 3.11
     extras: ""  # Core only
   - python: 3.11
     extras: "[viz]"
   - python: 3.11
     extras: "[all]"
   ```

4. **Import complexity**: Lazy imports and guards add code:
   ```python
   # Every optional feature needs try/except
   try:
       import optional_dep
   except ImportError:
       raise MissingDependencyError(...)
   ```

### Mitigations

1. **Clear documentation**: README explains install options prominently
2. **Helpful errors**: `MissingDependencyError` includes install command
3. **`[all]` shortcut**: One command for everything
4. **CI coverage**: Test all combinations in CI

## Alternatives Considered

### 1. Single Package with All Dependencies

Include everything in base install.

```toml
dependencies = [
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "shap>=0.42.0",
    "xgboost>=2.0.0",
    "lightgbm>=4.0.0",
]
```

**Rejected because:**
- Large install size
- More dependency conflicts
- Users pay for features they don't use

### 2. Separate Packages

Split into multiple packages.

```
forge-core
forge-viz
forge-shap
forge-boosting
```

**Rejected because:**
- Complex dependency management
- Users must track multiple packages
- Version synchronization issues

### 3. Conditional Imports Only

No extras, just try/except at runtime.

```python
try:
    import matplotlib
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def plot():
    if not HAS_MATPLOTLIB:
        raise ImportError("Install matplotlib")
```

**Rejected because:**
- No pip integration
- Users don't know what to install upfront
- Less discoverable than extras

### 4. Plugin Architecture

Optional features as plugins.

```bash
pip install forge-features
pip install forge-viz-plugin
```

**Rejected because:**
- Over-engineering for current scope
- Complex plugin discovery
- Harder to maintain

## Configuration Examples

### pyproject.toml for Downstream Projects

```toml
# User's pyproject.toml
[project]
dependencies = [
    "forge-features[viz,shap]>=0.1.0",
]
```

### Docker Images

```dockerfile
# Minimal image
FROM python:3.11-slim
RUN pip install forge-features

# Full-featured image
FROM python:3.11-slim
RUN pip install forge-features[all]
```

### CI/CD

```yaml
# GitHub Actions
jobs:
  test:
    strategy:
      matrix:
        extras: ["", "[viz]", "[shap]", "[all]"]
    steps:
      - run: pip install forge-features${{ matrix.extras }}
      - run: pytest
```

## Related Decisions

- [ADR-0007: Lazy Import Pattern](adr-0007-lazy-import-pattern.md) - Lazy imports enable graceful degradation
- [ADR-0012: Custom Exception Hierarchy](adr-0012-custom-exception-hierarchy.md) - `MissingDependencyError` for clear messaging
