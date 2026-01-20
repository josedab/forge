# AutoFeatureTransformer

The main entry point for automatic feature engineering.

::: forge.transformers.auto_transformer.AutoFeatureTransformer

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_features` | `int \| float \| None` | `None` | Maximum features to keep. If float, interpreted as fraction. |
| `numeric_transformations` | `list[str] \| None` | `None` | Numeric transformations to apply: `["log", "sqrt", "bin"]` |
| `categorical_encoding` | `str` | `"auto"` | Encoding strategy: `"auto"`, `"target"`, `"frequency"`, `"onehot"` |
| `temporal_features` | `list[str] \| None` | `None` | Temporal components to extract |
| `missing_strategy` | `str` | `"auto"` | Imputation strategy: `"auto"`, `"mean"`, `"median"`, `"mode"` |
| `selection_method` | `str` | `"importance"` | Selection method: `"importance"`, `"mutual_info"`, `"shap"` |
| `n_jobs` | `int` | `-1` | Number of parallel jobs |
| `random_state` | `int \| None` | `None` | Random seed for reproducibility |
| `verbose` | `int` | `0` | Verbosity level |

## Methods

### fit

```python
def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
    """Fit the transformer to the data."""
```

### transform

```python
def transform(self, X: pd.DataFrame) -> pd.DataFrame:
    """Transform the data."""
```

### fit_transform

```python
def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
    """Fit and transform in one step."""
```

### get_feature_names_out

```python
def get_feature_names_out(self) -> list[str]:
    """Get output feature names."""
```

### get_feature_importance

```python
def get_feature_importance(self) -> pd.DataFrame:
    """Get feature importance scores."""
```

## Example

```python
from forge import AutoFeatureTransformer

transformer = AutoFeatureTransformer(
    max_features=100,
    numeric_transformations=["log", "sqrt"],
    categorical_encoding="target",
    selection_method="importance",
    n_jobs=-1,
    verbose=1,
)

X_engineered = transformer.fit_transform(X_train, y_train)
X_test_engineered = transformer.transform(X_test)

# Get importance
importance = transformer.get_feature_importance()
print(importance.head(10))
```
