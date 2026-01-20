# Feature Selection Strategy Guide

Choosing the right feature selection method can significantly impact model performance. This guide helps you decide which selection method to use based on your data and requirements.

## Quick Decision Tree

```
Start Here
    │
    ├── Need fast, unsupervised filtering?
    │   └── YES → Use Variance + Correlation Selectors
    │
    ├── Have a classification target?
    │   ├── Categorical features? → Chi-Square Selector
    │   ├── Numeric features? → ANOVA Selector
    │   └── Mixed features? → Mutual Information Selector
    │
    ├── Have a regression target?
    │   └── Use Mutual Information or Importance Selector
    │
    ├── Need interpretability?
    │   └── YES → Use SHAP Selector (slower but explainable)
    │
    └── Need speed at scale?
        └── YES → Use Importance Selector with Random Forest
```

## Method Comparison

| Method | Speed | Target Required | Best For | Limitations |
|--------|-------|-----------------|----------|-------------|
| **VarianceSelector** | ⚡⚡⚡ | No | Removing constant/near-constant features | Doesn't consider target |
| **CorrelationSelector** | ⚡⚡⚡ | No | Removing redundant features | Only captures linear relationships |
| **ChiSquareSelector** | ⚡⚡ | Yes (classification) | Categorical features | Only for categorical features |
| **ANOVASelector** | ⚡⚡ | Yes (classification) | Numeric features | Assumes normal distribution |
| **MutualInfoSelector** | ⚡ | Yes | Non-linear relationships | Slower, requires hyperparameter tuning |
| **ImportanceSelector** | ⚡ | Yes | General purpose, mixed data | Model-dependent |
| **SHAPSelector** | 🐢 | Yes | Interpretability, accurate importance | Slowest, memory-intensive |

## Detailed Strategy by Scenario

### Scenario 1: Rapid Prototyping

**Goal**: Quick feature reduction for initial experiments

**Recommended Approach**:
```python
from forge.selectors import VarianceSelector, CorrelationSelector

# Two-stage filtering: remove low variance, then correlated features
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('variance', VarianceSelector(threshold=0.01)),
    ('correlation', CorrelationSelector(threshold=0.9)),
])

X_filtered = pipeline.fit_transform(X)
```

**Why**: Fast, no target required, removes obvious noise.

### Scenario 2: Classification with Mixed Features

**Goal**: Select best features for a classification model

**Recommended Approach**:
```python
from forge.selectors import MutualInfoSelector

# Mutual information works with any feature type
selector = MutualInfoSelector(
    k=50,  # Select top 50 features
    discrete_features='auto',  # Auto-detect categorical
    random_state=42,
)

X_selected = selector.fit_transform(X, y)
```

**Why**: Captures non-linear relationships, works with mixed types.

### Scenario 3: High-Dimensional Data (1000+ features)

**Goal**: Aggressive feature reduction for high-dimensional data

**Recommended Approach**:
```python
from forge.selectors import (
    VarianceSelector,
    CorrelationSelector,
    ImportanceSelector,
)
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    # Stage 1: Remove obvious noise (fast)
    ('variance', VarianceSelector(threshold=0.001)),

    # Stage 2: Remove redundancy (fast)
    ('correlation', CorrelationSelector(threshold=0.95)),

    # Stage 3: Model-based selection (slower but accurate)
    ('importance', ImportanceSelector(
        estimator='random_forest',
        k=100,
        n_jobs=-1,
    )),
])

X_selected = pipeline.fit_transform(X, y)
```

**Why**: Multi-stage approach handles scale efficiently.

### Scenario 4: Production Model with Interpretability

**Goal**: Select features that are explainable to stakeholders

**Recommended Approach**:
```python
from forge.selectors import SHAPSelector

selector = SHAPSelector(
    k=20,  # Keep only 20 most impactful features
    model_type='tree',  # or 'linear' for linear models
)

X_selected = selector.fit_transform(X, y)

# Get SHAP values for explanation
shap_values = selector.shap_values_

# Feature importance ranking
importance = selector.feature_importances_
```

**Why**: SHAP provides theoretically-grounded importance with explanations.

### Scenario 5: Time Series / Temporal Data

**Goal**: Select features for time series prediction

**Recommended Approach**:
```python
from forge.selectors import ImportanceSelector, CorrelationSelector

# First remove redundant lag features
correlation_filter = CorrelationSelector(
    threshold=0.95,
    method='spearman',  # Better for non-linear relationships
)

# Then select by importance
importance_selector = ImportanceSelector(
    estimator='gradient_boosting',
    k=50,
)

# Combine
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('correlation', correlation_filter),
    ('importance', importance_selector),
])

X_selected = pipeline.fit_transform(X, y)
```

**Why**: Lag features are often highly correlated; importance selection handles temporal patterns.

### Scenario 6: Imbalanced Classification

**Goal**: Select features that work well with class imbalance

**Recommended Approach**:
```python
from forge.selectors import ImportanceSelector
from sklearn.ensemble import RandomForestClassifier

# Use class-weighted model for selection
weighted_rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
)

selector = ImportanceSelector(
    estimator=weighted_rf,
    k=50,
)

X_selected = selector.fit_transform(X, y)
```

**Why**: Class weighting ensures minority class patterns are considered.

## Parameter Tuning Guide

### VarianceSelector
- `threshold=0.0`: Remove only constant features
- `threshold=0.01`: Remove near-constant features (recommended starting point)
- `threshold=0.1`: Aggressive removal

### CorrelationSelector
- `threshold=0.99`: Very conservative (only exact duplicates)
- `threshold=0.95`: Conservative (recommended for most cases)
- `threshold=0.9`: Moderate
- `threshold=0.8`: Aggressive removal

### Chi-Square / ANOVA / Mutual Information
- `k=10-50`: For interpretable models
- `k=100-200`: For complex models
- `percentile=10-30`: Alternative to fixed k

### ImportanceSelector
- `threshold=0.01`: Keep features with at least 1% importance
- `k=50-100`: Fixed number of features
- `estimator`: 'random_forest' (balanced) or 'gradient_boosting' (more accurate)

### SHAPSelector
- `k=10-30`: Typical for interpretable models
- `model_type`: 'tree' (fast) or 'kernel' (any model, slow)

## Common Mistakes to Avoid

### 1. Using Target Information in Unsupervised Selection
❌ **Wrong**: Selecting features based on correlation with target before train/test split

✅ **Correct**: Only use unsupervised methods (variance, correlation) before split, or use cross-validation

### 2. Over-filtering
❌ **Wrong**: Removing too many features without validation

✅ **Correct**: Use cross-validation to find optimal feature count
```python
from sklearn.model_selection import cross_val_score

for k in [10, 20, 50, 100]:
    selector = ImportanceSelector(k=k)
    X_selected = selector.fit_transform(X_train, y_train)
    scores = cross_val_score(model, X_selected, y_train, cv=5)
    print(f"k={k}: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### 3. Ignoring Feature Interactions
❌ **Wrong**: Removing correlated features without considering their combined predictive power

✅ **Correct**: Use model-based selection that can capture interactions
```python
# Gradient boosting captures interactions
selector = ImportanceSelector(estimator='gradient_boosting', k=50)
```

### 4. Not Considering Stability
❌ **Wrong**: Selecting different features on different data splits

✅ **Correct**: Check selection stability across folds
```python
from sklearn.model_selection import StratifiedKFold

selected_features = []
for train_idx, val_idx in StratifiedKFold(5).split(X, y):
    selector = ImportanceSelector(k=50)
    selector.fit(X.iloc[train_idx], y.iloc[train_idx])
    selected_features.append(set(selector.get_feature_names_out()))

# Features selected in all folds are most stable
stable_features = set.intersection(*selected_features)
```

## Recommended Defaults by Use Case

| Use Case | Recommended Method | Parameters |
|----------|-------------------|------------|
| Quick exploration | VarianceSelector | `threshold=0.01` |
| Binary classification | MutualInfoSelector | `k=50` |
| Multiclass classification | ImportanceSelector | `estimator='random_forest', k=50` |
| Regression | ImportanceSelector | `estimator='gradient_boosting', k=50` |
| High interpretability | SHAPSelector | `k=20` |
| Very high dimensions | Pipeline: Variance → Correlation → Importance | See Scenario 3 |
| Production pipeline | SHAP (if time allows) or Importance | Document selection for reproducibility |

## Performance Benchmarks

Selection time on 100,000 rows × 200 features:

| Method | Time | Memory |
|--------|------|--------|
| VarianceSelector | 0.02s | Low |
| CorrelationSelector | 0.08s | Medium |
| ChiSquareSelector | 0.05s | Low |
| ANOVASelector | 0.05s | Low |
| MutualInfoSelector | 0.95s | Medium |
| ImportanceSelector (RF) | 1.2s | Medium |
| ImportanceSelector (GB) | 2.5s | Medium |
| SHAPSelector | 8.5s | High |

Choose faster methods for iteration, slower methods for final model selection.
