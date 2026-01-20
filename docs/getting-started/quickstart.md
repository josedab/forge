# Quick Start

This guide will get you up and running with Forge in just a few minutes.

## Basic Usage

The simplest way to use Forge is with `AutoFeatureTransformer`:

```python
import pandas as pd
from forge import AutoFeatureTransformer

# Load your data
X = pd.DataFrame({
    'age': [25, 30, 35, 40],
    'income': [50000, 60000, 75000, 90000],
    'category': ['A', 'B', 'A', 'C'],
    'signup_date': pd.to_datetime(['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05'])
})
y = pd.Series([0, 1, 0, 1])

# Create and fit transformer
transformer = AutoFeatureTransformer(max_features=50)
X_engineered = transformer.fit_transform(X, y)

# View results
print(f"Original features: {X.shape[1]}")
print(f"Engineered features: {X_engineered.shape[1]}")
print(transformer.get_feature_names_out()[:10])
```

## sklearn Pipeline Integration

Forge integrates seamlessly with scikit-learn pipelines:

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from forge import AutoFeatureTransformer

# Create pipeline
pipeline = Pipeline([
    ('features', AutoFeatureTransformer(max_features=50)),
    ('classifier', RandomForestClassifier(n_estimators=100)),
])

# Train and evaluate
pipeline.fit(X_train, y_train)
scores = cross_val_score(pipeline, X, y, cv=5)
print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

## Data Analysis

Analyze your data before feature engineering:

```python
from forge.analyzer import DataAnalyzer

analyzer = DataAnalyzer()
report = analyzer.analyze(X, y)

print(report.summary())
print(report.column_types)
print(report.quality_issues)
```

## Custom Feature Generation

For more control, use individual generators:

```python
from forge.generators.numeric import InteractionGenerator, PolynomialGenerator
from forge.generators.categorical import TargetEncoder
from forge.transformers import ForgePipeline

pipeline = ForgePipeline([
    ('interactions', InteractionGenerator(columns=['price', 'quantity'])),
    ('polynomials', PolynomialGenerator(degree=2)),
    ('encoding', TargetEncoder(columns=['category', 'region'])),
])

X_features = pipeline.fit_transform(X, y)
```

## Next Steps

- Learn more about [feature generation](../guide/generation.md)
- Explore [feature selection](../guide/selection.md) options
- See [full examples](../examples/basic.md)
