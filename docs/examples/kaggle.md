# Kaggle Workflow Example

A complete example for a Kaggle competition workflow.

## Setup

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from forge import AutoFeatureTransformer
from forge.analyzer import DataAnalyzer

# Load competition data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X = train.drop(['id', 'target'], axis=1)
y = train['target']
X_test = test.drop('id', axis=1)
```

## Data Analysis

```python
# Analyze training data
analyzer = DataAnalyzer()
report = analyzer.analyze(X, y)

print("=== Data Summary ===")
print(report.summary())

print("\n=== Column Types ===")
for col, dtype in report.column_types.items():
    print(f"{col}: {dtype}")

print("\n=== Missing Values ===")
for col, pct in report.missing_percentages.items():
    if pct > 0:
        print(f"{col}: {pct:.1%}")

print("\n=== Quality Issues ===")
for issue in report.quality_issues:
    print(f"- {issue}")
```

## Feature Engineering Pipeline

```python
# Create feature engineering pipeline
pipeline = Pipeline([
    ('features', AutoFeatureTransformer(
        max_features=200,
        numeric_transformations=['log', 'sqrt'],
        categorical_encoding='target',
        temporal_features=['month', 'dayofweek', 'hour'],
        missing_strategy='auto',
        selection_method='importance',
        n_jobs=-1,
        verbose=1,
    )),
    ('classifier', GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
    )),
])
```

## Cross-Validation

```python
# Evaluate with cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc')

print(f"\nCV ROC-AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")
print(f"Fold scores: {scores}")
```

## Training and Submission

```python
# Train on full data
pipeline.fit(X, y)

# Get feature importance
transformer = pipeline.named_steps['features']
importance = transformer.get_feature_importance()
print("\nTop 20 Features:")
print(importance.head(20))

# Generate predictions
predictions = pipeline.predict_proba(X_test)[:, 1]

# Create submission
submission = pd.DataFrame({
    'id': test['id'],
    'target': predictions
})
submission.to_csv('submission.csv', index=False)
print(f"\nSubmission saved with {len(submission)} predictions")
```

## Feature Analysis

```python
# Analyze generated features
feature_names = transformer.get_feature_names_out()
print(f"\nTotal features generated: {len(feature_names)}")

# Group by type
from collections import Counter
feature_types = Counter()
for name in feature_names:
    if '_log' in name or '_sqrt' in name:
        feature_types['transformed'] += 1
    elif '_x_' in name or '_div_' in name:
        feature_types['interaction'] += 1
    elif '_encoded' in name:
        feature_types['categorical'] += 1
    elif any(t in name for t in ['_month', '_day', '_hour']):
        feature_types['temporal'] += 1
    else:
        feature_types['original'] += 1

print("\nFeature breakdown:")
for ftype, count in feature_types.most_common():
    print(f"  {ftype}: {count}")
```

## Visualization

```python
from forge.visualization import plot_importance, plot_correlation

# Plot top feature importance
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Importance plot
plot_importance(importance, top_n=20, ax=axes[0])
axes[0].set_title('Top 20 Feature Importance')

# Get top features for correlation
top_features = importance.head(20)['feature'].tolist()
X_eng = transformer.transform(X)
plot_correlation(X_eng[top_features], ax=axes[1])
axes[1].set_title('Top Features Correlation')

plt.tight_layout()
plt.savefig('feature_analysis.png', dpi=150)
plt.show()
```
