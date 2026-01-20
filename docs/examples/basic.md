# Basic Usage Examples

## Complete Feature Engineering Workflow

```python
import pandas as pd
from forge import AutoFeatureTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv('your_data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Feature engineering
transformer = AutoFeatureTransformer(
    max_features=100,
    selection_method='importance',
    verbose=1
)
X_train_eng = transformer.fit_transform(X_train, y_train)
X_test_eng = transformer.transform(X_test)

print(f"Original features: {X_train.shape[1]}")
print(f"Engineered features: {X_train_eng.shape[1]}")

# Train model
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train_eng, y_train)

# Evaluate
predictions = clf.predict(X_test_eng)
print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
```

## Data Analysis Before Engineering

```python
from forge.analyzer import DataAnalyzer

analyzer = DataAnalyzer()
report = analyzer.analyze(X, y)

# View summary
print(report.summary())

# Check column types
print("\nColumn Types:")
for col, dtype in report.column_types.items():
    print(f"  {col}: {dtype}")

# Check for quality issues
if report.quality_issues:
    print("\nQuality Issues:")
    for issue in report.quality_issues:
        print(f"  - {issue}")
```

## Custom Generator Pipeline

```python
from forge.generators.numeric import InteractionGenerator, TransformationGenerator
from forge.generators.categorical import TargetEncoder
from forge.generators.temporal import DatetimeComponentGenerator
from forge.transformers import ForgePipeline

# Define custom pipeline
pipeline = ForgePipeline([
    # Numeric transformations
    ('log_transform', TransformationGenerator(
        columns=['income', 'amount'],
        transformations=['log']
    )),

    # Interactions
    ('interactions', InteractionGenerator(
        columns=['price', 'quantity'],
        operations=['multiply']
    )),

    # Categorical encoding
    ('target_encode', TargetEncoder(
        columns=['category', 'region']
    )),

    # Temporal features
    ('datetime', DatetimeComponentGenerator(
        columns=['date'],
        components=['month', 'dayofweek', 'hour']
    )),
])

# Fit and transform
X_features = pipeline.fit_transform(X, y)
print(f"Generated {X_features.shape[1]} features")
```

## Feature Selection Only

```python
from forge.selectors import ImportanceSelector, CorrelationSelector

# Remove correlated features first
corr_selector = CorrelationSelector(threshold=0.9)
X_decorr = corr_selector.fit_transform(X)

# Then select by importance
imp_selector = ImportanceSelector(k=50)
X_selected = imp_selector.fit_transform(X_decorr, y)

# Get selected feature names
selected_names = imp_selector.get_feature_names_out()
print(f"Selected features: {selected_names}")
```

## Visualization

```python
from forge.visualization import plot_importance, plot_correlation

# Feature importance
importance = transformer.get_feature_importance()
plot_importance(importance, top_n=20)

# Correlation matrix
plot_correlation(X_engineered, figsize=(12, 10))
```
