# Pipeline Integration

Forge is designed to work seamlessly with scikit-learn pipelines.

## Using with sklearn Pipeline

The `AutoFeatureTransformer` works as a drop-in transformer:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from forge import AutoFeatureTransformer

pipeline = Pipeline([
    ('features', AutoFeatureTransformer(max_features=100)),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier()),
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

## ForgePipeline

For chaining Forge transformers:

```python
from forge.transformers import ForgePipeline
from forge.generators.numeric import InteractionGenerator
from forge.generators.categorical import TargetEncoder
from forge.selectors import ImportanceSelector

pipeline = ForgePipeline([
    ('interactions', InteractionGenerator(columns=['a', 'b'])),
    ('encoding', TargetEncoder()),
    ('selection', ImportanceSelector(k=50)),
])

X_processed = pipeline.fit_transform(X, y)
```

## Grid Search with Forge

Tune Forge parameters with GridSearchCV:

```python
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ('features', AutoFeatureTransformer()),
    ('classifier', RandomForestClassifier()),
])

param_grid = {
    'features__max_features': [50, 100, 200],
    'features__selection_method': ['importance', 'mutual_info'],
    'classifier__n_estimators': [100, 200],
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X, y)

print(f"Best params: {grid_search.best_params_}")
```

## Saving and Loading Pipelines

Forge pipelines can be serialized:

```python
from forge.transformers import save_pipeline, load_pipeline

# Save
save_pipeline(pipeline, 'my_pipeline.pkl')

# Load
loaded_pipeline = load_pipeline('my_pipeline.pkl')
predictions = loaded_pipeline.transform(X_new)
```

Or use joblib directly:

```python
import joblib

# Save
joblib.dump(pipeline, 'my_pipeline.joblib')

# Load
pipeline = joblib.load('my_pipeline.joblib')
```

## Feature Names

Get feature names after transformation:

```python
# After fitting
feature_names = pipeline.named_steps['features'].get_feature_names_out()

# With ForgePipeline
feature_names = pipeline.get_feature_names_out()
```
