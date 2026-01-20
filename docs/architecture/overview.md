# Architecture Overview

Forge is built on a modular architecture designed for extensibility, performance, and seamless integration with the scikit-learn ecosystem.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        AutoFeatureTransformer                            │
│  (Main entry point - orchestrates the entire feature engineering flow)  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
            ┌───────────┐   ┌───────────┐   ┌───────────┐
            │  Analyzer │   │ Generators│   │ Selectors │
            │           │   │           │   │           │
            │ - Types   │   │ - Numeric │   │ - Stats   │
            │ - Stats   │   │ - Categ.  │   │ - Import. │
            │ - Quality │   │ - Temporal│   │ - Correl. │
            │           │   │ - Text    │   │ - SHAP    │
            └───────────┘   └───────────┘   └───────────┘
                                    │
                                    ▼
                         ┌───────────────────┐
                         │  sklearn Pipeline │
                         │   Compatibility   │
                         └───────────────────┘
```

## Core Components

### 1. DataAnalyzer

The `DataAnalyzer` component automatically analyzes input data to:

- **Infer column types**: Detect numeric, categorical, datetime, and text columns
- **Compute statistics**: Profile each column with relevant statistics
- **Assess quality**: Identify missing values, outliers, and data issues

Located in: `src/forge/analyzer/`

### 2. Feature Generators

Feature generators create new features from existing columns. All generators inherit from `BaseFeatureGenerator` and follow sklearn conventions.

**Generator Types:**

| Type | Module | Features Generated |
|------|--------|-------------------|
| Numeric | `generators/numeric/` | Interactions, polynomials, transformations, binning |
| Categorical | `generators/categorical/` | Encodings, combinations, target statistics |
| Temporal | `generators/temporal/` | Date parts, lags, rolling windows |
| Text | `generators/text/` | Lengths, TF-IDF, word counts |

**Registry Pattern:** Generators register themselves in a central registry for discovery and auto-selection.

### 3. Feature Selectors

Selectors reduce the feature space to the most relevant features. All selectors inherit from `BaseFeatureSelector`.

**Selection Methods:**

| Method | Class | Description |
|--------|-------|-------------|
| Statistical | `StatisticalSelector` | Chi-square, ANOVA, mutual information |
| Importance | `ImportanceSelector` | Tree-based feature importance |
| Correlation | `CorrelationSelector` | Remove correlated features |
| Variance | `VarianceSelector` | Remove low-variance features |
| SHAP | `ShapSelector` | SHAP value-based selection |

### 4. Transformers

Transformers provide sklearn-compatible interfaces:

- **AutoFeatureTransformer**: Main entry point, orchestrates generation and selection
- **ForgePipeline**: Custom pipeline for chaining transformers
- **ForgeTransformerMixin**: Base mixin for all Forge transformers

## Data Flow

```
Input DataFrame (X, y)
         │
         ▼
    ┌─────────┐
    │ Analyze │  ← Infer types, compute stats
    └────┬────┘
         │
         ▼
   ┌──────────┐
   │ Generate │  ← Create new features based on column types
   └────┬─────┘
         │
         ▼
    ┌─────────┐
    │ Select  │  ← Keep top N most relevant features
    └────┬────┘
         │
         ▼
Output DataFrame (X_transformed)
```

## Design Principles

### 1. sklearn Compatibility

All transformers follow sklearn conventions:
- Inherit from `BaseEstimator` and `TransformerMixin`
- Implement `fit()`, `transform()`, `fit_transform()`
- Support `get_feature_names_out()` for feature tracking
- Work with sklearn `Pipeline`, `GridSearchCV`, `cross_val_score`

See [ADR-0001](adr-0001-sklearn-compatibility.md) for details.

### 2. Plugin Architecture

Generators and selectors use a registry pattern for extensibility:
- Register new components without modifying core code
- Auto-discovery of available generators
- Easy to add custom transformers

See [ADR-0002](adr-0002-plugin-registry-pattern.md) for details.

### 3. Type-Aware Processing

Forge automatically infers column types and applies appropriate transformations:
- Different generators for different data types
- Automatic encoding selection based on cardinality
- Smart handling of mixed-type columns

See [ADR-0003](adr-0003-type-inference-approach.md) for details.

## Extension Points

### Adding a Custom Generator

```python
from forge.generators.base import BaseFeatureGenerator
from forge.generators.registry import register_generator

@register_generator("custom", column_types=["numeric"])
class MyGenerator(BaseFeatureGenerator):
    def fit(self, X, y=None):
        # Learn from data
        return self

    def transform(self, X):
        # Generate features
        return X_transformed
```

### Adding a Custom Selector

```python
from forge.selectors.base import BaseFeatureSelector

class MySelector(BaseFeatureSelector):
    def fit(self, X, y=None):
        # Compute feature scores
        self._support_mask = compute_mask(X, y)
        return self

    def get_support(self, indices=False):
        if indices:
            return np.where(self._support_mask)[0]
        return self._support_mask
```

## Performance Considerations

1. **Parallel Processing**: Generators support parallel execution via `n_jobs`
2. **Lazy Evaluation**: Features are generated on-demand where possible
3. **Memory Efficiency**: Large datasets processed in chunks
4. **Caching**: Intermediate results cached during pipeline execution

## Related Documentation

### Architecture Decision Records (ADRs)

The following ADRs document the key architectural decisions made in Forge:

**Core Design Decisions:**

- [ADR-0001: sklearn Compatibility](adr-0001-sklearn-compatibility.md) - Why all transformers follow sklearn conventions
- [ADR-0004: DataFrame-First Design](adr-0004-dataframe-first-design.md) - Why all inputs/outputs are pandas DataFrames
- [ADR-0005: Feature Name Tracking](adr-0005-feature-name-tracking.md) - How feature names are tracked through pipelines

**Organization & Structure:**

- [ADR-0002: Plugin Registry Pattern](adr-0002-plugin-registry-pattern.md) - How generators register and are discovered
- [ADR-0006: Hierarchical Generator Organization](adr-0006-hierarchical-generator-organization.md) - Why generators are organized by data type
- [ADR-0007: Lazy Import Pattern](adr-0007-lazy-import-pattern.md) - How lazy imports avoid circular dependencies

**Pipeline Architecture:**

- [ADR-0008: Composition-Based Pipeline](adr-0008-composition-based-pipeline.md) - Why AutoFeatureTransformer uses composition
- [ADR-0009: Sequential Selector Pipeline](adr-0009-sequential-selector-pipeline.md) - How selectors are applied sequentially
- [ADR-0010: Explicit Analysis Phase](adr-0010-explicit-analysis-phase.md) - Why analysis runs as a dedicated first phase

**Type System & Inference:**

- [ADR-0003: Type Inference Approach](adr-0003-type-inference-approach.md) - How column types are automatically inferred

**Dependencies & Error Handling:**

- [ADR-0011: Optional Dependencies Strategy](adr-0011-optional-dependencies-strategy.md) - How optional features are packaged
- [ADR-0012: Custom Exception Hierarchy](adr-0012-custom-exception-hierarchy.md) - Domain-specific exception types
