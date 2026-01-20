# ADR-0008: Composition-Based Pipeline Architecture

## Status

Accepted

## Context

The `AutoFeatureTransformer` is Forge's main entry point, orchestrating:
- Data analysis and type inference
- Feature generation (multiple generators)
- Feature selection (multiple selectors)
- Missing value imputation

We needed to decide how to structure this orchestration:

1. **Inheritance-based**: Create class hierarchy with specialized transformers
2. **Composition-based**: Main class holds instances of other components
3. **Configuration-based**: Single class with extensive configuration options
4. **Pipeline-based**: Chain independent transformers in sklearn Pipeline

## Decision

Use **composition** where `AutoFeatureTransformer` contains instances of generators, selectors, and other components:

```python
class AutoFeatureTransformer(BaseEstimator, TransformerMixin, ForgeTransformerMixin):
    """Main entry point for automatic feature engineering."""

    def __init__(
        self,
        max_features: int | float | None = None,
        numeric_transformations: list[str] | None = None,
        categorical_encoding: str = "auto",
        selection_method: str = "importance",
        n_jobs: int = -1,
        random_state: int | None = None,
        verbose: int = 0,
    ):
        self.max_features = max_features
        self.numeric_transformations = numeric_transformations
        self.categorical_encoding = categorical_encoding
        self.selection_method = selection_method
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the transformer by composing internal components."""
        # 1. Analyze data
        self._analyzer = DataAnalyzer()
        self._analysis = self._analyzer.analyze(X, y)

        # 2. Create imputer
        self._imputer = AutoImputer(strategy="auto")
        self._imputer.fit(X, y)

        # 3. Create generators based on analysis
        self._generators: list[BaseFeatureGenerator] = []
        self._create_generators_for_types(self._analysis.column_types)

        # 4. Fit generators
        X_imputed = self._imputer.transform(X)
        for generator in self._generators:
            generator.fit(X_imputed, y)

        # 5. Create and fit selectors
        self._selectors: list[BaseFeatureSelector] = []
        self._create_selectors()

        X_generated = self._transform_generators(X_imputed)
        for selector in self._selectors:
            selector.fit(X_generated, y)
            X_generated = selector.transform(X_generated)

        self._is_fitted = True
        return self

    def _create_generators_for_types(self, column_types: dict[str, ColumnType]) -> None:
        """Create appropriate generators based on detected column types."""
        numeric_cols = [c for c, t in column_types.items() if t == ColumnType.NUMERIC]
        categorical_cols = [c for c, t in column_types.items() if t == ColumnType.CATEGORICAL]
        temporal_cols = [c for c, t in column_types.items() if t == ColumnType.DATETIME]

        # Add numeric generators
        if numeric_cols and self.numeric_transformations:
            if "interactions" in self.numeric_transformations:
                self._generators.append(InteractionGenerator(columns=numeric_cols))
            if "polynomials" in self.numeric_transformations:
                self._generators.append(PolynomialGenerator(columns=numeric_cols, degree=2))

        # Add categorical encoders
        if categorical_cols:
            encoder = self._create_encoder(categorical_cols)
            self._generators.append(encoder)

        # Add temporal generators
        if temporal_cols:
            self._generators.append(DatePartExtractor(columns=temporal_cols))
```

### Key Design Principles

1. **Components are replaceable**: Each component can be swapped:
   ```python
   transformer = AutoFeatureTransformer()
   transformer.fit(X, y)

   # Replace selector after fitting (for experimentation)
   transformer._selectors = [CustomSelector()]
   ```

2. **Components are inspectable**: Access internal state:
   ```python
   # See what generators were created
   for gen in transformer._generators:
       print(type(gen).__name__, gen.get_feature_names_out())
   ```

3. **Flexible configuration**: Parameters control component creation:
   ```python
   # Different configurations create different component sets
   AutoFeatureTransformer(
       numeric_transformations=["interactions"],  # Only interactions
       selection_method="shap",                   # SHAP selector
   )

   AutoFeatureTransformer(
       numeric_transformations=["polynomials", "log"],  # Polynomials + log
       selection_method="correlation",                   # Correlation selector
   )
   ```

## Consequences

### Positive

1. **Flexibility**: Users can customize component behavior:
   ```python
   transformer = AutoFeatureTransformer(
       numeric_transformations=["interactions", "polynomials"],
       categorical_encoding="target",
       selection_method="importance",
       max_features=100,
   )
   ```

2. **Testability**: Components can be tested independently:
   ```python
   # Test generator in isolation
   generator = InteractionGenerator(columns=["a", "b"])
   generator.fit(X)
   assert "a_b_interaction" in generator.get_feature_names_out()

   # Test selector in isolation
   selector = ImportanceSelector(max_features=10)
   selector.fit(X, y)
   assert sum(selector.get_support()) == 10
   ```

3. **Extensibility**: Easy to add new components:
   ```python
   # Custom generator integrates seamlessly
   class MyGenerator(BaseFeatureGenerator):
       ...

   transformer._generators.append(MyGenerator())
   ```

4. **Debuggability**: Inspect pipeline state at any point:
   ```python
   # See analysis results
   print(transformer._analysis.column_types)

   # Check which features each generator creates
   for gen in transformer._generators:
       print(f"{type(gen).__name__}: {len(gen.get_feature_names_out())} features")
   ```

5. **Incremental development**: Add features without changing architecture:
   ```python
   # Adding new generator type is localized change
   if "text" in column_types:
       self._generators.append(TfidfGenerator(columns=text_cols))
   ```

### Negative

1. **Internal state complexity**: Many internal attributes:
   ```python
   self._analyzer
   self._analysis
   self._imputer
   self._generators  # list
   self._selectors   # list
   ```

2. **Order dependence**: Components must be created and applied in correct order:
   ```python
   # Must analyze before creating generators
   # Must generate before selecting
   # Incorrect order causes errors
   ```

3. **Configuration explosion**: Many parameters control behavior:
   ```python
   AutoFeatureTransformer(
       max_features=...,
       numeric_transformations=...,
       categorical_encoding=...,
       temporal_features=...,
       missing_strategy=...,
       selection_method=...,
       correlation_threshold=...,
       variance_threshold=...,
       n_jobs=...,
       random_state=...,
       verbose=...,
   )
   ```

### Mitigations

1. **Sensible defaults**: Most parameters have good defaults
2. **Preset configurations**: Common configurations as class methods:
   ```python
   AutoFeatureTransformer.quick()       # Fast, minimal features
   AutoFeatureTransformer.thorough()    # Complete feature search
   AutoFeatureTransformer.for_tree()    # Optimized for tree models
   ```
3. **Validation**: Check configuration consistency in `fit()`

## Alternatives Considered

### 1. Inheritance-Based Hierarchy

Create specialized transformers via inheritance.

```python
class BaseAutoTransformer:
    ...

class NumericAutoTransformer(BaseAutoTransformer):
    """Only numeric features."""

class CategoricalAutoTransformer(BaseAutoTransformer):
    """Only categorical features."""

class FullAutoTransformer(NumericAutoTransformer, CategoricalAutoTransformer):
    """All features."""
```

**Rejected because:**
- Diamond inheritance problems
- Inflexible combinations
- Hard to customize individual components

### 2. Pure sklearn Pipeline

Use sklearn Pipeline to chain components.

```python
pipeline = Pipeline([
    ("analyzer", DataAnalyzerTransformer()),
    ("imputer", AutoImputer()),
    ("numeric", NumericFeatureGenerator()),
    ("categorical", CategoricalEncoder()),
    ("selector", FeatureSelector()),
])
```

**Rejected because:**
- No conditional component creation based on analysis
- Generators need to know column types before construction
- Less integrated user experience

### 3. Configuration Object

Single class with configuration object.

```python
config = FeatureConfig(
    numeric=NumericConfig(interactions=True, polynomials=True),
    categorical=CategoricalConfig(encoding="target"),
    selection=SelectionConfig(method="importance", max_features=100),
)
transformer = AutoFeatureTransformer(config)
```

**Rejected because:**
- More complex API
- Configuration objects need their own validation
- Over-engineering for current needs

## Related Decisions

- [ADR-0001: sklearn Compatibility](adr-0001-sklearn-compatibility.md) - Components follow sklearn interface
- [ADR-0002: Plugin Registry Pattern](adr-0002-plugin-registry-pattern.md) - Registry used to discover available generators
- [ADR-0010: Explicit Analysis Phase](adr-0010-explicit-analysis-phase.md) - Analysis drives generator creation
