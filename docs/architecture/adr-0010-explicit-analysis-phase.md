# ADR-0010: Explicit Analysis Phase

## Status

Accepted

## Context

Forge needs to understand input data before generating features:
- What are the column types? (numeric, categorical, datetime, text)
- What is the data quality? (missing values, outliers, constants)
- What are the statistical properties? (distributions, cardinality)

This information drives:
- Which generators to apply to which columns
- What encoding strategy to use for categoricals
- How to handle missing values
- What transformations make sense

We needed to decide when and how to perform this analysis:

1. **Implicit analysis**: Each component analyzes what it needs
2. **Explicit analysis phase**: Dedicated analysis step before generation
3. **User-provided metadata**: Require users to specify column types
4. **Lazy analysis**: Analyze columns on first access

## Decision

Implement an **explicit analysis phase** as the first step in the pipeline:

```python
class AutoFeatureTransformer:
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit transformer with explicit analysis phase."""

        # ============================================
        # PHASE 1: ANALYSIS
        # ============================================
        self._analyzer = DataAnalyzer(
            categorical_threshold=self.categorical_threshold,
            missing_threshold=self.missing_threshold,
        )
        self._analysis = self._analyzer.analyze(X, y)

        # Analysis results available for inspection
        # self._analysis.column_types: dict[str, ColumnType]
        # self._analysis.statistics: dict[str, ColumnStatistics]
        # self._analysis.quality: DataQualityReport

        # ============================================
        # PHASE 2: COMPONENT CREATION (uses analysis)
        # ============================================
        self._create_imputer(self._analysis)
        self._create_generators(self._analysis)
        self._create_selectors(self._analysis)

        # ============================================
        # PHASE 3: FITTING
        # ============================================
        X_processed = self._fit_transform_pipeline(X, y)

        return self
```

### DataAnalyzer Implementation

```python
class DataAnalyzer:
    """Analyze DataFrame to extract metadata for feature engineering."""

    def __init__(
        self,
        categorical_threshold: float = 0.05,
        missing_threshold: float = 0.3,
        constant_threshold: float = 0.99,
    ):
        self.categorical_threshold = categorical_threshold
        self.missing_threshold = missing_threshold
        self.constant_threshold = constant_threshold

    def analyze(self, X: pd.DataFrame, y: pd.Series | None = None) -> AnalysisResult:
        """Perform comprehensive data analysis."""
        return AnalysisResult(
            column_types=self._infer_types(X),
            statistics=self._compute_statistics(X),
            quality=self._assess_quality(X),
            target_analysis=self._analyze_target(y) if y is not None else None,
        )

    def _infer_types(self, X: pd.DataFrame) -> dict[str, ColumnType]:
        """Infer semantic column types."""
        return {col: self._infer_column_type(X[col]) for col in X.columns}

    def _compute_statistics(self, X: pd.DataFrame) -> dict[str, ColumnStatistics]:
        """Compute statistics for each column."""
        return {col: self._column_statistics(X[col]) for col in X.columns}

    def _assess_quality(self, X: pd.DataFrame) -> DataQualityReport:
        """Assess data quality issues."""
        return DataQualityReport(
            missing_columns=self._find_high_missing(X),
            constant_columns=self._find_constants(X),
            duplicate_columns=self._find_duplicates(X),
        )


@dataclass
class AnalysisResult:
    """Results from data analysis phase."""

    column_types: dict[str, ColumnType]
    statistics: dict[str, ColumnStatistics]
    quality: DataQualityReport
    target_analysis: TargetAnalysis | None
```

### Analysis-Driven Component Creation

```python
def _create_generators(self, analysis: AnalysisResult) -> None:
    """Create generators based on analysis results."""
    self._generators = []

    # Group columns by type
    numeric_cols = [
        col for col, typ in analysis.column_types.items()
        if typ == ColumnType.NUMERIC
    ]
    categorical_cols = [
        col for col, typ in analysis.column_types.items()
        if typ == ColumnType.CATEGORICAL
    ]
    datetime_cols = [
        col for col, typ in analysis.column_types.items()
        if typ == ColumnType.DATETIME
    ]

    # Create type-appropriate generators
    if numeric_cols:
        self._generators.extend(self._create_numeric_generators(numeric_cols))

    if categorical_cols:
        # Use cardinality from analysis to choose encoder
        encoder = self._choose_encoder(categorical_cols, analysis.statistics)
        self._generators.append(encoder)

    if datetime_cols:
        self._generators.append(DatePartExtractor(columns=datetime_cols))
```

## Consequences

### Positive

1. **Informed decisions**: Generators created with full knowledge of data:
   ```python
   # Analysis reveals column cardinality
   # High cardinality → target encoding
   # Low cardinality → one-hot encoding
   if analysis.statistics[col].n_unique > 20:
       encoder = TargetEncoder(columns=[col])
   else:
       encoder = OneHotEncoder(columns=[col])
   ```

2. **Single pass analysis**: Data analyzed once, results reused:
   ```python
   # All components share analysis results
   # No redundant type inference or statistics computation
   ```

3. **Transparency**: Users can inspect analysis results:
   ```python
   transformer.fit(X, y)

   # See what types were inferred
   print(transformer._analysis.column_types)
   # {'age': NUMERIC, 'category': CATEGORICAL, 'date': DATETIME}

   # See quality issues
   print(transformer._analysis.quality.missing_columns)
   # ['income', 'address']  # >30% missing
   ```

4. **Override capability**: Users can provide their own analysis:
   ```python
   # Override inferred types
   transformer = AutoFeatureTransformer(
       column_types={'user_id': 'categorical'}  # Override numeric inference
   )
   ```

5. **Standalone analysis**: DataAnalyzer usable independently:
   ```python
   from forge.analyzer import DataAnalyzer

   analyzer = DataAnalyzer()
   report = analyzer.analyze(X, y)

   # Use for data exploration, not just feature engineering
   print(report.summary())
   ```

### Negative

1. **Upfront cost**: Analysis runs even for simple transformations:
   ```python
   # Even if user only wants polynomial features
   # Full analysis still runs
   transformer = AutoFeatureTransformer(
       numeric_transformations=["polynomials"]
   )
   transformer.fit(X, y)  # Analyzes all columns
   ```

2. **Memory overhead**: Analysis results stored in memory:
   ```python
   # Statistics for all columns retained
   # Could be significant for wide datasets
   self._analysis.statistics  # dict with stats per column
   ```

3. **Delayed feedback**: Type inference errors surface at fit time:
   ```python
   # User might expect 'category' to be categorical
   # But high cardinality causes numeric inference
   # Only discovered after fit()
   ```

### Mitigations

1. **Lazy statistics**: Only compute statistics for columns that will be used
2. **Analysis caching**: Cache results for repeated fits on same data
3. **Early validation**: Warn about suspicious inferences immediately

## Alternatives Considered

### 1. Implicit Analysis

Each component analyzes what it needs.

```python
class InteractionGenerator:
    def fit(self, X, y=None):
        # Analyze to find numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        self._columns = numeric_cols
```

**Rejected because:**
- Redundant analysis across components
- Inconsistent type inference
- No centralized quality checks

### 2. User-Provided Metadata

Require users to specify column types.

```python
transformer = AutoFeatureTransformer(
    numeric_columns=['age', 'income'],
    categorical_columns=['category', 'region'],
    datetime_columns=['date'],
)
```

**Rejected because:**
- Poor user experience
- Tedious for many columns
- Violates "automatic" in "automatic feature engineering"

### 3. Lazy Analysis

Analyze columns on first access.

```python
class LazyAnalyzer:
    def get_type(self, column_name):
        if column_name not in self._cache:
            self._cache[column_name] = self._infer_type(column_name)
        return self._cache[column_name]
```

**Rejected because:**
- Unpredictable performance
- Hard to parallelize
- Analysis timing non-deterministic

### 4. Schema-Based

Use external schema definition.

```python
schema = DataSchema.from_json("schema.json")
transformer = AutoFeatureTransformer(schema=schema)
```

**Rejected because:**
- Requires schema maintenance
- Extra file to manage
- Schema can become stale

## Related Decisions

- [ADR-0003: Type Inference Approach](adr-0003-type-inference-approach.md) - How types are inferred
- [ADR-0008: Composition-Based Pipeline](adr-0008-composition-based-pipeline.md) - Analysis drives component creation
- [ADR-0006: Hierarchical Generator Organization](adr-0006-hierarchical-generator-organization.md) - Generators organized by type for easy selection
