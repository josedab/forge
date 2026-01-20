# ADR-0009: Sequential Selector Pipeline

## Status

Accepted

## Context

After generating features, Forge needs to select the most relevant ones. Multiple selection strategies exist:

- **Variance-based**: Remove low-variance features
- **Correlation-based**: Remove highly correlated features
- **Statistical**: Chi-square, ANOVA F-test, mutual information
- **Importance-based**: Tree-based feature importance
- **SHAP-based**: SHAP value importance

We needed to decide how to combine these selectors:

1. **Sequential pipeline**: Apply selectors one after another
2. **Voting/ensemble**: Each selector votes, keep features with most votes
3. **Union**: Keep features selected by any selector
4. **Intersection**: Keep only features selected by all selectors
5. **Weighted combination**: Weighted score across selectors

## Decision

Apply selectors **sequentially** in a fixed order:

```
Generated Features (N features)
         │
         ▼
┌─────────────────────────┐
│ 1. VarianceSelector     │  ← Remove near-constant features
│    (threshold=0.01)     │
└───────────┬─────────────┘
            │ (N - removed)
            ▼
┌─────────────────────────┐
│ 2. CorrelationSelector  │  ← Remove highly correlated features
│    (threshold=0.95)     │
└───────────┬─────────────┘
            │ (N - correlated)
            ▼
┌─────────────────────────┐
│ 3. ImportanceSelector   │  ← Keep top-K by importance
│    (max_features=100)   │
└───────────┬─────────────┘
            │ (max 100)
            ▼
    Selected Features
```

### Implementation

```python
class AutoFeatureTransformer:
    def _create_selectors(self) -> None:
        """Create selector pipeline in fixed order."""
        self._selectors = []

        # 1. Always remove low-variance features first
        self._selectors.append(
            VarianceSelector(threshold=self.variance_threshold)
        )

        # 2. Remove highly correlated features
        if self.remove_correlated:
            self._selectors.append(
                CorrelationSelector(threshold=self.correlation_threshold)
            )

        # 3. Final selection by importance/statistical method
        if self.max_features is not None:
            if self.selection_method == "importance":
                self._selectors.append(
                    ImportanceSelector(
                        max_features=self.max_features,
                        random_state=self.random_state,
                    )
                )
            elif self.selection_method == "statistical":
                self._selectors.append(
                    StatisticalSelector(
                        max_features=self.max_features,
                        method=self.statistical_method,
                    )
                )
            elif self.selection_method == "shap":
                self._selectors.append(
                    ShapSelector(
                        max_features=self.max_features,
                        random_state=self.random_state,
                    )
                )

    def _fit_selectors(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit selectors sequentially, each on output of previous."""
        X_current = X
        for selector in self._selectors:
            selector.fit(X_current, y)
            X_current = selector.transform(X_current)
        return X_current
```

### Order Rationale

1. **Variance first**: Removes obviously useless features (constant or near-constant). Fast, no false positives.

2. **Correlation second**: Removes redundant information. Among correlated features, keeps one representative.

3. **Importance last**: Most computationally expensive. Benefits from reduced feature set from previous steps.

## Consequences

### Positive

1. **Interpretability**: Clear understanding of why features were removed:
   ```python
   # Can inspect each selector's decisions
   print(f"Removed {n_low_var} low-variance features")
   print(f"Removed {n_correlated} correlated features")
   print(f"Selected top {n_important} by importance")
   ```

2. **Efficiency**: Later selectors work on reduced feature sets:
   ```python
   # 1000 features → 950 after variance → 600 after correlation → 100 final
   # Importance selector only evaluates 600, not 1000
   ```

3. **Predictable behavior**: Same input always produces same output (given random state).

4. **Debuggability**: Can identify which stage removed a feature:
   ```python
   # Check if feature was removed by variance
   if not variance_selector.get_support()[feature_idx]:
       print(f"Feature {name} removed: low variance")
   ```

5. **Simplicity**: Easy to understand and explain:
   - "We remove useless features, then redundant features, then keep the best."

### Negative

1. **Order sensitivity**: Different orders could yield different results:
   ```python
   # If correlation checked before variance:
   # - Might keep a correlated feature that has low variance
   # - That feature gets removed by variance selector anyway

   # Current order avoids this issue
   ```

2. **Potential over-filtering**: Sequential removal can be aggressive:
   ```python
   # Feature A: low variance but highly predictive
   # Removed by variance selector, never evaluated by importance
   ```

3. **No recovery**: Once a selector removes a feature, later selectors can't recover it:
   ```python
   # If correlation selector removes feature X
   # Importance selector never sees X, even if X is important
   ```

4. **Threshold interactions**: Thresholds at each stage interact:
   ```python
   # Tight variance threshold + tight correlation threshold
   # = possibly too few features for importance selector
   ```

### Mitigations

1. **Conservative default thresholds**:
   ```python
   variance_threshold = 0.01      # Only truly constant features
   correlation_threshold = 0.95   # Only nearly identical features
   ```

2. **Skip selectors option**:
   ```python
   AutoFeatureTransformer(
       remove_low_variance=False,  # Skip variance selector
       remove_correlated=False,    # Skip correlation selector
   )
   ```

3. **Feature importance before removal**:
   ```python
   # Compute importance on full set for reporting
   # Then apply selectors
   ```

## Alternatives Considered

### 1. Voting/Ensemble Selection

Each selector votes on features, keep those with majority votes.

```python
votes = np.zeros(n_features)
for selector in selectors:
    selector.fit(X, y)
    votes += selector.get_support().astype(int)

selected = votes >= (len(selectors) / 2)
```

**Rejected because:**
- Variance selector would "vote" for almost all features
- Unequal selector quality not accounted for
- Harder to explain results

### 2. Union Selection

Keep features selected by any selector.

```python
selected = np.zeros(n_features, dtype=bool)
for selector in selectors:
    selected |= selector.get_support()
```

**Rejected because:**
- Doesn't reduce features much
- Defeats purpose of selection
- Includes low-quality features

### 3. Intersection Selection

Keep only features selected by all selectors.

```python
selected = np.ones(n_features, dtype=bool)
for selector in selectors:
    selected &= selector.get_support()
```

**Rejected because:**
- Too aggressive
- Different selectors have different criteria
- Might select zero features

### 4. Weighted Score Combination

Compute weighted score across selectors.

```python
scores = np.zeros(n_features)
for selector, weight in zip(selectors, weights):
    scores += weight * selector.get_scores()

selected = np.argsort(scores)[-max_features:]
```

**Rejected because:**
- How to weight different selectors?
- Score scales differ (variance vs. importance)
- Complex to tune

## Configuration

Users can customize the sequential pipeline:

```python
# Skip variance and correlation, just use importance
AutoFeatureTransformer(
    remove_low_variance=False,
    remove_correlated=False,
    selection_method="importance",
    max_features=50,
)

# Aggressive filtering
AutoFeatureTransformer(
    variance_threshold=0.1,        # Remove more low-variance
    correlation_threshold=0.8,     # Remove more correlated
    max_features=20,               # Keep only top 20
)

# Conservative filtering
AutoFeatureTransformer(
    variance_threshold=0.001,      # Only constant features
    correlation_threshold=0.99,    # Only duplicates
    max_features=500,              # Keep many features
)
```

## Related Decisions

- [ADR-0008: Composition-Based Pipeline](adr-0008-composition-based-pipeline.md) - Selectors are composed components
- [ADR-0001: sklearn Compatibility](adr-0001-sklearn-compatibility.md) - Selectors follow sklearn selector interface
