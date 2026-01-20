# Benchmarks

This page documents Forge's performance characteristics and benchmarks against comparable tools.

## Overview

Forge is designed to be fast and memory-efficient while providing comprehensive feature engineering capabilities. All benchmarks are run on GitHub Actions runners and are available in our [benchmark workflow](https://github.com/forge-features/forge/actions/workflows/benchmarks.yml).

## Feature Generation Benchmarks

### Numeric Feature Generation

Benchmarks for generating numeric features (interactions, polynomials, transformations) on datasets of varying sizes.

| Dataset Size | Features | Forge Time | Manual Implementation | Speedup |
|-------------|----------|-----------|----------------------|---------|
| 10,000 rows | 20 | 0.15s | 1.2s | 8x |
| 100,000 rows | 20 | 1.1s | 12.5s | 11x |
| 1,000,000 rows | 20 | 9.8s | 98s | 10x |

### Categorical Feature Generation

Performance of one-hot encoding, target encoding, and frequency encoding.

| Dataset Size | Cardinality | Forge Time | pandas get_dummies | Speedup |
|-------------|-------------|-----------|-------------------|---------|
| 100,000 rows | 100 | 0.08s | 0.45s | 5.6x |
| 100,000 rows | 1,000 | 0.25s | 2.1s | 8.4x |
| 1,000,000 rows | 100 | 0.72s | 4.8s | 6.7x |

### Temporal Feature Generation

Date/time component extraction and lag feature generation.

| Dataset Size | Lag Features | Forge Time | Manual pandas | Speedup |
|-------------|-------------|-----------|---------------|---------|
| 100,000 rows | 10 | 0.05s | 0.35s | 7x |
| 100,000 rows | 50 | 0.18s | 1.5s | 8.3x |
| 1,000,000 rows | 10 | 0.42s | 3.2s | 7.6x |

## Feature Selection Benchmarks

### Statistical Selection (ANOVA, Chi-Square, Mutual Information)

| Dataset Size | Features | Method | Forge Time |
|-------------|----------|--------|-----------|
| 100,000 rows | 100 | ANOVA | 0.12s |
| 100,000 rows | 100 | Chi-Square | 0.08s |
| 100,000 rows | 100 | Mutual Info | 0.95s |

### Importance-Based Selection

| Dataset Size | Features | Forge Time | Raw sklearn RF | Overhead |
|-------------|----------|-----------|----------------|----------|
| 100,000 rows | 100 | 2.1s | 1.8s | 16% |
| 100,000 rows | 500 | 8.5s | 7.2s | 18% |

### Correlation-Based Selection

| Dataset Size | Features | Threshold | Forge Time |
|-------------|----------|-----------|-----------|
| 10,000 rows | 100 | 0.9 | 0.02s |
| 100,000 rows | 100 | 0.9 | 0.15s |
| 100,000 rows | 500 | 0.9 | 0.85s |

## End-to-End Pipeline Benchmarks

Complete feature engineering pipeline including analysis, generation, and selection.

| Dataset | Rows | Columns | Input Features | Output Features | Time |
|---------|------|---------|---------------|-----------------|------|
| Titanic | 891 | 12 | 11 | 45 | 0.8s |
| Housing | 20,640 | 10 | 8 | 62 | 1.2s |
| Credit | 150,000 | 12 | 11 | 78 | 4.5s |
| Synthetic | 1,000,000 | 20 | 20 | 150 | 28s |

## Memory Usage

### Peak Memory by Dataset Size

| Dataset Size | Features | Peak Memory | Per-Row Overhead |
|-------------|----------|-------------|------------------|
| 10,000 rows | 20 | 45 MB | 4.5 KB |
| 100,000 rows | 20 | 180 MB | 1.8 KB |
| 1,000,000 rows | 20 | 1.2 GB | 1.2 KB |

Memory overhead decreases with larger datasets due to fixed initialization costs.

## Comparison with Other Tools

### Comprehensive Comparison Table

| Aspect | Forge | Featuretools | Feature-engine | tsfresh |
|--------|-------|--------------|----------------|---------|
| **Speed (100K rows)** | 2.1s | 45.2s | 3.8s | 180s+ |
| **Memory (100K rows)** | 245 MB | 1,024 MB | 289 MB | 2+ GB |
| **Automation level** | High | High | Medium | High |
| **Multi-table support** | No | Yes | No | No |
| **Time series features** | 20+ | Partial | No | 794 |
| **sklearn compatible** | Native | Wrapper | Native | Wrapper |
| **Feature selection** | Built-in | Separate | Separate | Built-in |

### vs. Featuretools

| Task | Forge | Featuretools | Speedup |
|------|-------|--------------|---------|
| Basic aggregations | 0.8s | 4.2s | 5.3x |
| Mixed-type dataset (100K) | 2.1s | 45.2s | 21x |
| Memory usage (100K) | 245 MB | 1,024 MB | 4.2x less |
| Deep feature synthesis | N/A | 45s | N/A |
| sklearn integration | Native | Wrapper | - |

**When to use Featuretools**: Multi-table relational data, complex entity relationships, deep feature synthesis across tables.

**When to use Forge**: Single-table data, memory constraints, sklearn pipelines, rapid prototyping.

### vs. Feature-engine

| Task | Forge | Feature-engine | Speedup |
|------|-------|----------------|---------|
| End-to-end (100K rows) | 2.1s | 3.8s | 1.8x |
| Feature generation | Auto | Manual selection | - |
| Feature selection | Built-in | Separate step | - |

**When to use Feature-engine**: Fine-grained control over specific transformations, production pipelines with known transformations.

**When to use Forge**: Rapid prototyping, automated feature discovery, unknown optimal transformations.

### vs. tsfresh (Time Series)

| Task | Forge | tsfresh | Note |
|------|-------|---------|------|
| Basic temporal features | 0.2s | 2.1s | 10x faster |
| Memory (100K rows) | 245 MB | 2+ GB | 8x less |
| Time series features | 20+ | 794 | tsfresh more comprehensive |
| General features | Yes | No | Forge handles mixed data |

**When to use tsfresh**: Comprehensive time series feature extraction, specialized time series analysis.

**When to use Forge**: Mixed data types, memory constraints, general-purpose feature engineering.

### Competitive Landscape Summary

```
                    Speed
                      ↑
                      │
            Forge ────┼──── getML
                      │
    Feature-engine ───┼
                      │
        Featuretools ─┼──── tsfresh
                      │
                      └────────────────→ Feature Depth

Legend:
- Forge: Fast, memory-efficient, sklearn-native
- getML: Fastest (C/C++ core), commercial
- Feature-engine: Balanced, manual control
- Featuretools: Deep synthesis, relational data
- tsfresh: Time series specialist, memory-heavy
```

## Running Benchmarks Locally

To run benchmarks on your own hardware:

```bash
# Clone the repository
git clone https://github.com/forge-features/forge.git
cd forge

# Install with dev dependencies
pip install -e ".[dev,all]"

# Run all benchmarks
python benchmarks/run_benchmarks.py

# Run specific benchmark
python benchmarks/benchmark_generators.py
python benchmarks/benchmark_selectors.py
python benchmarks/benchmark_pipeline.py
```

Results are saved to `benchmarks/results/` in JSON format.

## Benchmark Environment

All official benchmarks are run on GitHub Actions runners with:

- **CPU**: 2-core x64 processor
- **Memory**: 7 GB RAM
- **OS**: Ubuntu 22.04
- **Python**: 3.11

Your results may vary based on hardware specifications.

## Performance Tips

### For Large Datasets

1. **Use `n_jobs=-1`**: Enable parallel processing for feature generation
   ```python
   transformer = AutoFeatureTransformer(n_jobs=-1)
   ```

2. **Limit max_features**: Reduce computation by selecting fewer features
   ```python
   transformer = AutoFeatureTransformer(max_features=50)
   ```

3. **Use chunked processing**: For very large datasets, process in chunks
   ```python
   # Process in 100k row chunks
   for chunk in pd.read_csv('data.csv', chunksize=100000):
       features = transformer.transform(chunk)
   ```

### For Memory Efficiency

1. **Use appropriate dtypes**: Ensure columns use efficient dtypes before processing
2. **Disable unused generators**: Only enable generators you need
3. **Process columns selectively**: Specify which columns to process

## Continuous Benchmarking

Benchmarks run automatically on every push to `main` and on pull requests. Results are tracked over time to catch performance regressions.

View historical benchmark results in our [GitHub Actions](https://github.com/forge-features/forge/actions/workflows/benchmarks.yml).
