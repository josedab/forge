# Feature Status

This document tracks the maturity level of each Forge module so users and contributors know what to expect.

## Maturity Levels

| Level | Meaning | API Stability |
|-------|---------|---------------|
| **Stable** | Production-ready, well-tested, documented | Breaking changes only in major versions |
| **Beta** | Functional and tested, API may change | Minor version changes possible |
| **Alpha** | Experimental, may have rough edges | API is not guaranteed |

## Module Status

### Core (Stable)

| Module | Description | Status |
|--------|-------------|--------|
| `forge.analyzer` | Data analysis and type inference | ✅ Stable |
| `forge.generators` | Feature generation (numeric, categorical, temporal, text) | ✅ Stable |
| `forge.selectors` | Feature selection (statistical, importance, correlation, variance) | ✅ Stable |
| `forge.transformers` | sklearn-compatible transformers (`AutoFeatureTransformer`, `ForgePipeline`) | ✅ Stable |
| `forge.missing` | Missing value imputation and indicators | ✅ Stable |
| `forge.outliers` | Outlier detection and handling | ✅ Stable |
| `forge.utils` | Shared utilities (validation, parallel, logging) | ✅ Stable |
| `forge.types` | Type definitions | ✅ Stable |
| `forge.exceptions` | Custom exception hierarchy | ✅ Stable |

### Extensions (Beta)

| Module | Description | Status | Requires |
|--------|-------------|--------|----------|
| `forge.visualization` | Feature importance, correlation, and distribution plots | 🔶 Beta | `forge-features[viz]` |
| `forge.dsl` | Domain-specific language for pipeline definitions | 🔶 Beta | — |
| `forge.search` | Feature search (random, evolutionary, RL-based) | 🔶 Beta | — |
| `forge.monitoring` | Drift detection, PSI, observability, retraining triggers | 🔶 Beta | — |
| `forge.documentation` | Auto-generated feature documentation | 🔶 Beta | — |
| `forge.transfer` | Pipeline transfer and adaptation hub | 🔶 Beta | — |
| `forge.contracts` | Feature contracts and validation | 🔶 Beta | — |
| `forge.freshness` | Feature freshness tracking | 🔶 Beta | — |
| `forge.registry` | Feature registry | 🔶 Beta | — |
| `forge.cache` | Feature caching layer | 🔶 Beta | — |

### Advanced (Alpha)

| Module | Description | Status | Requires |
|--------|-------------|--------|----------|
| `forge.llm` | LLM-powered feature generation and explanation | 🧪 Alpha | `forge-features[llm]` |
| `forge.nlgen` | Natural language-to-features pipeline | 🧪 Alpha | — |
| `forge.privacy` | Differential privacy and federated learning | 🧪 Alpha | — |
| `forge.fairness` | Bias detection and mitigation | 🧪 Alpha | — |
| `forge.serving` | Real-time serving and ONNX export | 🧪 Alpha | `forge-features[serving]` |
| `forge.streaming` | Streaming feature computation | 🧪 Alpha | — |
| `forge.marketplace` | Feature pack marketplace | 🧪 Alpha | — |
| `forge.studio` | Visual pipeline builder | 🧪 Alpha | — |
| `forge.dashboard` | Streamlit dashboard | 🧪 Alpha | `forge-features[dashboard]` |
| `forge.catalog` | Feature catalog API | 🧪 Alpha | `forge-features[catalog-api]` |
| `forge.cli` | Command-line interface | 🧪 Alpha | — |
| `forge.multitable` | Multi-table feature synthesis | 🧪 Alpha | — |
| `forge.multimodal` | Multi-modal feature fusion | 🧪 Alpha | — |
| `forge.graph` | Graph-based features | 🧪 Alpha | — |
| `forge.neural` | Neural feature learning | 🧪 Alpha | `forge-features[neural]` |
| `forge.simulator` | Data simulation for testing | 🧪 Alpha | — |
| `forge.recommend` | Feature recommendations | 🧪 Alpha | — |

### Infrastructure (Beta)

| Module | Description | Status | Requires |
|--------|-------------|--------|----------|
| `forge.backends` | Compute backends (Dask, Ray, Spark) | 🔶 Beta | `forge-features[dask]` / `[ray]` / `[spark]` |
| `forge.distributed` | Distributed execution coordination | 🔶 Beta | — |
| `forge.core_engine` | Core execution engine | 🔶 Beta | — |
| `forge.experiments` | Experiment tracking | 🔶 Beta | — |
| `forge.packs` | Feature pack system | 🔶 Beta | — |

## Updating This Document

When changing a module's status:
1. Update the table above
2. Add a note in the CHANGELOG
3. If moving from Alpha → Beta, ensure test coverage > 80%
4. If moving from Beta → Stable, ensure documentation is complete
