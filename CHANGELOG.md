# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0 (2026-02-11)


### Features

* add CLI, dashboard, catalog, experiments, and utility modules ([27b444c](https://github.com/josedab/forge/commit/27b444c9423ae7b0dde349513a3229e31f8b0d73))
* **analyzer:** add data analysis module ([27ad2bc](https://github.com/josedab/forge/commit/27ad2bc41c83b6911ba76e7fa412b174a671b9f1))
* **backends:** add compute backends for Dask, Ray, and Spark ([98963f4](https://github.com/josedab/forge/commit/98963f44f6dd2a462fa6554fd2bb2941a1667b45))
* **cache:** add feature caching layer with TTL support ([79f6931](https://github.com/josedab/forge/commit/79f6931b402280f21efbc274ad125e7bb51273f5))
* **contracts:** add feature contracts and validation framework ([90d0b7e](https://github.com/josedab/forge/commit/90d0b7e6a64786ec0ac68e54b754d90b7f30befd))
* **core:** add version, types, and exceptions ([e908a82](https://github.com/josedab/forge/commit/e908a82b0ee158ee973127e0eadf3a608a097659))
* **core:** export new modules from forge package ([6c149c3](https://github.com/josedab/forge/commit/6c149c37605d62190b5c7deab22ce3b827288dcd))
* **core:** update package exports and module init files ([a7f63ba](https://github.com/josedab/forge/commit/a7f63bae4c39f6b31b56e180f20246b23a740fd1))
* **distributed:** add distributed execution coordination ([61b4cdb](https://github.com/josedab/forge/commit/61b4cdb80341519aca2aa0a2e21b02747a4d5a60))
* **documentation:** add automated feature documentation generator ([afc4bc7](https://github.com/josedab/forge/commit/afc4bc73761bdc00b037e632c9aeba3da3cd6606))
* **documentation:** add enhanced doc generator with multi-format export ([ba9952a](https://github.com/josedab/forge/commit/ba9952a97d5e91fc00f30c9287cc911111b3f107))
* **documentation:** add governance and PII scanner ([c95df59](https://github.com/josedab/forge/commit/c95df59e4b533c9fc78dbac8b512680549ccfa92))
* **dsl:** add DSL parser and compiler for pipeline specifications ([3069bcd](https://github.com/josedab/forge/commit/3069bcd190c7e8148dd540c4353d9bf7564e1d27))
* **dsl:** add SQL transpiler for Postgres, BigQuery, and Snowflake ([c369416](https://github.com/josedab/forge/commit/c36941638391fdcf36d1ed8fdff4f16af52e4337))
* **fairness:** add bias detection and fairness assessment ([70cb746](https://github.com/josedab/forge/commit/70cb746237cdb63cf7cbbb8a4d7814a697134b3d))
* **generators:** add categorical feature generators ([2b4061a](https://github.com/josedab/forge/commit/2b4061afd980f12bfe50a7a8b62166d968a83324))
* **generators:** add feature interaction discovery ([43e74b8](https://github.com/josedab/forge/commit/43e74b850eb64da5ed4417b93e02d0bae2a06629))
* **generators:** add generator base and registry ([befd4c6](https://github.com/josedab/forge/commit/befd4c658364c18b4fd95984491bd94b4f44a615))
* **generators:** add numeric feature generators ([e72b0b0](https://github.com/josedab/forge/commit/e72b0b09476850850eb5e57abdf9ad2a9e6826fb))
* **generators:** add temporal feature generators ([0d1aa3f](https://github.com/josedab/forge/commit/0d1aa3f8a547a58b7d75e8da6de03561ff702dd9))
* **generators:** add text feature generators ([2839d5e](https://github.com/josedab/forge/commit/2839d5eb8c5fde447266d4c9d32f131eba478e79))
* **generators:** add time-series feature generators ([4922b51](https://github.com/josedab/forge/commit/4922b5176ca28f3d894775a843ecedb345f015da))
* **graph:** add graph-based feature generation ([05fa5bf](https://github.com/josedab/forge/commit/05fa5bfa3fbfc2dc9c1d0d2e159910207a809f2a))
* **llm:** add LLM-powered feature discovery module ([a5c6a4b](https://github.com/josedab/forge/commit/a5c6a4b98f8ff9684b4b7ea1f88817ec2774ea45))
* **marketplace:** add feature pack marketplace and hub ([c1c6394](https://github.com/josedab/forge/commit/c1c63943d9872407b781d1d6ffeb1ec4c7e2bc04))
* **missing:** add missing value handling ([8687615](https://github.com/josedab/forge/commit/86876154d7a37df56a8707cbcb5b21099f115510))
* **monitoring:** add drift detection module ([a4e3c5a](https://github.com/josedab/forge/commit/a4e3c5a7dfb1ba4ff893426824e67adb4dde9681))
* **monitoring:** add observability, metrics store, and drift detection enhancements ([7648615](https://github.com/josedab/forge/commit/7648615c2dd0f670ba0d219807a2f9adaf06d621))
* **monitoring:** add retraining triggers and importance tracking ([9a49f95](https://github.com/josedab/forge/commit/9a49f952398dd3bb34ccb4e3dd725ef68896efb0))
* **multimodal:** add multi-modal feature fusion ([6533178](https://github.com/josedab/forge/commit/6533178ca629d7633ee593cb8f4f1dff07fa2201))
* **multitable:** add multi-table feature synthesis ([8896293](https://github.com/josedab/forge/commit/8896293b6cd4b36464bebaaa71ec9ea7e94186c6))
* **neural:** add neural feature learning module ([ee986a1](https://github.com/josedab/forge/commit/ee986a16b184426046beb1c68f385244704807cb))
* **nlgen:** add natural language feature engineering auto-pilot ([36d64e6](https://github.com/josedab/forge/commit/36d64e64377c175daec8cb88633a3cfcb83073d6))
* **nlgen:** add NL feature synthesis and LLM-powered generation ([752680d](https://github.com/josedab/forge/commit/752680d446a846143e93c8f5819561ccd02e819b))
* **outliers:** add outlier handling module ([593924a](https://github.com/josedab/forge/commit/593924a2a683e21ac3056800b176ed8ad0358877))
* **privacy:** add differential privacy and privacy budget management ([e532e51](https://github.com/josedab/forge/commit/e532e51ecb73ba48620cc3fc520f63bc9ac008a5))
* **privacy:** add federated feature engineering with secure aggregation ([5d8e530](https://github.com/josedab/forge/commit/5d8e5301b3563e13684e2eb55efc519f26b8d979))
* **registry:** add feature registry with local and Feast backends ([0d9f861](https://github.com/josedab/forge/commit/0d9f861cd43319f7b926f0e4d0d0069bb0cff952))
* **search:** add feature search with grammar and evolutionary strategies ([d1c719f](https://github.com/josedab/forge/commit/d1c719fafbe9377b10de67cb2409ff07e0be9a51))
* **search:** add reinforcement learning feature search ([ba8c22d](https://github.com/josedab/forge/commit/ba8c22de298c14a9a6d88318d8feb046737ad693))
* **selectors:** add AutoML and ensemble feature selectors ([913516a](https://github.com/josedab/forge/commit/913516af47b75f968ddd5decbf60d6547da692fe))
* **selectors:** add feature selection module ([5b33717](https://github.com/josedab/forge/commit/5b3371705766837e2db1ad9bdece45445a8fbda5))
* **serving:** add ONNX export, NumPy fast path, and stream processor ([ccb4934](https://github.com/josedab/forge/commit/ccb4934b08667fa8a34959f32818afe9b2b39b4f))
* **serving:** add real-time serving and compiled pipeline support ([da53b2a](https://github.com/josedab/forge/commit/da53b2a9ea8728a14eb745e7537476998f509c34))
* **streaming:** add streaming feature computation engine ([77065d6](https://github.com/josedab/forge/commit/77065d6b05c50b137e52ddfc24906ae08cc9f23e))
* **studio:** add pipeline persistence with versioning ([6a01508](https://github.com/josedab/forge/commit/6a01508051c58682bab8454b40f4ec00911cef7c))
* **studio:** add visual pipeline builder framework ([384fa77](https://github.com/josedab/forge/commit/384fa7723285bcaa0dc0f5a13844c114546a442b))
* **transfer:** add marketplace search, ratings, and pack bundles ([c16ac10](https://github.com/josedab/forge/commit/c16ac10b3d760553d7e7dfe871875c64bcf962e8))
* **transfer:** add pipeline transfer hub and adaptation engine ([c6df4e1](https://github.com/josedab/forge/commit/c6df4e1d11eca7cbe8c7da7746d8101eba7190ef))
* **transformers:** add main transformer classes ([aeade2b](https://github.com/josedab/forge/commit/aeade2bea824575c3ccd1ac18f33c767d587b86a))
* **transformers:** add pipeline debugger with step-through and lineage DAG ([b0490ff](https://github.com/josedab/forge/commit/b0490fffc0ad29e787566eeebb6dc0fa71bb962e))
* **transformers:** add sqrt transform support to AutoFeatureTransformer ([f558bd0](https://github.com/josedab/forge/commit/f558bd0daad4a1005d173f1a2cc61130ed06dac9))
* **utils:** add utility modules ([77be080](https://github.com/josedab/forge/commit/77be080414b85eead2c9a866bfa55f138fae476f))
* **visualization:** add visualization module ([b631e4b](https://github.com/josedab/forge/commit/b631e4b15b1d1cd33ecd2c928226a6469583240d))


### Bug Fixes

* **analyzer:** remove deprecated infer_datetime_format parameter ([b7c254b](https://github.com/josedab/forge/commit/b7c254b9126075199416b1b27ffd5dd47e596bd2))
* **analyzer:** remove trailing commas from list initializations ([8cf7018](https://github.com/josedab/forge/commit/8cf701853721fdfaf5c36b5c687736de0b90776a))
* **generators:** fix syntax errors with trailing commas and missing list separators ([092284c](https://github.com/josedab/forge/commit/092284c90bd74445d6a0ce2f4926f58b122034e6))
* **selectors:** remove duplicate import and fix trailing commas ([50bb9f0](https://github.com/josedab/forge/commit/50bb9f05181c8f4e9f8305d001d552599a9a2263))
* **transformers:** handle onehot encoding type in FeatureDescriber ([66756f3](https://github.com/josedab/forge/commit/66756f32f03ba67803f09382dec72480927fd91b))
* **types:** remove trailing comma in FeatureInfo dataclass ([d4b0813](https://github.com/josedab/forge/commit/d4b08130346e0a85e1aedf1285ef8139c5a901e9))


### Performance Improvements

* **benchmarks:** add benchmark suite ([4c810db](https://github.com/josedab/forge/commit/4c810db8eaf61ac8e42dc4178fb3c36a74a6990d))


### Documentation

* add academic paper files ([d4c0e7e](https://github.com/josedab/forge/commit/d4c0e7e8ace810e34e6433a6b96267f6e5b9e9c2))
* add documentation and guides ([46cec88](https://github.com/josedab/forge/commit/46cec88262c5a4ea158f496c84d575359f5e9841))
* add smoke tests, fast test targets, and DX improvements ([9ef262c](https://github.com/josedab/forge/commit/9ef262c5eeddaa9bfbc1c52c8d2f68ffde7c0807))
* **api:** update API documentation for generators and selectors ([a9881fa](https://github.com/josedab/forge/commit/a9881fa2e30d5d7fba03c44fb83e52be08973815))
* **examples:** add usage examples ([cc7d3b3](https://github.com/josedab/forge/commit/cc7d3b301ad663b4fcf6e56272ad31eb5467a7d8))
* **notebooks:** add interactive notebooks ([bcad60d](https://github.com/josedab/forge/commit/bcad60d12142f63bea6cb5b152e19bf1dab6eb9c))

## [Unreleased]

## [0.1.0] - 2025-01-18

### Added

- **Core Framework**
  - `AutoFeatureTransformer` - Main entry point for automatic feature engineering
  - `ForgePipeline` - Custom pipeline for chaining feature transformations
  - Full scikit-learn compatibility (BaseEstimator, TransformerMixin)

- **Data Analysis**
  - `DataAnalyzer` - Automatic data profiling and analysis
  - Automatic column type inference (numeric, categorical, temporal, text)
  - Statistical profiling and data quality assessment
  - Analysis reports with actionable insights

- **Feature Generators**
  - Numeric: Aggregations, interactions, transformations, polynomials
  - Categorical: One-hot, target, frequency, ordinal encoding
  - Temporal: Components extraction, lags, rolling windows, differences
  - Text: Basic stats (length, word count), TF-IDF vectorization

- **Feature Selectors**
  - Statistical: Chi-square, ANOVA F-test, mutual information
  - Importance-based: Tree-based feature importance
  - Correlation: Remove highly correlated features
  - Variance: Remove low-variance features
  - SHAP: SHAP value-based selection (optional)

- **Missing Value Handling**
  - Multiple imputation strategies (mean, median, mode, constant, knn)
  - Missing value indicators

- **Visualization** (optional)
  - Feature importance plots
  - Correlation heatmaps
  - SHAP visualizations
  - Distribution plots

- **Utilities**
  - Input validation
  - Parallel processing support
  - Pipeline serialization (save/load)
  - Configurable logging

### Dependencies

- Core: numpy, pandas, scikit-learn, scipy
- Optional: matplotlib, seaborn (viz), shap, xgboost, lightgbm (boosting)

[Unreleased]: https://github.com/forge-features/forge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/forge-features/forge/releases/tag/v0.1.0
