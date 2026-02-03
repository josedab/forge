"""Forge CLI entry point — argparse-based command dispatcher.

Provides subcommands: init, analyze, generate, select, run, test, validate,
info, serve, benchmark.

Usage:
    python -m forge.cli.main analyze data.csv
    python -m forge.cli.main generate data.csv --max-features 50
    python -m forge.cli.main select data.csv --target target --method importance
    python -m forge.cli.main benchmark data.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _load_data(path: str, target: str | None = None) -> tuple[pd.DataFrame, pd.Series | None]:  # type: ignore[type-arg]
    """Load a CSV/Parquet file and optionally split target."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)

    y = None
    if target and target in df.columns:
        y = df.pop(target)

    return df, y


def cmd_analyze(args: argparse.Namespace) -> int:
    """Analyze a dataset and report statistics, types, and quality."""
    from forge.dashboard.profiler import DatasetProfiler

    df, _ = _load_data(args.data)
    print(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns from {args.data}")

    profiler = DatasetProfiler()
    profile = profiler.profile(df)

    print(f"\n{'=' * 60}")
    print("DATASET OVERVIEW")
    print(f"{'=' * 60}")
    print(f"  Rows:           {profile.n_rows:,}")
    print(f"  Columns:        {profile.n_columns}")
    print(f"  Memory:         {profile.memory_mb:.2f} MB")
    print(f"  Overall Quality:{profile.overall_quality:.0%}")
    print(f"  Duplicates:     {profile.duplicate_rows:,} ({profile.duplicate_pct:.1f}%)")

    print(f"\n{'=' * 60}")
    print("COLUMN TYPES")
    print(f"{'=' * 60}")
    for type_name, count in sorted(profile.type_summary.items()):
        print(f"  {type_name:<15} {count}")

    print(f"\n{'=' * 60}")
    print("COLUMN PROFILES")
    print(f"{'=' * 60}")
    print(f"  {'Column':<25} {'Type':<12} {'Null%':>6} {'Unique':>7} {'Quality':>8}")
    print(f"  {'-' * 25} {'-' * 12} {'-' * 6} {'-' * 7} {'-' * 8}")
    for cp in profile.column_profiles:
        print(
            f"  {cp.name:<25} {cp.inferred_type:<12} {cp.null_pct:>5.1f}% "
            f"{cp.unique_count:>7} {cp.quality_score:>7.0%}"
        )

    if profile.recommendations:
        print(f"\n{'=' * 60}")
        print("RECOMMENDATIONS")
        print(f"{'=' * 60}")
        for i, rec in enumerate(profile.recommendations, 1):
            print(f"  {i}. {rec}")

    if args.output:
        output = {
            "n_rows": profile.n_rows,
            "n_columns": profile.n_columns,
            "memory_mb": profile.memory_mb,
            "quality": profile.overall_quality,
            "type_summary": profile.type_summary,
            "columns": [
                {
                    "name": cp.name,
                    "type": cp.inferred_type,
                    "null_pct": cp.null_pct,
                    "unique": cp.unique_count,
                    "quality": cp.quality_score,
                    "issues": cp.quality_issues,
                    "generators": cp.recommended_generators,
                }
                for cp in profile.column_profiles
            ],
            "recommendations": profile.recommendations,
        }
        Path(args.output).write_text(json.dumps(output, indent=2, default=str))
        print(f"\nReport saved to {args.output}")

    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    """Generate features from a dataset."""
    from forge.transformers.auto_transformer import AutoFeatureTransformer

    df, y = _load_data(args.data, target=args.target)
    print(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")

    max_features = args.max_features if args.max_features > 0 else None
    transformer = AutoFeatureTransformer(
        max_features=max_features,
        selection_method=args.selection,
        verbose=1 if args.verbose else 0,
    )

    start = time.time()
    X_out = transformer.fit_transform(df, y)
    elapsed = time.time() - start

    feature_names = transformer.get_feature_names_out()
    print(f"\nGenerated {len(feature_names)} features in {elapsed:.2f}s")
    print(f"Output shape: {X_out.shape}")

    if args.output:
        if args.output.endswith(".parquet"):
            X_out.to_parquet(args.output, index=False)
        else:
            X_out.to_csv(args.output, index=False)
        print(f"Saved to {args.output}")

    return 0


def cmd_select(args: argparse.Namespace) -> int:
    """Select features from a dataset."""
    df, y = _load_data(args.data, target=args.target)
    print(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")

    method = args.method
    k = args.k if args.k > 0 else max(1, df.shape[1] // 2)

    if method == "importance":
        from forge.selectors.importance import ImportanceSelector
        selector = ImportanceSelector(n_features=k)
    elif method == "correlation":
        from forge.selectors.correlation import CorrelationSelector
        selector = CorrelationSelector(threshold=args.threshold)
    elif method == "variance":
        from forge.selectors.variance import VarianceSelector
        selector = VarianceSelector(threshold=args.threshold)
    elif method == "statistical":
        from forge.selectors.statistical import StatisticalSelector
        selector = StatisticalSelector(n_features=k)
    else:
        print(f"Unknown method: {method}. Use: importance, correlation, variance, statistical")
        return 1

    start = time.time()
    X_out = selector.fit_transform(df, y)
    elapsed = time.time() - start

    selected = selector.get_feature_names_out()
    print(f"\nSelected {len(selected)} / {df.shape[1]} features in {elapsed:.2f}s")
    print(f"Features: {', '.join(selected[:20])}")

    if args.output:
        if args.output.endswith(".parquet"):
            X_out.to_parquet(args.output, index=False)
        else:
            X_out.to_csv(args.output, index=False)
        print(f"Saved to {args.output}")

    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Benchmark feature engineering on a dataset."""
    from forge.transformers.auto_transformer import AutoFeatureTransformer

    df, y = _load_data(args.data, target=args.target)
    print(f"Benchmarking on {df.shape[0]} rows × {df.shape[1]} columns")

    results: list[dict[str, Any]] = []
    n_runs = args.runs

    for i in range(n_runs):
        start = time.time()
        transformer = AutoFeatureTransformer(max_features=50)
        X_out = transformer.fit_transform(df, y)
        elapsed = time.time() - start
        results.append({
            "run": i + 1,
            "time_s": round(elapsed, 3),
            "features_in": df.shape[1],
            "features_out": X_out.shape[1],
        })
        print(f"  Run {i + 1}/{n_runs}: {elapsed:.3f}s → {X_out.shape[1]} features")

    times = [r["time_s"] for r in results]
    print(f"\n{'=' * 40}")
    print(f"  Mean time:   {np.mean(times):.3f}s")
    print(f"  Std time:    {np.std(times):.3f}s")
    print(f"  Min time:    {np.min(times):.3f}s")
    print(f"  Max time:    {np.max(times):.3f}s")
    print(f"{'=' * 40}")

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"Results saved to {args.output}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="forge",
        description="Forge — Automated Feature Engineering Platform",
    )
    parser.add_argument(
        "--version", action="store_true", help="Show version and exit"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Analyze a dataset")
    p_analyze.add_argument("data", help="Path to CSV/Parquet file")
    p_analyze.add_argument("-o", "--output", help="Save report as JSON")

    # generate
    p_gen = subparsers.add_parser("generate", help="Generate features")
    p_gen.add_argument("data", help="Path to CSV/Parquet file")
    p_gen.add_argument("--target", help="Target column name")
    p_gen.add_argument("--max-features", type=int, default=50, help="Max features")
    p_gen.add_argument("--selection", default="importance", help="Selection method")
    p_gen.add_argument("-o", "--output", help="Save output features")
    p_gen.add_argument("--verbose", action="store_true")

    # select
    p_sel = subparsers.add_parser("select", help="Select features")
    p_sel.add_argument("data", help="Path to CSV/Parquet file")
    p_sel.add_argument("--target", help="Target column name")
    p_sel.add_argument("--method", default="importance",
                       choices=["importance", "correlation", "variance", "statistical"])
    p_sel.add_argument("-k", type=int, default=0, help="Number of features to select")
    p_sel.add_argument("--threshold", type=float, default=0.95, help="Threshold for correlation/variance")
    p_sel.add_argument("-o", "--output", help="Save selected features")

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Benchmark feature engineering")
    p_bench.add_argument("data", help="Path to CSV/Parquet file")
    p_bench.add_argument("--target", help="Target column name")
    p_bench.add_argument("--runs", type=int, default=3, help="Number of runs")
    p_bench.add_argument("-o", "--output", help="Save results as JSON")

    # init
    p_init = subparsers.add_parser("init", help="Initialize a new Forge project")
    p_init.add_argument("name", help="Project name")
    p_init.add_argument("--template", default="default", help="Project template")

    # run
    p_run = subparsers.add_parser("run", help="Run a pipeline script")
    p_run.add_argument("script", nargs="?", default="pipeline.py", help="Pipeline script")

    # test
    p_test = subparsers.add_parser("test", help="Run feature tests")
    p_test.add_argument("test_dir", nargs="?", default="tests", help="Test directory")

    # validate
    p_val = subparsers.add_parser("validate", help="Validate project config")
    p_val.add_argument("config", nargs="?", default="forge.yaml", help="Config file")

    # info
    subparsers.add_parser("info", help="Show project information")

    # serve
    p_serve = subparsers.add_parser("serve", help="Generate serving config")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--host", default="0.0.0.0")

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI main entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        from forge._version import __version__
        print(f"forge {__version__}")
        return 0

    if not args.command:
        parser.print_help()
        return 0

    from forge.cli import ForgeCLI

    if args.command == "analyze":
        return cmd_analyze(args)
    elif args.command == "generate":
        return cmd_generate(args)
    elif args.command == "select":
        return cmd_select(args)
    elif args.command == "benchmark":
        return cmd_benchmark(args)
    else:
        cli = ForgeCLI()
        if args.command == "init":
            result = cli.init(args.name, template=args.template)
        elif args.command == "run":
            result = cli.run(args.script)
        elif args.command == "test":
            result = cli.test(args.test_dir)
        elif args.command == "validate":
            result = cli.validate(args.config)
        elif args.command == "info":
            result = cli.info()
        elif args.command == "serve":
            result = cli.serve_config(port=args.port, host=args.host)
        else:
            parser.print_help()
            return 1

        print(result.message)
        if result.data:
            for k, v in result.data.items():
                print(f"  {k}: {v}")
        return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
