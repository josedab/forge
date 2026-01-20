#!/usr/bin/env python3
"""Main benchmark runner script."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from benchmark_generators import run_generator_benchmarks
from benchmark_selectors import run_selector_benchmarks
from benchmark_pipeline import run_pipeline_benchmarks, compare_results


def format_for_github_benchmark(results: list[dict]) -> list[dict]:
    """Format results for github-action-benchmark."""
    formatted = []
    for r in results:
        if "error" in r:
            continue

        name_parts = [r.get("generator") or r.get("selector") or r.get("pipeline", "")]
        if "n_rows" in r:
            name_parts.append(f"{r['n_rows']} rows")
        if "n_features" in r:
            name_parts.append(f"{r['n_features']} features")
        if "n_cols" in r:
            name_parts.append(f"{r['n_cols']} cols")

        formatted.append({
            "name": " - ".join(name_parts),
            "unit": "seconds",
            "value": r["mean_time"],
            "range": f"+/- {r['std_time']:.4f}",
        })

    return formatted


def main():
    parser = argparse.ArgumentParser(description="Run Forge benchmarks")
    parser.add_argument(
        "--suite",
        choices=["all", "generators", "selectors", "pipeline"],
        default="all",
        help="Which benchmark suite to run",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results/output.json",
        help="Output file path for results",
    )
    parser.add_argument(
        "--format",
        choices=["json", "github"],
        default="github",
        help="Output format (json or github-action-benchmark)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Forge Benchmark Suite")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    all_results = []

    if args.suite in ("all", "generators"):
        print("\n--- Generator Benchmarks ---")
        generator_results = run_generator_benchmarks()
        all_results.extend(generator_results)

    if args.suite in ("all", "selectors"):
        print("\n--- Selector Benchmarks ---")
        selector_results = run_selector_benchmarks()
        all_results.extend(selector_results)

    if args.suite in ("all", "pipeline"):
        print("\n--- Pipeline Benchmarks ---")
        pipeline_results = run_pipeline_benchmarks()
        all_results.extend(pipeline_results)
        compare_results(pipeline_results)

    # Format and save results
    if args.format == "github":
        output_data = format_for_github_benchmark(all_results)
    else:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "results": all_results,
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
