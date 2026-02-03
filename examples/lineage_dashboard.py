#!/usr/bin/env python
"""Feature lineage dashboard example for Forge.

This example demonstrates FeatureLineageGraph for tracking
feature provenance and HTMLReportExporter for generating
interactive lineage reports.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from forge.dashboard import FeatureLineageGraph, HTMLReportExporter


def example_build_lineage_graph():
    """Example: Build a feature lineage DAG manually."""
    print("=" * 60)
    print("Example: Build Feature Lineage Graph")
    print("=" * 60)

    graph = FeatureLineageGraph()

    # Add source columns (raw input data)
    for col in ["age", "income", "region", "credit_score"]:
        graph.add_source(col)

    # Add generated features with their source lineage
    generated = [
        ("log_income", ["income"], "LogTransformer", "log1p(income)", 0.85),
        ("income_per_age", ["income", "age"], "InteractionGen", "income / age", 0.72),
        ("region_encoded", ["region"], "TargetEncoder", "target_encode(region)", 0.45),
        ("credit_bin", ["credit_score"], "BinningTransformer", "quantile_bin", 0.63),
        ("age_squared", ["age"], "PolynomialGen", "age ** 2", 0.38),
    ]
    for name, srcs, gen, transf, imp in generated:
        graph.add_generated(name, sources=srcs, generator=gen,
                            transformation=transf, importance=imp)

    # Mark features as selected or dropped
    graph.add_selected("log_income", importance=0.85)
    graph.add_selected("income_per_age", importance=0.72)
    graph.add_selected("credit_bin", importance=0.63)
    graph.add_dropped("age_squared", reason="Low importance (0.38 < threshold 0.40)")
    graph.add_dropped("region_encoded", reason="High correlation with log_income")

    # Print summary
    print(f"\n{graph.summary()}")

    # Trace lineage
    print(f"\nSources for 'income_per_age': {graph.get_sources('income_per_age')}")
    print(f"Features derived from 'income': {graph.get_derived('income')}")

    # Importance ranking
    print(f"\nImportance ranking:")
    for name, imp in graph.get_importance_ranking():
        print(f"  {name:25s} {imp:.4f}")

    return graph


def example_lineage_dataframe():
    """Example: Convert lineage graph to a DataFrame for analysis."""
    print("\n" + "=" * 60)
    print("Example: Lineage as DataFrame")
    print("=" * 60)

    graph = FeatureLineageGraph()

    # Build a small pipeline lineage
    sources = ["price", "volume", "timestamp"]
    for s in sources:
        graph.add_source(s)

    generated = [
        ("log_price", ["price"], "LogTransformer", 0.92),
        ("price_sma_20", ["price"], "RollingAggregator", 0.88),
        ("volume_zscore", ["volume"], "ZScoreNormalizer", 0.65),
        ("hour_of_day", ["timestamp"], "TemporalExtractor", 0.43),
        ("price_x_volume", ["price", "volume"], "InteractionGenerator", 0.78),
    ]
    for name, srcs, gen, imp in generated:
        graph.add_generated(name, sources=srcs, generator=gen, importance=imp)

    # Convert to DataFrame
    df = graph.to_dataframe()
    print(f"\nLineage DataFrame ({df.shape[0]} features):")
    print(df.to_string(index=False))

    # Serialize to dict
    graph_dict = graph.to_dict()
    print(f"\nSerialized graph has {len(graph_dict['nodes'])} nodes and "
          f"{len(graph_dict['edges'])} edges")


def example_html_report():
    """Example: Export lineage as an interactive HTML report."""
    print("\n" + "=" * 60)
    print("Example: HTML Report Export")
    print("=" * 60)

    graph = FeatureLineageGraph()
    for col in ["age", "income", "debt", "credit_score", "region"]:
        graph.add_source(col)

    features_data = [
        ("log_income", ["income"], "LogTransform", 0.91),
        ("debt_to_income", ["debt", "income"], "RatioGenerator", 0.87),
        ("credit_bin", ["credit_score"], "BinningTransform", 0.75),
        ("income_x_credit", ["income", "credit_score"], "Interaction", 0.69),
        ("age_group", ["age"], "AgeBinning", 0.52),
        ("region_target_enc", ["region"], "TargetEncoder", 0.48),
    ]
    for name, srcs, gen, imp in features_data:
        graph.add_generated(name, sources=srcs, generator=gen, importance=imp)
        graph.add_selected(name, importance=imp)

    # Export to HTML
    exporter = HTMLReportExporter(
        title="Credit Risk Model - Feature Lineage",
        include_importance=True,
        include_table=True,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "lineage_report.html"
        result_path = exporter.export(graph, output_path)
        print(f"\nExported HTML report to: {result_path}")

        # Show report size
        size_kb = result_path.stat().st_size / 1024
        print(f"Report size: {size_kb:.1f} KB")

        # Verify HTML content
        html = result_path.read_text()
        print(f"\nReport contains: title={'Credit Risk' in html}, "
              f"table={'feature-table' in html}, bars={'bar-row' in html}")


if __name__ == "__main__":
    example_build_lineage_graph()
    example_lineage_dataframe()
    example_html_report()

    print("\n" + "=" * 60)
    print("All lineage dashboard examples completed!")
    print("=" * 60)
