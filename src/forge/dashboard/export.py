"""Export lineage graphs to various formats and platforms."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from forge.dashboard.lineage import FeatureLineageGraph
from forge.exceptions import MissingDependencyError

logger = logging.getLogger(__name__)


class HTMLReportExporter:
    """Export feature lineage as a self-contained HTML report.

    Generates an interactive HTML page with a DAG visualization,
    importance bar chart, and summary statistics.

    Args:
        title: Report title.
        include_importance: Whether to include importance chart.
        include_table: Whether to include feature detail table.

    Example:
        >>> exporter = HTMLReportExporter(title="Model Features")
        >>> exporter.export(graph, "report.html")
    """

    def __init__(
        self,
        title: str = "Feature Lineage Report",
        include_importance: bool = True,
        include_table: bool = True,
    ) -> None:
        self.title = title
        self.include_importance = include_importance
        self.include_table = include_table

    def export(self, graph: FeatureLineageGraph, path: str | Path) -> Path:
        """Export the graph to an HTML file.

        Args:
            graph: Feature lineage graph.
            path: Output file path.

        Returns:
            Path to the generated HTML file.
        """
        path = Path(path)
        html = self._render(graph)
        path.write_text(html, encoding="utf-8")
        logger.info("Exported HTML report to %s", path)
        return path

    def _render(self, graph: FeatureLineageGraph) -> str:
        nodes_json = json.dumps([n.to_dict() for n in graph.nodes])
        edges_json = json.dumps([e.to_dict() for e in graph.edges])
        ranking = graph.get_importance_ranking()

        importance_html = ""
        if self.include_importance and ranking:
            bars = []
            max_imp = max(imp for _, imp in ranking) if ranking else 1.0
            for name, imp in ranking[:20]:
                pct = (imp / max_imp) * 100 if max_imp > 0 else 0
                bars.append(
                    f'<div class="bar-row">'
                    f'<span class="bar-label">{name}</span>'
                    f'<div class="bar" style="width:{pct:.0f}%">{imp:.4f}</div>'
                    f'</div>'
                )
            importance_html = '<h2>Feature Importance</h2>\n' + "\n".join(bars)

        table_html = ""
        if self.include_table:
            df = graph.to_dataframe()
            if not df.empty:
                table_html = "<h2>Feature Details</h2>\n" + df.to_html(
                    index=False, classes="feature-table"
                )

        return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{self.title}</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 2em; color: #333; }}
h1 {{ color: #1a1a2e; }}
.summary {{ background: #f0f0f5; padding: 1em; border-radius: 8px; }}
.bar-row {{ display: flex; align-items: center; margin: 4px 0; }}
.bar-label {{ width: 200px; font-size: 0.9em; }}
.bar {{ background: #4361ee; color: white; padding: 2px 8px; border-radius: 4px;
        font-size: 0.8em; min-width: 40px; }}
.feature-table {{ border-collapse: collapse; width: 100%; margin-top: 1em; }}
.feature-table th, .feature-table td {{
    border: 1px solid #ddd; padding: 8px; text-align: left;
}}
.feature-table th {{ background: #4361ee; color: white; }}
.feature-table tr:nth-child(even) {{ background: #f9f9f9; }}
</style>
</head>
<body>
<h1>{self.title}</h1>
<div class="summary"><pre>{graph.summary()}</pre></div>
{importance_html}
{table_html}
<script>
const nodes = {nodes_json};
const edges = {edges_json};
</script>
</body>
</html>"""


class MLflowExporter:
    """Export lineage graph as MLflow artifacts.

    Logs the lineage graph, importance data, and summary
    as artifacts to the active MLflow run.

    Example:
        >>> exporter = MLflowExporter()
        >>> exporter.export(graph)  # Logs to current MLflow run
    """

    def export(
        self,
        graph: FeatureLineageGraph,
        run_id: str | None = None,
        artifact_path: str = "feature_lineage",
    ) -> None:
        """Export lineage to MLflow.

        Args:
            graph: Feature lineage graph.
            run_id: MLflow run ID. None for the active run.
            artifact_path: Artifact sub-path within the run.
        """
        try:
            import mlflow
        except ImportError:
            raise MissingDependencyError("mlflow", "MLflow lineage export")

        graph_dict = graph.to_dict()
        ranking = graph.get_importance_ranking()

        if run_id:
            client = mlflow.tracking.MlflowClient()
            # Log as JSON artifact
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                graph_path = Path(tmpdir) / "lineage_graph.json"
                graph_path.write_text(json.dumps(graph_dict, indent=2))
                client.log_artifact(run_id, str(graph_path), artifact_path)
        else:
            mlflow.log_dict(graph_dict, f"{artifact_path}/lineage_graph.json")
            for name, imp in ranking:
                mlflow.log_metric(f"feature_importance_{name}", imp)

        logger.info("Exported lineage to MLflow (artifact_path=%s)", artifact_path)


class WandbExporter:
    """Export lineage graph to Weights & Biases.

    Logs the lineage as a W&B Table and importance as a bar chart.

    Example:
        >>> exporter = WandbExporter()
        >>> exporter.export(graph)
    """

    def export(
        self,
        graph: FeatureLineageGraph,
        project: str | None = None,
        run_name: str | None = None,
    ) -> None:
        """Export lineage to W&B.

        Args:
            graph: Feature lineage graph.
            project: W&B project name.
            run_name: W&B run name.
        """
        try:
            import wandb
        except ImportError:
            raise MissingDependencyError("wandb", "Weights & Biases lineage export")

        df = graph.to_dataframe()
        ranking = graph.get_importance_ranking()

        run = wandb.run
        if run is None:
            kwargs: dict[str, Any] = {}
            if project:
                kwargs["project"] = project
            if run_name:
                kwargs["name"] = run_name
            run = wandb.init(**kwargs)

        run.log({"feature_lineage": wandb.Table(dataframe=df)})

        for name, imp in ranking:
            run.log({f"importance/{name}": imp})

        run.log({"lineage_summary": graph.summary()})
        logger.info("Exported lineage to Weights & Biases")
