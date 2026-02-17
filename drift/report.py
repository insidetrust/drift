"""JSON/CSV/HTML export for drift sessions and SPICE scans."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("drift")


class DriftReport:
    """Export a drift chat session to JSON/CSV/HTML."""

    def __init__(
        self,
        model_id: str,
        turns: list[dict[str, Any]],
        trajectory: list[dict[str, Any]],
        summary: dict[str, Any],
    ):
        self.model_id = model_id
        self.turns = turns
        self.trajectory = trajectory
        self.summary = summary

    @classmethod
    def from_session(cls, session: Any) -> "DriftReport":
        """Build a report from a DriftSession."""
        turns = [
            {
                "index": r.index,
                "role": r.role,
                "content": r.content,
                "projection": r.projection,
                "drift_detected": r.drift_detected,
                "steering_coefficient": r.steering_coefficient,
                "generation_time": r.generation_time,
            }
            for r in session.history
        ]
        trajectory = session.monitor.get_trajectory() if session.monitor else []
        summary = session.monitor.get_summary() if session.monitor else {}

        return cls(
            model_id=session.config.model.model_id,
            turns=turns,
            trajectory=trajectory,
            summary=summary,
        )

    def write_json(self, path: Path) -> None:
        """Write report as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model_id": self.model_id,
            "summary": self.summary,
            "trajectory": self.trajectory,
            "turns": self.turns,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Report written to %s", path)

    def write_csv(self, path: Path) -> None:
        """Write turn data as CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.turns:
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.turns[0].keys())
            writer.writeheader()
            writer.writerows(self.turns)
        logger.info("CSV written to %s", path)

    def write_html(self, path: Path) -> None:
        """Write an HTML report with embedded Chart.js trajectory chart."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare chart data
        labels = [str(t["turn"]) for t in self.trajectory] if self.trajectory else []
        projections = [t["projection"] for t in self.trajectory] if self.trajectory else []
        threshold = self.summary.get("threshold", 0.3)

        # Build conversation HTML
        conv_html = ""
        for turn in self.turns:
            role_class = "user" if turn["role"] == "user" else "assistant"
            proj_badge = ""
            if turn.get("projection") is not None:
                drift_class = "drift" if turn.get("drift_detected") else "ok"
                proj_badge = f' <span class="badge {drift_class}">{turn["projection"]:.3f}</span>'
            content_escaped = (
                turn["content"]
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br>")
            )
            conv_html += f'<div class="turn {role_class}"><strong>{turn["role"].title()}</strong>{proj_badge}<p>{content_escaped}</p></div>\n'

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DRIFT Report — {self.model_id}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 20px; background: #0d1117; color: #c9d1d9; }}
    .container {{ max-width: 900px; margin: 0 auto; }}
    h1 {{ color: #58a6ff; }}
    h2 {{ color: #8b949e; border-bottom: 1px solid #30363d; padding-bottom: 8px; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 16px 0; }}
    .stat {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; text-align: center; }}
    .stat .value {{ font-size: 1.5em; font-weight: bold; color: #58a6ff; }}
    .stat .label {{ font-size: 0.85em; color: #8b949e; }}
    .chart-container {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin: 16px 0; }}
    .turn {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; margin: 8px 0; }}
    .turn.user {{ border-left: 3px solid #3fb950; }}
    .turn.assistant {{ border-left: 3px solid #58a6ff; }}
    .badge {{ font-size: 0.75em; padding: 2px 6px; border-radius: 4px; margin-left: 8px; }}
    .badge.ok {{ background: #238636; color: white; }}
    .badge.drift {{ background: #da3633; color: white; }}
    p {{ margin: 8px 0 0 0; }}
</style>
</head>
<body>
<div class="container">
    <h1>DRIFT Report</h1>
    <p>Model: <strong>{self.model_id}</strong></p>

    <h2>Summary</h2>
    <div class="summary">
        <div class="stat"><div class="value">{self.summary.get('turns', 0)}</div><div class="label">Turns</div></div>
        <div class="stat"><div class="value">{self.summary.get('drift_events', 0)}</div><div class="label">Drift Events</div></div>
        <div class="stat"><div class="value">{self.summary.get('min_projection', 0):.3f}</div><div class="label">Min Projection</div></div>
        <div class="stat"><div class="value">{self.summary.get('mean_projection', 0):.3f}</div><div class="label">Mean Projection</div></div>
    </div>

    <h2>Drift Trajectory</h2>
    <div class="chart-container">
        <canvas id="trajectoryChart"></canvas>
    </div>

    <h2>Conversation</h2>
    {conv_html}
</div>

<script>
const ctx = document.getElementById('trajectoryChart').getContext('2d');
new Chart(ctx, {{
    type: 'line',
    data: {{
        labels: {json.dumps(labels)},
        datasets: [{{
            label: 'Projection',
            data: {json.dumps(projections)},
            borderColor: '#58a6ff',
            backgroundColor: 'rgba(88, 166, 255, 0.1)',
            fill: true,
            tension: 0.3,
        }}, {{
            label: 'Threshold',
            data: {json.dumps([threshold] * len(labels))},
            borderColor: '#da3633',
            borderDash: [5, 5],
            pointRadius: 0,
            fill: false,
        }}]
    }},
    options: {{
        responsive: true,
        scales: {{
            y: {{ title: {{ display: true, text: 'Projection onto Assistant Axis', color: '#8b949e' }}, grid: {{ color: '#30363d' }}, ticks: {{ color: '#8b949e' }} }},
            x: {{ title: {{ display: true, text: 'Turn', color: '#8b949e' }}, grid: {{ color: '#30363d' }}, ticks: {{ color: '#8b949e' }} }}
        }},
        plugins: {{ legend: {{ labels: {{ color: '#c9d1d9' }} }} }}
    }}
}});
</script>
</body>
</html>"""

        with open(path, "w") as f:
            f.write(html)
        logger.info("HTML report written to %s", path)


class SpiceScanReport:
    """Export SPICE scan results."""

    def __init__(
        self,
        results: list[Any],  # list[ScanResult]
        model_id: str = "",
        coefficients: list[float] | None = None,
    ):
        self.results = results
        self.model_id = model_id
        self.coefficients = coefficients or []

    def write_json(self, path: Path) -> None:
        """Write scan results as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model_id": self.model_id,
            "coefficients": self.coefficients,
            "results": [
                {
                    "payload_id": r.payload_id,
                    "payload_category": r.payload_category,
                    "coefficient": r.coefficient,
                    "compliance_score": r.compliance_score,
                    "refused": r.refused,
                    "projection": r.projection,
                    "drift_detected": r.drift_detected,
                    "response": r.response[:500],
                }
                for r in self.results
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def write_csv(self, path: Path) -> None:
        """Write scan results as CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.results:
            return
        fieldnames = [
            "payload_id", "payload_category", "coefficient",
            "compliance_score", "refused", "projection", "drift_detected",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.results:
                writer.writerow({
                    "payload_id": r.payload_id,
                    "payload_category": r.payload_category,
                    "coefficient": r.coefficient,
                    "compliance_score": r.compliance_score,
                    "refused": r.refused,
                    "projection": r.projection,
                    "drift_detected": r.drift_detected,
                })

    def write_html(self, path: Path) -> None:
        """Write an HTML report with compliance heatmap."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Build heatmap data: category × coefficient → mean compliance
        categories = sorted(set(r.payload_category for r in self.results if r.payload_category))
        if not categories:
            categories = ["all"]

        coefficients = sorted(set(r.coefficient for r in self.results))

        heatmap_data = {}
        for cat in categories:
            heatmap_data[cat] = {}
            for coeff in coefficients:
                matching = [
                    r for r in self.results
                    if (r.payload_category == cat or cat == "all") and r.coefficient == coeff
                ]
                if matching:
                    heatmap_data[cat][coeff] = sum(r.compliance_score for r in matching) / len(matching)
                else:
                    heatmap_data[cat][coeff] = 0.0

        # Build table rows
        table_rows = ""
        for cat in categories:
            cells = f"<td><strong>{cat}</strong></td>"
            for coeff in coefficients:
                val = heatmap_data[cat].get(coeff, 0.0)
                # Color: green (0) to red (1)
                r_val = int(val * 255)
                g_val = int((1 - val) * 255)
                cells += f'<td style="background: rgba({r_val}, {g_val}, 0, 0.3); text-align: center;">{val:.2f}</td>'
            table_rows += f"<tr>{cells}</tr>\n"

        coeff_headers = "".join(f"<th>{c:+.1f}</th>" for c in coefficients)

        # Overall stats
        total = len(self.results)
        refused = sum(1 for r in self.results if r.refused)
        mean_compliance = sum(r.compliance_score for r in self.results) / max(total, 1)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>DRIFT SPICE Scan — {self.model_id}</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 20px; background: #0d1117; color: #c9d1d9; }}
    .container {{ max-width: 1000px; margin: 0 auto; }}
    h1 {{ color: #58a6ff; }}
    h2 {{ color: #8b949e; border-bottom: 1px solid #30363d; padding-bottom: 8px; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 16px 0; }}
    .stat {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; text-align: center; }}
    .stat .value {{ font-size: 1.5em; font-weight: bold; color: #58a6ff; }}
    .stat .label {{ font-size: 0.85em; color: #8b949e; }}
    table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
    th, td {{ border: 1px solid #30363d; padding: 8px 12px; }}
    th {{ background: #161b22; color: #8b949e; }}
    tr:hover {{ background: rgba(88, 166, 255, 0.05); }}
</style>
</head>
<body>
<div class="container">
    <h1>DRIFT SPICE Scan Report</h1>
    <p>Model: <strong>{self.model_id}</strong></p>

    <h2>Summary</h2>
    <div class="summary">
        <div class="stat"><div class="value">{total}</div><div class="label">Total Tests</div></div>
        <div class="stat"><div class="value">{refused}</div><div class="label">Refused</div></div>
        <div class="stat"><div class="value">{mean_compliance:.2f}</div><div class="label">Mean Compliance</div></div>
        <div class="stat"><div class="value">{len(coefficients)}</div><div class="label">Coefficients</div></div>
    </div>

    <h2>Compliance by Category &times; Steering</h2>
    <p>Higher values indicate greater compliance with payload instructions (less safety).</p>
    <table>
        <thead><tr><th>Category</th>{coeff_headers}</tr></thead>
        <tbody>{table_rows}</tbody>
    </table>
</div>
</body>
</html>"""

        with open(path, "w") as f:
            f.write(html)

    def print_summary(self, console: Any) -> None:
        """Print a summary table to the Rich console."""
        from rich.table import Table

        table = Table(title="Scan Results Summary", show_header=True)
        table.add_column("Coefficient", justify="right", style="yellow")
        table.add_column("Tests", justify="right")
        table.add_column("Refused", justify="right")
        table.add_column("Mean Compliance", justify="right")

        coefficients = sorted(set(r.coefficient for r in self.results))
        for coeff in coefficients:
            matching = [r for r in self.results if r.coefficient == coeff]
            n = len(matching)
            refused = sum(1 for r in matching if r.refused)
            mean_c = sum(r.compliance_score for r in matching) / max(n, 1)
            table.add_row(
                f"{coeff:+.1f}",
                str(n),
                str(refused),
                f"{mean_c:.3f}",
            )

        console.print(table)
