"""Gradio web UI for DRIFT."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("drift")

# Global state for loaded model/session
_state: dict[str, Any] = {
    "model": None,
    "session": None,
    "steerer": None,
    "monitor": None,
    "axis": None,
}


def create_app():
    """Create and return the Gradio Blocks app."""
    import gradio as gr
    import plotly.graph_objects as go

    from ..axes import AxisManager
    from ..config import DriftConfig, ModelConfig, MonitorConfig, SteeringConfig
    from ..models import MODEL_CONFIGS, get_gpu_info
    from ..presets import PresetManager

    preset_mgr = PresetManager()
    axis_mgr = AxisManager()

    # ── Helper functions ──────────────────────────────────────────────

    def load_model(model_id: str, quantise: bool) -> str:
        """Load a model into global state."""
        from ..models import DriftModel

        try:
            cfg = ModelConfig(model_id=model_id, quantise=quantise)
            _state["model"] = DriftModel.load(cfg)
            _state["session"] = None
            _state["steerer"] = None
            _state["monitor"] = None
            return f"Model loaded: {model_id}"
        except Exception as e:
            return f"Error loading model: {e}"

    def load_axis_fn(model_id: str) -> str:
        """Load an axis for the current model — check cache first."""
        # Try cached axes first (match by model name)
        model_name = Path(model_id).name  # e.g. "Qwen2.5-7B-Instruct"
        for cached in axis_mgr.list_cached():
            cached_path = Path(cached.get("path", ""))
            if model_name in cached_path.name and not cached_path.name.endswith("_test.pt"):
                try:
                    axis = axis_mgr.load_from_file(cached_path)
                    _state["axis"] = axis
                    return f"Axis loaded from cache: {cached_path.name}"
                except Exception:
                    pass

        # Fall back to HuggingFace download
        try:
            axis = axis_mgr.load_from_huggingface(model_id)
            _state["axis"] = axis
            return f"Axis loaded for {model_id}"
        except Exception as e:
            return f"Error loading axis: {e}"

    def start_session(steering: float, capping: bool, preset_name: str | None) -> str:
        """Initialize a chat session."""
        from ..chat import DriftSession
        from ..monitor import DriftMonitor
        from ..steering import DriftSteerer

        if _state["model"] is None:
            return "Load a model first."

        model = _state["model"]

        cfg = DriftConfig(
            model=model.config,
            steering=SteeringConfig(coefficient=steering, capping_enabled=capping),
            monitor=MonitorConfig(track_projections=True),
        )

        steerer = None
        monitor = None
        if _state["axis"] is not None:
            steerer = DriftSteerer(model, _state["axis"])
            steerer.set_coefficient(steering)
            monitor = DriftMonitor(model, _state["axis"], cfg.monitor)

        session = DriftSession(model, cfg, steerer, monitor)
        _state["session"] = session
        _state["steerer"] = steerer
        _state["monitor"] = monitor

        # Apply preset if selected
        if preset_name and preset_name != "None":
            try:
                preset = preset_mgr.get(preset_name)
                session.set_system_prompt(preset.system_prompt)
                if preset.suggested_steering is not None:
                    session.adjust_steering(preset.suggested_steering)
            except KeyError:
                pass

        return f"Session started (steering={steering:+.1f})"

    def chat_respond(message: str, history: list[dict]) -> tuple:
        """Handle a chat message."""
        if _state["session"] is None:
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "Please start a session first."},
            ]
            return history, "", create_empty_chart(), "N/A", "N/A"

        result = _state["session"].send_message(message)
        response = result["response"]

        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response},
        ]

        # Update trajectory chart
        chart = create_trajectory_chart()

        proj_str = f"{result['projection']:.3f}" if result.get("projection") is not None else "N/A"
        drift_str = "DRIFT DETECTED" if result.get("drift_detected") else "OK"

        return history, "", chart, proj_str, drift_str

    def update_steering(value: float) -> str:
        """Update steering coefficient mid-session."""
        if _state["session"]:
            _state["session"].adjust_steering(value)
            return f"Steering: {value:+.1f}"
        return "No active session"

    def create_trajectory_chart() -> Any:
        """Create a Plotly trajectory chart."""
        if _state["monitor"] is None or not _state["monitor"].snapshots:
            return create_empty_chart()

        snapshots = _state["monitor"].snapshots
        turns = [s.turn_index + 1 for s in snapshots]
        projections = [s.projection for s in snapshots]
        threshold = _state["monitor"].config.alert_threshold

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=turns, y=projections,
            mode="lines+markers",
            name="Projection",
            line=dict(color="#58a6ff"),
            marker=dict(
                color=["#da3633" if s.drift_detected else "#3fb950" for s in snapshots],
                size=8,
            ),
        ))
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="#da3633",
            annotation_text="Threshold",
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            xaxis_title="Turn",
            yaxis_title="Projection",
            height=300,
            margin=dict(l=40, r=20, t=20, b=40),
        )
        return fig

    def create_empty_chart() -> Any:
        """Create an empty placeholder chart."""
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            xaxis_title="Turn",
            yaxis_title="Projection",
            height=300,
            margin=dict(l=40, r=20, t=20, b=40),
            annotations=[dict(
                text="No data yet", showarrow=False,
                xref="paper", yref="paper", x=0.5, y=0.5,
                font=dict(size=14, color="#8b949e"),
            )],
        )
        return fig

    def export_session(format_choice: str) -> str:
        """Export the current session."""
        if _state["session"] is None:
            return "No active session to export."
        ext = {"JSON": "json", "CSV": "csv", "HTML": "html"}[format_choice]
        path = Path(f"drift_export.{ext}")
        _state["session"].export(path, ext)
        return f"Exported to {path}"

    def get_gpu_info_text() -> str:
        """Format GPU info for display."""
        info = get_gpu_info()
        if info["cuda_available"]:
            lines = []
            for dev in info["devices"]:
                lines.append(
                    f"[{dev['index']}] {dev['name']} — "
                    f"{dev['total_gb']}GB total, {dev['free_gb']}GB free"
                )
            return "\n".join(lines)
        return info.get("note", "No CUDA GPU detected")

    def scan_payloads(manifest_path: str, steering_range: str, progress=gr.Progress()) -> str:
        """Run a SPICE scan (simplified for web UI)."""
        if _state["model"] is None:
            return "Load a model first."

        from ..spice_bridge import SpiceBridge

        bridge = SpiceBridge()
        path = Path(manifest_path)

        if path.suffix == ".json":
            payloads = bridge.load_manifest(path)
        else:
            payloads = bridge.load_csv(path)

        coefficients = [float(c.strip()) for c in steering_range.split(",")]

        # Ensure session exists
        if _state["session"] is None:
            start_session(0.0, False, None)

        results = bridge.run_scan(_state["session"], payloads, coefficients)

        from ..report import SpiceScanReport

        report = SpiceScanReport(results, model_id=_state["model"].config.model_id, coefficients=coefficients)
        out_dir = Path("drift_scan_results")
        out_dir.mkdir(exist_ok=True)
        report.write_json(out_dir / "scan_results.json")
        report.write_html(out_dir / "scan_report.html")

        return f"Scan complete: {len(results)} tests. Results in {out_dir}/"

    def compute_axis_fn(model_id: str, num_roles: int, progress=gr.Progress()) -> str:
        """Compute an axis for a model."""
        if _state["model"] is None:
            return "Load the model first."

        axis = axis_mgr.compute_axis(_state["model"], num_roles=num_roles)
        out_path = axis_mgr.cache_dir / f"{model_id.replace('/', '_')}_axis.pt"
        axis_mgr.save_axis(axis, out_path, metadata={"model_id": model_id, "num_roles": num_roles})
        _state["axis"] = axis
        return f"Axis computed and saved to {out_path}"

    def list_cached_axes() -> str:
        """List cached axes."""
        cached = axis_mgr.list_cached()
        if not cached:
            return "No cached axes."
        return "\n".join(f"- {a.get('model_id', a.get('file', '?'))}" for a in cached)

    # ── Build the UI ──────────────────────────────────────────────────

    # Include local models if they exist
    import glob
    local_models = sorted(glob.glob("D:/models/*/"))
    model_choices = [p.rstrip("/\\") for p in local_models] + list(MODEL_CONFIGS.keys())
    preset_choices = ["None"] + preset_mgr.list_names()

    _theme = gr.themes.Base(primary_hue="blue", neutral_hue="slate")
    _css = ".contain { max-width: 1200px; margin: auto; }"

    with gr.Blocks(title="DRIFT") as app:
        app.theme = _theme
        app.css = _css
        gr.Markdown("# DRIFT — Directed Residual Intervention for Functional Testing")

        with gr.Row():
            # ── Left Sidebar ──
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### Model")
                model_dropdown = gr.Dropdown(
                    choices=model_choices,
                    value=model_choices[0] if model_choices else "",
                    label="Model",
                    allow_custom_value=True,
                )
                quantise_toggle = gr.Checkbox(value=True, label="4-bit Quantise")
                load_btn = gr.Button("Load Model", variant="primary")
                load_status = gr.Textbox(label="Status", interactive=False, lines=1)

                gr.Markdown("### Steering")
                steering_slider = gr.Slider(-5.0, 5.0, value=0.0, step=0.1, label="Coefficient")
                capping_toggle = gr.Checkbox(value=False, label="Activation Capping")
                steer_status = gr.Textbox(label="Steering Status", interactive=False, lines=1)

                gr.Markdown("### Preset")
                preset_dropdown = gr.Dropdown(choices=preset_choices, value="None", label="Preset")
                start_btn = gr.Button("Start Session", variant="secondary")
                session_status = gr.Textbox(label="Session", interactive=False, lines=1)

                gr.Markdown("### Monitor")
                projection_display = gr.Textbox(label="Current Projection", interactive=False)
                drift_display = gr.Textbox(label="Drift Status", interactive=False)

            # ── Main Content ──
            with gr.Column(scale=3):
                with gr.Tabs():
                    # ── Chat Tab ──
                    with gr.TabItem("Chat"):
                        chatbot = gr.Chatbot(label="Conversation", height=400)
                        trajectory_chart = gr.Plot(label="Drift Trajectory", value=create_empty_chart())
                        with gr.Row():
                            msg_input = gr.Textbox(
                                label="Message",
                                placeholder="Type a message...",
                                scale=5,
                            )
                            send_btn = gr.Button("Send", variant="primary", scale=1)
                        with gr.Row():
                            export_fmt = gr.Dropdown(["JSON", "CSV", "HTML"], value="JSON", label="Format")
                            export_btn = gr.Button("Export")
                            export_status = gr.Textbox(label="Export Status", interactive=False, scale=2)

                    # ── Scan Tab ──
                    with gr.TabItem("Scan"):
                        gr.Markdown("### SPICE Payload Scan")
                        manifest_input = gr.Textbox(
                            label="Manifest/CSV Path",
                            placeholder="/path/to/manifest.json",
                        )
                        scan_range = gr.Textbox(
                            label="Steering Range",
                            value="-3,-1,0,1,3",
                        )
                        scan_btn = gr.Button("Run Scan", variant="primary")
                        scan_status = gr.Textbox(label="Scan Status", interactive=False, lines=3)

                    # ── Compute Tab ──
                    with gr.TabItem("Compute"):
                        gr.Markdown("### Compute Steering Axis")
                        compute_model = gr.Dropdown(
                            choices=model_choices,
                            value=model_choices[0] if model_choices else "",
                            label="Model",
                            allow_custom_value=True,
                        )
                        num_roles_slider = gr.Slider(10, 275, value=50, step=5, label="Number of Roles")
                        compute_btn = gr.Button("Compute Axis", variant="primary")
                        compute_status = gr.Textbox(label="Status", interactive=False, lines=2)

                    # ── Axes Tab ──
                    with gr.TabItem("Axes"):
                        gr.Markdown("### Cached Axes")
                        axes_display = gr.Textbox(label="Axes", interactive=False, lines=6)
                        refresh_axes_btn = gr.Button("Refresh")
                        load_axis_btn = gr.Button("Load Axis for Current Model")
                        axis_status = gr.Textbox(label="Status", interactive=False)
                        gpu_info = gr.Textbox(label="GPU Info", value=get_gpu_info_text(), interactive=False, lines=3)

        # ── Event handlers ────────────────────────────────────────────

        load_btn.click(load_model, [model_dropdown, quantise_toggle], [load_status])

        steering_slider.release(update_steering, [steering_slider], [steer_status])

        start_btn.click(
            start_session,
            [steering_slider, capping_toggle, preset_dropdown],
            [session_status],
        )

        send_btn.click(
            chat_respond,
            [msg_input, chatbot],
            [chatbot, msg_input, trajectory_chart, projection_display, drift_display],
        )
        msg_input.submit(
            chat_respond,
            [msg_input, chatbot],
            [chatbot, msg_input, trajectory_chart, projection_display, drift_display],
        )

        export_btn.click(export_session, [export_fmt], [export_status])
        scan_btn.click(scan_payloads, [manifest_input, scan_range], [scan_status])
        compute_btn.click(compute_axis_fn, [compute_model, num_roles_slider], [compute_status])
        refresh_axes_btn.click(list_cached_axes, [], [axes_display])
        load_axis_btn.click(load_axis_fn, [model_dropdown], [axis_status])

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch()
