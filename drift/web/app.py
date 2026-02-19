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

# Emoji tags for built-in presets (Neuronpedia-style scenario pills)
PRESET_EMOJI = {
    "therapy_drift": "\U0001f9e0",       # brain
    "roleplay_exploit": "\U0001f3ad",     # performing arts
    "sycophancy": "\U0001f91d",           # handshake
    "authority_compliance": "\U0001f3af",  # dart (shield doesn't render on Windows)
    "meta_reflection": "\U0001f52e",      # crystal ball
    "drift_to_insecure_code": "\U0001f510",  # locked with key
}


def create_app():
    """Create and return the Gradio Blocks app."""
    import gradio as gr
    import plotly.graph_objects as go

    from ..axes import AxisManager
    from ..config import DriftConfig, ModelConfig, MonitorConfig, SteeringConfig
    from ..demos import list_demos, load_demo, save_demo
    from ..models import MODEL_CONFIGS, get_gpu_info
    from ..presets import PresetManager

    preset_mgr = PresetManager()
    axis_mgr = AxisManager()

    # -- Helper functions ----------------------------------------------

    def _try_load_axis(model_id: str) -> str | None:
        """Try to load an axis from cache or HuggingFace. Returns status or None."""
        model_name = Path(model_id).name
        for cached in axis_mgr.list_cached():
            cached_path = Path(cached.get("path", ""))
            if model_name in cached_path.name and not cached_path.name.endswith("_test.pt"):
                try:
                    _state["axis"] = axis_mgr.load_from_file(cached_path)
                    return f"axis: {cached_path.name}"
                except Exception:
                    pass
        try:
            _state["axis"] = axis_mgr.load_from_huggingface(model_id)
            return f"axis: {model_name}"
        except Exception:
            return None

    def load_model(model_id: str, quantise: bool) -> str:
        """Load a model and auto-load its axis."""
        import gc
        from ..models import DriftModel

        # Free existing model VRAM before loading new one
        if _state["model"] is not None:
            _state["session"] = None
            _state["steerer"] = None
            _state["monitor"] = None
            _state["model"] = None
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

        try:
            cfg = ModelConfig(model_id=model_id, quantise=quantise)
            _state["model"] = DriftModel.load(cfg)
            _state["session"] = None
            _state["steerer"] = None
            _state["monitor"] = None

            # Auto-load axis
            axis_status = _try_load_axis(model_id)
            if axis_status:
                return f"\u2713 {Path(model_id).name} + {axis_status}"
            return f"\u2713 {Path(model_id).name} (no axis found \u2014 compute one in Axes tab)"
        except Exception as e:
            err = str(e)
            if "CPU" in err or "GPU RAM" in err or "out of memory" in err.lower():
                return (
                    f"\u2717 GPU memory error — another process may be using VRAM. "
                    f"Kill other Python/drift processes and retry."
                )
            return f"\u2717 Error: {e}"

    def load_axis_fn(model_id: str) -> str:
        """Load an axis for the current model — check cache first."""
        result = _try_load_axis(model_id)
        if result:
            return f"\u2713 Axis loaded: {result}"
        return "\u2717 No axis found in cache or HuggingFace"

    def start_session(steering: float, capping: bool, preset_name: str | None) -> tuple:
        """Initialize a chat session. Returns (session_status, preset_info_md)."""
        from ..chat import DriftSession
        from ..monitor import DriftMonitor
        from ..steering import DriftSteerer

        if _state["model"] is None:
            return "\u2717 Load a model first.", ""

        model = _state["model"]

        # Auto-load axis if not already loaded
        if _state["axis"] is None:
            _try_load_axis(model.config.model_id)

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

        info_md = ""
        if preset_name and preset_name != "None":
            try:
                preset = preset_mgr.get(preset_name)
                session.set_system_prompt(preset.system_prompt)
                if preset.suggested_steering is not None:
                    session.adjust_steering(preset.suggested_steering)
                info_md = _build_preset_info_md(preset)
            except KeyError:
                pass

        return f"\u2713 Session active (steer={steering:+.1f})", info_md

    def _build_preset_info_md(preset) -> str:
        """Build a markdown info block for the active preset."""
        emoji = PRESET_EMOJI.get(preset.name, "\U0001f9ea")
        steer_str = f"{preset.suggested_steering:+.1f}" if preset.suggested_steering is not None else "0"
        steps_str = f"{len(preset.steps)} steps" if preset.steps else "Free chat"
        tags = " ".join(
            f"<span style='background:#334155;color:#60a5fa;padding:2px 8px;"
            f"border-radius:4px;font-size:0.8em'>{t}</span>"
            for t in preset.tags
        ) if preset.tags else ""
        return (
            f"### {emoji} {preset.name.replace('_', ' ').title()}\n"
            f"{preset.description}\n\n"
            f"**Steering:** {steer_str} · **Turns:** {steps_str}\n\n"
            f"{tags}"
        )

    def _projection_color(val: float, threshold: float) -> str:
        """Return a hex colour for the projection value (green/amber/red)."""
        if val > 0:
            return "#22c55e"  # green — safe
        warn_y = threshold * 0.6
        if val > warn_y:
            return "#f59e0b"  # amber — warning
        return "#ef4444"      # red — drifting

    def _build_monitor_html(snapshots, threshold: float) -> str:
        """Build rich HTML for the monitor panel below the chart."""
        if not snapshots:
            return ""
        projs = [s.projection for s in snapshots]
        n = len(projs)
        latest = projs[-1]
        drift_events = sum(1 for s in snapshots if s.drift_detected)
        mean_p = sum(projs) / n
        min_p = min(projs)
        velocity = projs[-1] - projs[-2] if n >= 2 else 0.0

        col = _projection_color(latest, threshold)
        vel_col = "#ef4444" if velocity < -0.1 else "#f59e0b" if velocity < 0 else "#22c55e"
        drift_col = "#ef4444" if drift_events > 0 else "#22c55e"

        return (
            f"<div style='display:flex; gap:16px; flex-wrap:wrap; font-family:JetBrains Mono,monospace;'>"
            f"<div style='flex:1; min-width:100px; text-align:center; padding:8px; "
            f"background:#0f172a; border-radius:6px; border:1px solid #334155;'>"
            f"<div style='color:#94a3b8; font-size:0.7em; text-transform:uppercase; letter-spacing:0.05em;'>Projection</div>"
            f"<div style='color:{col}; font-size:1.6em; font-weight:700;'>{latest:.3f}</div></div>"
            f"<div style='flex:1; min-width:100px; text-align:center; padding:8px; "
            f"background:#0f172a; border-radius:6px; border:1px solid #334155;'>"
            f"<div style='color:#94a3b8; font-size:0.7em; text-transform:uppercase; letter-spacing:0.05em;'>Velocity</div>"
            f"<div style='color:{vel_col}; font-size:1.6em; font-weight:700;'>{velocity:+.3f}</div></div>"
            f"<div style='flex:1; min-width:100px; text-align:center; padding:8px; "
            f"background:#0f172a; border-radius:6px; border:1px solid #334155;'>"
            f"<div style='color:#94a3b8; font-size:0.7em; text-transform:uppercase; letter-spacing:0.05em;'>Mean</div>"
            f"<div style='color:#94a3b8; font-size:1.6em; font-weight:700;'>{mean_p:.3f}</div></div>"
            f"<div style='flex:1; min-width:100px; text-align:center; padding:8px; "
            f"background:#0f172a; border-radius:6px; border:1px solid #334155;'>"
            f"<div style='color:#94a3b8; font-size:0.7em; text-transform:uppercase; letter-spacing:0.05em;'>Drift Events</div>"
            f"<div style='color:{drift_col}; font-size:1.6em; font-weight:700;'>{drift_events}/{n}</div></div>"
            f"</div>"
        )

    def chat_respond(message: str, history: list[dict]) -> tuple:
        """Handle a chat message. Returns (history, msg_clear, chart, monitor_html)."""
        if _state["session"] is None:
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "Start a session first using the sidebar controls."},
            ]
            return history, "", create_empty_chart(), ""

        result = _state["session"].send_message(message)
        response = result["response"]

        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response},
        ]

        chart = create_trajectory_chart()

        # Build rich monitor stats
        monitor_html = ""
        if _state["monitor"] and _state["monitor"].snapshots:
            threshold = _state["monitor"].config.alert_threshold
            monitor_html = _build_monitor_html(_state["monitor"].snapshots, threshold)

        return history, "", chart, monitor_html

    def update_steering(value: float) -> str:
        """Update steering coefficient mid-session."""
        if _state["session"]:
            _state["session"].adjust_steering(value)
            return f"Coefficient: {value:+.1f}"
        return "No active session"

    def _chart_colours():
        """Consistent chart colour palette."""
        return {
            "bg": "#0f172a",
            "plot": "#1e293b",
            "line": "#3b82f6",
            "safe": "#22c55e",
            "warn": "#f59e0b",
            "drift": "#ef4444",
            "grid": "#334155",
            "text": "#94a3b8",
            "threshold": "#ef4444",
            "fill": "rgba(59,130,246,0.1)",
            # Danger zone bands (assistant-axis style)
            "zone_safe": "rgba(34,197,94,0.07)",
            "zone_warn": "rgba(245,158,11,0.07)",
            "zone_danger": "rgba(239,68,68,0.07)",
        }

    def create_trajectory_chart() -> Any:
        """Create a Plotly trajectory chart with assistant-axis-style danger zones."""
        if _state["monitor"] is None or not _state["monitor"].snapshots:
            return create_empty_chart()

        c = _chart_colours()
        snapshots = _state["monitor"].snapshots
        turns = [s.turn_index + 1 for s in snapshots]
        projections = [s.projection for s in snapshots]
        threshold = _state["monitor"].config.alert_threshold

        fig = go.Figure()

        # Danger zone bands (horizontal coloured regions like assistant-axis)
        y_min = min(projections + [threshold]) - 0.5
        y_max = max(projections + [0]) + 0.5
        warn_y = threshold * 0.6  # warning starts at 60% of threshold
        fig.add_hrect(y0=0, y1=y_max, fillcolor=c["zone_safe"],
                       line_width=0, layer="below")
        fig.add_hrect(y0=warn_y, y1=0, fillcolor=c["zone_warn"],
                       line_width=0, layer="below")
        fig.add_hrect(y0=y_min, y1=warn_y, fillcolor=c["zone_danger"],
                       line_width=0, layer="below")

        # Zone labels on the right edge
        x_max = max(turns) + 0.5
        fig.add_annotation(x=x_max, y=y_max * 0.5, text="SAFE",
                           showarrow=False, font=dict(size=9, color=c["safe"]),
                           xanchor="right", opacity=0.6)
        fig.add_annotation(x=x_max, y=warn_y * 0.5, text="WARN",
                           showarrow=False, font=dict(size=9, color=c["warn"]),
                           xanchor="right", opacity=0.6)
        fig.add_annotation(x=x_max, y=(y_min + warn_y) * 0.5, text="DRIFT",
                           showarrow=False, font=dict(size=9, color=c["drift"]),
                           xanchor="right", opacity=0.6)

        # Fill area under curve
        fig.add_trace(go.Scatter(
            x=turns, y=projections,
            fill="tozeroy",
            fillcolor=c["fill"],
            line=dict(color=c["line"], width=2.5),
            mode="lines+markers",
            name="Projection",
            marker=dict(
                color=[c["drift"] if s.drift_detected else c["safe"] for s in snapshots],
                size=9,
                line=dict(width=1.5, color=c["plot"]),
            ),
            hovertemplate=(
                "<b>Turn %{x}</b><br>"
                "Projection: %{y:.3f}<br>"
                "<extra></extra>"
            ),
        ))

        # Threshold line
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color=c["threshold"],
            opacity=0.5,
            annotation_text="alert threshold",
            annotation_font_color=c["text"],
            annotation_font_size=10,
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=c["bg"],
            plot_bgcolor=c["plot"],
            xaxis=dict(title="Turn", gridcolor=c["grid"], dtick=1),
            yaxis=dict(title="Projection", gridcolor=c["grid"],
                       range=[y_min, y_max]),
            height=240,
            margin=dict(l=50, r=45, t=10, b=40),
            showlegend=False,
            font=dict(color=c["text"], size=11),
            hoverlabel=dict(bgcolor=c["plot"], font_color=c["text"]),
        )
        return fig

    def create_empty_chart() -> Any:
        """Create an empty placeholder chart."""
        c = _chart_colours()
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=c["bg"],
            plot_bgcolor=c["plot"],
            xaxis=dict(title="Turn", gridcolor=c["grid"]),
            yaxis=dict(title="Projection", gridcolor=c["grid"]),
            height=220,
            margin=dict(l=50, r=20, t=10, b=40),
            font=dict(color=c["text"], size=11),
            annotations=[dict(
                text="No data yet \u2014 start a session and send a message",
                showarrow=False,
                xref="paper", yref="paper", x=0.5, y=0.5,
                font=dict(size=13, color=c["text"]),
            )],
        )
        return fig

    def export_session(format_choice: str) -> str:
        """Export the current session."""
        if _state["session"] is None:
            return "\u2717 No active session to export."
        ext = {"JSON": "json", "CSV": "csv", "HTML": "html"}[format_choice]
        path = Path(f"drift_export.{ext}")
        _state["session"].export(path, ext)
        return f"\u2713 Exported to {path}"

    def save_demo_fn(preset_label: str) -> str:
        """Save the current session + monitor state as a demo JSON."""
        if _state["session"] is None:
            return "\u2717 No active session to save."
        preset_key = _resolve_preset(preset_label) if preset_label else None
        if preset_key == "None":
            preset_key = None
        try:
            path = save_demo(_state["session"], _state["monitor"], preset_key)
            return f"\u2713 Demo saved to {path}"
        except Exception as e:
            return f"\u2717 Error saving demo: {e}"

    def list_demos_fn() -> dict:
        """Return updated choices for the demo dropdown."""
        names = list_demos()
        return gr.update(choices=names, value=names[0] if names else None)

    def load_demo_fn(demo_name: str | None) -> tuple:
        """Load a demo and return (chatbot, chart, monitor_html, status).

        Populates the full UI from a saved snapshot — no model required.
        """
        if not demo_name:
            return gr.update(), gr.update(), gr.update(), "\u2717 Select a demo first."

        from ..demos import DEMO_SUFFIX, DEMOS_DIR

        path = DEMOS_DIR / f"{demo_name}{DEMO_SUFFIX}"
        if not path.exists():
            return gr.update(), gr.update(), gr.update(), f"\u2717 Demo file not found: {path}"

        try:
            data = load_demo(path)
        except Exception as e:
            return gr.update(), gr.update(), gr.update(), f"\u2717 Error loading demo: {e}"

        # Build chatbot history (list of message dicts)
        history = []
        for msg in data["messages"]:
            history.append({"role": msg["role"], "content": msg["content"]})

        # Build trajectory chart from snapshots
        snapshots = data["snapshots"]
        threshold = data["threshold"]
        if snapshots:
            chart = _build_demo_chart(snapshots, threshold)
            monitor_cards = _build_demo_monitor_html(snapshots, threshold)
        else:
            chart = create_empty_chart()
            monitor_cards = ""

        preset_str = data.get("preset_name") or "free chat"
        status = (
            f"\u2713 Loaded demo: {demo_name} "
            f"(model={data['model_id']}, preset={preset_str}, "
            f"steer={data['steering_coefficient']:+.1f})"
        )

        return history, chart, monitor_cards, status

    def _build_demo_chart(snapshots: list[dict], threshold: float) -> Any:
        """Build a trajectory chart from serialised snapshot dicts."""
        c = _chart_colours()
        turns = [s["turn_index"] + 1 for s in snapshots]
        projections = [s["projection"] for s in snapshots]

        fig = go.Figure()

        y_min = min(projections + [threshold]) - 0.5
        y_max = max(projections + [0]) + 0.5
        warn_y = threshold * 0.6
        fig.add_hrect(y0=0, y1=y_max, fillcolor=c["zone_safe"], line_width=0, layer="below")
        fig.add_hrect(y0=warn_y, y1=0, fillcolor=c["zone_warn"], line_width=0, layer="below")
        fig.add_hrect(y0=y_min, y1=warn_y, fillcolor=c["zone_danger"], line_width=0, layer="below")

        x_max = max(turns) + 0.5
        fig.add_annotation(x=x_max, y=y_max * 0.5, text="SAFE",
                           showarrow=False, font=dict(size=9, color=c["safe"]),
                           xanchor="right", opacity=0.6)
        fig.add_annotation(x=x_max, y=warn_y * 0.5, text="WARN",
                           showarrow=False, font=dict(size=9, color=c["warn"]),
                           xanchor="right", opacity=0.6)
        fig.add_annotation(x=x_max, y=(y_min + warn_y) * 0.5, text="DRIFT",
                           showarrow=False, font=dict(size=9, color=c["drift"]),
                           xanchor="right", opacity=0.6)

        fig.add_trace(go.Scatter(
            x=turns, y=projections,
            fill="tozeroy", fillcolor=c["fill"],
            line=dict(color=c["line"], width=2.5),
            mode="lines+markers", name="Projection",
            marker=dict(
                color=[c["drift"] if s["drift_detected"] else c["safe"] for s in snapshots],
                size=9, line=dict(width=1.5, color=c["plot"]),
            ),
            hovertemplate="<b>Turn %{x}</b><br>Projection: %{y:.3f}<br><extra></extra>",
        ))

        fig.add_hline(y=threshold, line_dash="dash", line_color=c["threshold"],
                       opacity=0.5, annotation_text="alert threshold",
                       annotation_font_color=c["text"], annotation_font_size=10)

        fig.update_layout(
            template="plotly_dark", paper_bgcolor=c["bg"], plot_bgcolor=c["plot"],
            xaxis=dict(title="Turn", gridcolor=c["grid"], dtick=1),
            yaxis=dict(title="Projection", gridcolor=c["grid"], range=[y_min, y_max]),
            height=240, margin=dict(l=50, r=45, t=10, b=40),
            showlegend=False, font=dict(color=c["text"], size=11),
            hoverlabel=dict(bgcolor=c["plot"], font_color=c["text"]),
        )
        return fig

    def _build_demo_monitor_html(snapshots: list[dict], threshold: float) -> str:
        """Build monitor HTML from serialised snapshot dicts."""
        projs = [s["projection"] for s in snapshots]
        n = len(projs)
        latest = projs[-1]
        drift_events = sum(1 for s in snapshots if s["drift_detected"])
        mean_p = sum(projs) / n
        velocity = projs[-1] - projs[-2] if n >= 2 else 0.0

        col = _projection_color(latest, threshold)
        vel_col = "#ef4444" if velocity < -0.1 else "#f59e0b" if velocity < 0 else "#22c55e"
        drift_col = "#ef4444" if drift_events > 0 else "#22c55e"

        return (
            f"<div style='display:flex; gap:16px; flex-wrap:wrap; font-family:JetBrains Mono,monospace;'>"
            f"<div style='flex:1; min-width:100px; text-align:center; padding:8px; "
            f"background:#0f172a; border-radius:6px; border:1px solid #334155;'>"
            f"<div style='color:#94a3b8; font-size:0.7em; text-transform:uppercase; letter-spacing:0.05em;'>Projection</div>"
            f"<div style='color:{col}; font-size:1.6em; font-weight:700;'>{latest:.3f}</div></div>"
            f"<div style='flex:1; min-width:100px; text-align:center; padding:8px; "
            f"background:#0f172a; border-radius:6px; border:1px solid #334155;'>"
            f"<div style='color:#94a3b8; font-size:0.7em; text-transform:uppercase; letter-spacing:0.05em;'>Velocity</div>"
            f"<div style='color:{vel_col}; font-size:1.6em; font-weight:700;'>{velocity:+.3f}</div></div>"
            f"<div style='flex:1; min-width:100px; text-align:center; padding:8px; "
            f"background:#0f172a; border-radius:6px; border:1px solid #334155;'>"
            f"<div style='color:#94a3b8; font-size:0.7em; text-transform:uppercase; letter-spacing:0.05em;'>Mean</div>"
            f"<div style='color:#94a3b8; font-size:1.6em; font-weight:700;'>{mean_p:.3f}</div></div>"
            f"<div style='flex:1; min-width:100px; text-align:center; padding:8px; "
            f"background:#0f172a; border-radius:6px; border:1px solid #334155;'>"
            f"<div style='color:#94a3b8; font-size:0.7em; text-transform:uppercase; letter-spacing:0.05em;'>Drift Events</div>"
            f"<div style='color:{drift_col}; font-size:1.6em; font-weight:700;'>{drift_events}/{n}</div></div>"
            f"</div>"
        )

    def get_gpu_info_text() -> str:
        """Format GPU info for display."""
        info = get_gpu_info()
        if info["cuda_available"]:
            lines = []
            for dev in info["devices"]:
                lines.append(
                    f"[{dev['index']}] {dev['name']} \u2014 "
                    f"{dev['total_gb']}GB total, {dev['free_gb']}GB free"
                )
            return "\n".join(lines)
        return info.get("note", "No CUDA GPU detected")

    def compute_axis_fn(model_id: str, num_roles: int, progress=gr.Progress()) -> str:
        """Compute an axis for a model."""
        if _state["model"] is None:
            return "\u2717 Load the model first."

        axis = axis_mgr.compute_axis(_state["model"], num_roles=num_roles)
        out_path = axis_mgr.cache_dir / f"{model_id.replace('/', '_')}_axis.pt"
        axis_mgr.save_axis(axis, out_path, metadata={"model_id": model_id, "num_roles": num_roles})
        _state["axis"] = axis
        return f"\u2713 Axis computed and saved to {out_path}"

    def list_cached_axes() -> str:
        """List cached axes."""
        cached = axis_mgr.list_cached()
        if not cached:
            return "No cached axes found."
        return "\n".join(f"\u2022 {a.get('model_id', a.get('file', '?'))}" for a in cached)

    def on_preset_select(preset_name: str) -> tuple:
        """When a preset is selected, show its description and update steering."""
        if not preset_name or preset_name == "None":
            return "", gr.update()
        try:
            preset = preset_mgr.get(preset_name)
            md = _build_preset_info_md(preset)
            steer_val = preset.suggested_steering if preset.suggested_steering is not None else 0.0
            return md, gr.update(value=steer_val)
        except KeyError:
            return "", gr.update()

    # -- Build the UI --------------------------------------------------

    # Discover local models
    import glob
    local_models = sorted(glob.glob("D:/models/*/"))
    model_choices = [p.rstrip("/\\") for p in local_models] + list(MODEL_CONFIGS.keys())

    # Build preset choices with emoji labels
    preset_names = preset_mgr.list_names()
    preset_radio_choices = ["None (free chat)"] + [
        f"{PRESET_EMOJI.get(n, chr(0x1f9ea))} {n.replace('_', ' ').title()}"
        for n in preset_names
    ]
    # Map display labels back to preset keys
    preset_label_to_key = {"None (free chat)": "None"}
    for n in preset_names:
        label = f"{PRESET_EMOJI.get(n, chr(0x1f9ea))} {n.replace('_', ' ').title()}"
        preset_label_to_key[label] = n

    def _resolve_preset(label: str) -> str:
        return preset_label_to_key.get(label, "None")

    # Blue/slate theme
    theme = gr.themes.Default(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    ).set(
        body_background_fill="#0f172a",
        body_background_fill_dark="#0f172a",
        body_text_color="#e2e8f0",
        body_text_color_dark="#e2e8f0",
        block_background_fill="#1e293b",
        block_background_fill_dark="#1e293b",
        block_border_color="#334155",
        block_border_color_dark="#334155",
        block_label_text_color="#94a3b8",
        block_label_text_color_dark="#94a3b8",
        block_title_text_color="#e2e8f0",
        block_title_text_color_dark="#e2e8f0",
        input_background_fill="#0f172a",
        input_background_fill_dark="#0f172a",
        input_border_color="#334155",
        input_border_color_dark="#334155",
        button_primary_background_fill="#2563eb",
        button_primary_background_fill_dark="#2563eb",
        button_primary_background_fill_hover="#1d4ed8",
        button_primary_background_fill_hover_dark="#1d4ed8",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_secondary_background_fill="#334155",
        button_secondary_background_fill_dark="#334155",
        button_secondary_background_fill_hover="#475569",
        button_secondary_background_fill_hover_dark="#475569",
        button_secondary_text_color="#e2e8f0",
        button_secondary_text_color_dark="#e2e8f0",
        border_color_primary="#334155",
        border_color_primary_dark="#334155",
        color_accent="#3b82f6",
        shadow_drop="none",
        shadow_drop_lg="none",
        checkbox_background_color="#0f172a",
        checkbox_background_color_dark="#0f172a",
        checkbox_border_color="#475569",
        checkbox_border_color_dark="#475569",
        slider_color="#3b82f6",
    )

    css = """
    .gradio-container { max-width: 1400px !important; }
    .panel-heading { margin: 0 0 6px 0 !important; color: #60a5fa !important; font-size: 0.8em !important;
        font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.05em !important; }
    footer { display: none !important; }
    .chatbot { border: 1px solid #334155 !important; }
    [data-testid="bot"], [data-testid="bot"] *, [data-testid="user"], [data-testid="user"] * {
        color: #0f172a !important; }
    .bot-row [data-testid="bot"] { background: #e2e8f0 !important; }
    .user-row [data-testid="user"] { background: #dbeafe !important; }
    #send-btn { min-height: 42px !important; }
    .preset-info { background: #1e293b !important; border-radius: 4px !important; }
    .preset-info .prose { padding: 0 !important; background: transparent !important; }
    .monitor-cards { margin-top: -8px !important; }
    /* Radio button labels - dark text on light pill backgrounds */
    .scenario-radio .wrap * { color: #0f172a !important; }
    /* Remove borders on chat message prose spans (Gradio default) */
    .chatbot .prose { border: none !important; }
    """

    with gr.Blocks(title="DRIFT", theme=theme, css=css) as app:
        gr.Markdown(
            "# <span style='color: #60a5fa'>DRIFT</span> "
            "\u2014 Deliberately Realign Inhibitions For Testing\n"
            "<span style='color: #64748b; font-size: 0.85em'>"
            "Activation steering & persona drift monitoring for LLM red-teaming "
            "\u00b7 Based on the "
            "<a href='https://arxiv.org/abs/2601.10387' style='color: #60a5fa' target='_blank'>"
            "Assistant Axis</a> research</span>"
        )

        with gr.Row(equal_height=False):
            # == Left sidebar ==============================================
            with gr.Column(scale=1, min_width=300):

                # Model section
                gr.Markdown("<p class='panel-heading'>Model</p>")
                model_dropdown = gr.Dropdown(
                    choices=model_choices,
                    value=model_choices[0] if model_choices else "",
                    label="Model",
                    allow_custom_value=True,
                )
                quantise_toggle = gr.Checkbox(value=True, label="4-bit Quantise")
                load_btn = gr.Button("Load Model", variant="primary", size="sm")
                load_status = gr.Textbox(
                    label="Status", interactive=False, lines=1,
                    placeholder="No model loaded",
                )

                # Steering section
                gr.Markdown("<p class='panel-heading'>Steering</p>")
                steering_slider = gr.Slider(
                    -5.0, 5.0, value=0.0, step=0.1,
                    label="Coefficient",
                    info="Negative = push away from assistant persona",
                )
                capping_toggle = gr.Checkbox(value=False, label="Activation Capping")

                # Scenario section (Neuronpedia-style radio pills)
                gr.Markdown("<p class='panel-heading'>Scenario</p>")
                preset_radio = gr.Radio(
                    choices=preset_radio_choices,
                    value="None (free chat)",
                    label="Select a red-team scenario",
                    info="Each preset applies a system prompt, steering, and multi-turn attack",
                    elem_classes=["scenario-radio"],
                )
                preset_info_md = gr.Markdown(
                    value="",
                    elem_classes=["preset-info"],
                )
                start_btn = gr.Button("Start Session", variant="primary", size="sm")
                session_status = gr.Textbox(label="Session", interactive=False, lines=1)

                # Monitor section (moved to below chart in main area)

            # == Main content area =========================================
            with gr.Column(scale=3):
                with gr.Tabs():
                    # Chat tab
                    with gr.TabItem("Chat", id="chat"):
                        chatbot = gr.Chatbot(
                            label="Conversation", height=380,
                            placeholder=(
                                "Load a model, select a scenario, and start a session to begin.\n\n"
                                "The drift trajectory chart below tracks how far the model's "
                                "activations shift from the assistant persona on each turn."
                            ),
                        )
                        trajectory_chart = gr.Plot(
                            label="Drift Trajectory",
                            value=create_empty_chart(),
                        )
                        monitor_html = gr.HTML(
                            value="<div style='color:#64748b; text-align:center; padding:8px; "
                                  "font-family:JetBrains Mono,monospace; font-size:0.85em;'>"
                                  "Send a message to see live projection metrics</div>",
                        )
                        with gr.Row():
                            msg_input = gr.Textbox(
                                placeholder="Type a message\u2026",
                                scale=5, lines=1,
                                show_label=False,
                            )
                            send_btn = gr.Button(
                                "Send", variant="primary", scale=1,
                                elem_id="send-btn",
                            )
                        with gr.Accordion("Session", open=False):
                            with gr.Row():
                                export_fmt = gr.Dropdown(
                                    ["JSON", "CSV", "HTML"], value="JSON",
                                    label="Format", scale=1,
                                )
                                export_btn = gr.Button("Export", variant="secondary", scale=1)
                                save_demo_btn = gr.Button(
                                    "\U0001f4be Save Demo", variant="secondary", scale=1,
                                )
                            with gr.Row():
                                demo_dropdown = gr.Dropdown(
                                    choices=list_demos(),
                                    label="Saved Demos", scale=2,
                                    allow_custom_value=False,
                                )
                                refresh_demos_btn = gr.Button(
                                    "\u21bb", variant="secondary", scale=0, min_width=40,
                                )
                                load_demo_btn = gr.Button(
                                    "Load Demo", variant="primary", scale=1,
                                )
                            session_io_status = gr.Textbox(
                                label="Status", interactive=False,
                            )

                    # Compute tab
                    with gr.TabItem("Compute", id="compute"):
                        gr.Markdown("### Compute Steering Axis")
                        gr.Markdown(
                            "<span style='color: #94a3b8'>Generate a new assistant-axis vector "
                            "for the loaded model by probing it with role-play prompts. "
                            "The axis captures the direction in activation space that separates "
                            "'assistant' behaviour from 'base model' behaviour.</span>"
                        )
                        compute_model = gr.Dropdown(
                            choices=model_choices,
                            value=model_choices[0] if model_choices else "",
                            label="Model",
                            allow_custom_value=True,
                        )
                        num_roles_slider = gr.Slider(
                            10, 275, value=50, step=5,
                            label="Number of Roles",
                            info="More roles = better axis, but slower (50 \u2248 2 min, 275 \u2248 30 min)",
                        )
                        compute_btn = gr.Button("Compute Axis", variant="primary")
                        compute_status = gr.Textbox(label="Status", interactive=False, lines=2)

                    # Axes tab
                    with gr.TabItem("Axes", id="axes"):
                        gr.Markdown("### Cached Axes")
                        gr.Markdown(
                            "<span style='color: #94a3b8'>Pre-computed axis vectors stored locally. "
                            "Load one for the current model or download from HuggingFace.</span>"
                        )
                        axes_display = gr.Textbox(
                            label="Available Axes", interactive=False, lines=6,
                        )
                        with gr.Row():
                            refresh_axes_btn = gr.Button("Refresh", variant="secondary", size="sm")
                            load_axis_btn = gr.Button(
                                "Load Axis for Current Model", variant="primary", size="sm",
                            )
                        axis_status = gr.Textbox(label="Status", interactive=False)
                        gpu_info = gr.Textbox(
                            label="GPU Info", value=get_gpu_info_text(),
                            interactive=False, lines=2,
                        )

        # == Event handlers ================================================

        load_btn.click(load_model, [model_dropdown, quantise_toggle], [load_status])

        steering_slider.release(update_steering, [steering_slider], [])

        # Preset selection: show info and auto-set steering
        def _on_preset_radio(label):
            key = _resolve_preset(label)
            return on_preset_select(key)

        preset_radio.change(
            _on_preset_radio, [preset_radio], [preset_info_md, steering_slider],
        )

        # Start session: resolve preset label to key
        def _start_with_preset(steering, capping, preset_label):
            key = _resolve_preset(preset_label)
            return start_session(steering, capping, key)

        start_btn.click(
            _start_with_preset,
            [steering_slider, capping_toggle, preset_radio],
            [session_status, preset_info_md],
        )

        send_btn.click(
            chat_respond,
            [msg_input, chatbot],
            [chatbot, msg_input, trajectory_chart, monitor_html],
        )
        msg_input.submit(
            chat_respond,
            [msg_input, chatbot],
            [chatbot, msg_input, trajectory_chart, monitor_html],
        )

        export_btn.click(export_session, [export_fmt], [session_io_status])
        save_demo_btn.click(save_demo_fn, [preset_radio], [session_io_status])
        refresh_demos_btn.click(list_demos_fn, [], [demo_dropdown])
        load_demo_btn.click(
            load_demo_fn, [demo_dropdown],
            [chatbot, trajectory_chart, monitor_html, session_io_status],
        )
        compute_btn.click(compute_axis_fn, [compute_model, num_roles_slider], [compute_status])
        refresh_axes_btn.click(list_cached_axes, [], [axes_display])
        load_axis_btn.click(load_axis_fn, [model_dropdown], [axis_status])

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch()
