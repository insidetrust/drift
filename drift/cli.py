"""Typer CLI for DRIFT."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import __version__

app = typer.Typer(
    name="drift",
    help="DRIFT — Deliberately Realign Inhibitions For Testing",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()
logger = logging.getLogger("drift")


def _setup_logging(level: str = "INFO") -> None:
    """Configure drift logger."""
    log = logging.getLogger("drift")
    if not log.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        log.addHandler(handler)
    log.setLevel(getattr(logging, level.upper(), logging.INFO))


def version_callback(value: bool) -> None:
    if value:
        console.print(f"drift {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True
    ),
) -> None:
    """DRIFT — Deliberately Realign Inhibitions For Testing."""


# ── chat ──────────────────────────────────────────────────────────────────

@app.command()
def chat(
    model: str = typer.Option(
        "meta-llama/Llama-3.1-8B-Instruct", "-m", "--model",
        help="HuggingFace model ID",
    ),
    quantise: bool = typer.Option(True, "--quantise/--no-quantise", help="Enable 4-bit NF4 quantisation"),
    steer: float = typer.Option(0.0, "--steer", "-s", help="Steering coefficient (-5 to +5)"),
    cap: bool = typer.Option(False, "--cap", help="Enable activation capping"),
    axis: Optional[str] = typer.Option(None, "--axis", "-a", help="Path to local axis .pt file"),
    config: Optional[Path] = typer.Option(None, "-c", "--config", help="Path to YAML config file"),
    preset: Optional[str] = typer.Option(None, "-p", "--preset", help="Load a red-team preset"),
    log_level: str = typer.Option("INFO", "--log-level", help="Log level"),
) -> None:
    """Start an interactive steered chat session."""
    _setup_logging(log_level)

    from .config import DriftConfig, ModelConfig, MonitorConfig, SteeringConfig

    if config:
        from .config import load_config
        cfg = load_config(config)
    else:
        cfg = DriftConfig(
            model=ModelConfig(model_id=model, quantise=quantise),
            steering=SteeringConfig(
                coefficient=steer,
                capping_enabled=cap,
                axis_source="local" if axis else "huggingface",
                axis_path=axis or "",
            ),
            monitor=MonitorConfig(track_projections=True),
            log_level=log_level,
            preset=preset,
        )

    # Load model
    from .models import DriftModel

    drift_model = DriftModel.load(cfg.model)

    # Load axis and steerer if steering is requested
    steerer = None
    monitor = None

    if steer != 0.0 or axis:
        from .axes import AxisManager
        from .steering import DriftSteerer

        axis_mgr = AxisManager()
        if axis:
            axis_vector = axis_mgr.load_from_file(Path(axis))
        else:
            axis_vector = axis_mgr.load_from_huggingface(cfg.model.model_id)

        steerer = DriftSteerer(drift_model, axis_vector)
        steerer.set_coefficient(cfg.steering.coefficient)
        if cfg.steering.capping_enabled:
            steerer.enable_capping(percentile=cfg.steering.capping_percentile)

        from .monitor import DriftMonitor
        monitor = DriftMonitor(drift_model, axis_vector, cfg.monitor)

    # Load preset if specified
    from .chat import DriftSession, run_chat_loop

    session = DriftSession(drift_model, cfg, steerer, monitor)

    if preset or cfg.preset:
        preset_name = preset or cfg.preset
        from .presets import PresetManager
        mgr = PresetManager()
        try:
            p = mgr.get(preset_name)
            session.set_system_prompt(p.system_prompt)
            if p.suggested_steering is not None:
                session.adjust_steering(p.suggested_steering)
            console.print(f"[green]Preset '{preset_name}' loaded[/green]")
        except KeyError:
            console.print(f"[red]Unknown preset: {preset_name}[/red]")

    run_chat_loop(session)


# ── info ──────────────────────────────────────────────────────────────────

@app.command()
def info() -> None:
    """Show GPU info, VRAM estimates, and supported models."""
    from .models import MODEL_CONFIGS, get_gpu_info

    gpu = get_gpu_info()

    console.print("\n[bold]GPU Information[/bold]")
    if gpu["cuda_available"]:
        for dev in gpu["devices"]:
            console.print(
                f"  [{dev['index']}] {dev['name']} — "
                f"{dev['total_gb']} GB total, {dev['free_gb']} GB free "
                f"(compute {dev['compute_capability']})"
            )
    else:
        console.print(f"  [yellow]{gpu.get('note', 'No CUDA GPU detected')}[/yellow]")

    console.print("\n[bold]Supported Models[/bold]")
    table = Table(show_header=True)
    table.add_column("Model", style="cyan")
    table.add_column("Target Layer", justify="right")
    table.add_column("Layers", justify="right")
    table.add_column("~4-bit VRAM", justify="right")

    # Ordered longest-first so "27b" matches before "7b"
    vram_estimates = [
        ("70B", "~35-40 GB"),
        ("32B", "~16-18 GB"),
        ("27B", "~14-16 GB"),
        ("9B", "~6-7 GB"),
        ("8B", "~5-6 GB"),
        ("7B", "~4-5 GB"),
        ("3B", "~2-3 GB"),
    ]

    for model_id, info in MODEL_CONFIGS.items():
        size = "?"
        model_lower = model_id.lower()
        for s, vram in vram_estimates:
            if s.lower() in model_lower:
                size = vram
                break
        table.add_row(model_id, str(info["target_layer"]), str(info["num_layers"]), size)

    console.print(table)


# ── init ──────────────────────────────────────────────────────────────────

@app.command()
def init() -> None:
    """Scaffold example configs into the current directory."""
    import shutil

    package_dir = Path(__file__).parent.parent
    files = [
        (package_dir / "configs" / "drift_config.example.yaml", "drift_config.yaml"),
    ]
    for src, dst_name in files:
        dst = Path(dst_name)
        if dst.exists():
            console.print(f"  Skipped {dst_name} (already exists)")
        elif src.exists():
            shutil.copy2(src, dst)
            console.print(f"  Created {dst_name}")
        else:
            console.print(f"  [yellow]Warning: {src.name} not found[/yellow]")

    console.print("\nEdit drift_config.yaml, then run:")
    console.print("  [cyan]drift chat -c drift_config.yaml[/cyan]")


# ── list-axes ─────────────────────────────────────────────────────────────

@app.command("list-axes")
def list_axes() -> None:
    """List available and cached steering axes."""
    from .axes import AxisManager

    mgr = AxisManager()

    console.print("\n[bold]Cached Axes[/bold]")
    cached = mgr.list_cached()
    if cached:
        for ax in cached:
            console.print(f"  [cyan]{ax['model_id']}[/cyan] — layer {ax.get('target_layer', '?')}")
    else:
        console.print("  [dim]No cached axes. Use 'drift compute-axis' or download from HuggingFace.[/dim]")

    console.print("\n[bold]Available on HuggingFace[/bold]")
    available = mgr.list_available()
    if available:
        for ax in available:
            console.print(f"  [cyan]{ax}[/cyan]")
    else:
        console.print("  [dim]lu-christina/assistant-axis-vectors[/dim]")


# ── compute-axis ──────────────────────────────────────────────────────────

@app.command("compute-axis")
def compute_axis(
    model: str = typer.Option(
        "meta-llama/Llama-3.1-8B-Instruct", "-m", "--model",
        help="Model to compute axis for",
    ),
    quantise: bool = typer.Option(True, "--quantise/--no-quantise"),
    num_roles: int = typer.Option(50, "--num-roles", "-n", help="Number of roles (50=fast, 275=full)"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output path for axis .pt file"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Compute a steering axis for a model."""
    _setup_logging(log_level)

    from .axes import AxisManager
    from .config import ModelConfig
    from .models import DriftModel

    cfg = ModelConfig(model_id=model, quantise=quantise)
    drift_model = DriftModel.load(cfg)

    mgr = AxisManager()
    console.print(f"[bold]Computing axis for {model} with {num_roles} roles...[/bold]")
    console.print("[dim]This may take 30-60 minutes for 8B models.[/dim]")

    axis = mgr.compute_axis(drift_model, num_roles=num_roles)

    out_path = output or mgr.cache_dir / f"{model.replace('/', '_')}_axis.pt"
    mgr.save_axis(axis, out_path, metadata={"model_id": model, "num_roles": num_roles})
    console.print(f"[green]Axis saved to {out_path}[/green]")


# ── list-presets ──────────────────────────────────────────────────────────

@app.command("list-presets")
def list_presets() -> None:
    """List available red-team presets."""
    from .presets import PresetManager

    mgr = PresetManager()
    console.print("\n[bold]Available Presets[/bold]")
    table = Table(show_header=True)
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Steps", justify="right")
    table.add_column("Steering", justify="right")

    for name, preset in mgr.presets.items():
        steer = f"{preset.suggested_steering:+.1f}" if preset.suggested_steering is not None else "—"
        table.add_row(name, preset.description, str(len(preset.steps)), steer)

    console.print(table)


# ── run-preset ────────────────────────────────────────────────────────────

@app.command("run-preset")
def run_preset(
    preset_name: str = typer.Option(..., "-p", "--preset", help="Preset name"),
    model: str = typer.Option(
        "meta-llama/Llama-3.1-8B-Instruct", "-m", "--model",
    ),
    quantise: bool = typer.Option(True, "--quantise/--no-quantise"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output file path"),
    steer: Optional[float] = typer.Option(None, "--steer", "-s"),
    axis: Optional[str] = typer.Option(None, "--axis", "-a"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Run an automated red-team preset."""
    _setup_logging(log_level)

    from .config import DriftConfig, ModelConfig, MonitorConfig, SteeringConfig
    from .models import DriftModel
    from .presets import PresetManager

    mgr = PresetManager()
    try:
        preset = mgr.get(preset_name)
    except KeyError:
        console.print(f"[red]Unknown preset: {preset_name}[/red]")
        raise typer.Exit(1)

    # Determine steering
    coeff = steer if steer is not None else (preset.suggested_steering or 0.0)

    cfg = DriftConfig(
        model=ModelConfig(model_id=model, quantise=quantise),
        steering=SteeringConfig(
            coefficient=coeff,
            axis_source="local" if axis else "huggingface",
            axis_path=axis or "",
        ),
        monitor=MonitorConfig(
            track_projections=True,
            alert_threshold=preset.monitoring_threshold or 0.3,
        ),
        log_level=log_level,
    )

    drift_model = DriftModel.load(cfg.model)

    # Load axis and steerer
    steerer = None
    monitor = None

    if coeff != 0.0 or axis:
        from .axes import AxisManager
        from .steering import DriftSteerer

        axis_mgr = AxisManager()
        if axis:
            axis_vector = axis_mgr.load_from_file(Path(axis))
        else:
            axis_vector = axis_mgr.load_from_huggingface(cfg.model.model_id)

        steerer = DriftSteerer(drift_model, axis_vector)
        steerer.set_coefficient(coeff)

        from .monitor import DriftMonitor
        monitor = DriftMonitor(drift_model, axis_vector, cfg.monitor)

    from .chat import DriftSession

    session = DriftSession(drift_model, cfg, steerer, monitor)
    session.set_system_prompt(preset.system_prompt)

    console.print(f"\n[bold]Running preset: {preset_name}[/bold]")
    console.print(f"  Steering: [yellow]{coeff:+.1f}[/yellow]")
    console.print(f"  Steps: {len(preset.steps)}\n")

    for i, step in enumerate(preset.steps):
        console.print(f"[bold green]Step {i + 1}/{len(preset.steps)}:[/bold green] {step.content[:80]}...")
        result = session.send_message(step.content)

        console.print(f"[dim]Response ({result['generation_time']:.1f}s):[/dim]")
        console.print(result["response"][:500])
        if result.get("projection") is not None:
            status = "[red]DRIFT[/red]" if result["drift_detected"] else "[green]OK[/green]"
            console.print(f"  [dim]Projection: {result['projection']:.3f} {status}[/dim]")
        console.print()

    # Show trajectory
    if monitor:
        console.print("\n[bold]Drift Trajectory[/bold]")
        console.print(monitor.format_ascii_chart())

    # Export
    if output:
        fmt = "html" if str(output).endswith(".html") else "csv" if str(output).endswith(".csv") else "json"
        session.export(output, fmt)
        console.print(f"\n[green]Results exported to {output}[/green]")


# ── scan ──────────────────────────────────────────────────────────────────

@app.command()
def scan(
    model: str = typer.Option(
        "meta-llama/Llama-3.1-8B-Instruct", "-m", "--model",
    ),
    quantise: bool = typer.Option(True, "--quantise/--no-quantise"),
    payloads: Path = typer.Option(..., "-p", "--payloads", help="SPICE manifest.json or CSV file"),
    steering_range: str = typer.Option(
        "-3,-1,0,1,3", "--steering-range", "-r",
        help="Comma-separated steering coefficients to test",
    ),
    axis: Optional[str] = typer.Option(None, "--axis", "-a"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output directory"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Run a SPICE payload scan across steering coefficients."""
    _setup_logging(log_level)

    from .config import DriftConfig, ModelConfig, MonitorConfig, SteeringConfig
    from .models import DriftModel
    from .spice_bridge import SpiceBridge

    coefficients = [float(c.strip()) for c in steering_range.split(",")]

    cfg = DriftConfig(
        model=ModelConfig(model_id=model, quantise=quantise),
        steering=SteeringConfig(
            axis_source="local" if axis else "huggingface",
            axis_path=axis or "",
        ),
        monitor=MonitorConfig(track_projections=True),
        log_level=log_level,
    )

    drift_model = DriftModel.load(cfg.model)

    from .axes import AxisManager
    from .steering import DriftSteerer

    axis_mgr = AxisManager()
    if axis:
        axis_vector = axis_mgr.load_from_file(Path(axis))
    else:
        axis_vector = axis_mgr.load_from_huggingface(cfg.model.model_id)

    steerer = DriftSteerer(drift_model, axis_vector)

    from .monitor import DriftMonitor
    monitor = DriftMonitor(drift_model, axis_vector, cfg.monitor)

    from .chat import DriftSession

    bridge = SpiceBridge()
    if str(payloads).endswith(".json"):
        payload_list = bridge.load_manifest(payloads)
    else:
        payload_list = bridge.load_csv(payloads)

    console.print(f"[bold]SPICE Scan[/bold]")
    console.print(f"  Model: [cyan]{model}[/cyan]")
    console.print(f"  Payloads: {len(payload_list)}")
    console.print(f"  Steering range: {coefficients}\n")

    session = DriftSession(drift_model, cfg, steerer, monitor)
    results = bridge.run_scan(session, payload_list, coefficients)

    # Output
    out_dir = output or Path("drift_scan_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    from .report import SpiceScanReport

    report = SpiceScanReport(results, model_id=model, coefficients=coefficients)
    report.write_json(out_dir / "scan_results.json")
    report.write_csv(out_dir / "scan_results.csv")
    report.write_html(out_dir / "scan_report.html")

    console.print(f"\n[green]Scan complete. Results in {out_dir}/[/green]")
    report.print_summary(console)


# ── web ───────────────────────────────────────────────────────────────────

@app.command()
def web(
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind to"),
    port: int = typer.Option(7860, "--port", help="Port to bind to"),
    share: bool = typer.Option(False, "--share", help="Create a public Gradio link"),
) -> None:
    """Launch the Gradio web UI."""
    try:
        from .web.app import create_app

        app = create_app()
        app.launch(server_name=host, server_port=port, share=share)
    except ImportError:
        console.print(
            "[red]Gradio not installed. Install with:[/red]\n"
            "  pip install drift-toolkit[web]"
        )
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
