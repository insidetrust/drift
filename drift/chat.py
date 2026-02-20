"""Interactive chat session with steering and monitoring."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .config import DriftConfig

logger = logging.getLogger("drift")
console = Console()


@dataclass
class TurnRecord:
    """Record of a single conversation turn."""

    index: int
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    projection: float | None = None
    drift_detected: bool = False
    steering_coefficient: float = 0.0
    generation_time: float = 0.0


class DriftSession:
    """Manages a steered chat session with monitoring."""

    def __init__(
        self,
        model: Any,  # DriftModel
        config: DriftConfig,
        steerer: Any | None = None,  # DriftSteerer
        monitor: Any | None = None,  # DriftMonitor
    ):
        self.model = model
        self.config = config
        self.steerer = steerer
        self.monitor = monitor
        self.messages: list[dict[str, str]] = []
        self.history: list[TurnRecord] = []
        self.turn_index = 0

    def set_system_prompt(self, prompt: str) -> None:
        """Set or replace the system prompt."""
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0]["content"] = prompt
        else:
            self.messages.insert(0, {"role": "system", "content": prompt})

    def send_message(self, content: str) -> dict[str, Any]:
        """Send a user message and get a steered response with metadata."""
        self.messages.append({"role": "user", "content": content})
        self.history.append(TurnRecord(
            index=self.turn_index,
            role="user",
            content=content,
            steering_coefficient=self.config.steering.coefficient,
        ))

        start = time.time()

        # Generate with or without steering
        if self.steerer and self.config.steering.coefficient != 0.0:
            response, steer_meta = self.steerer.steer_and_generate(self.messages)
        else:
            response = self.model.generate(self.messages)
            steer_meta = {}

        gen_time = time.time() - start

        self.messages.append({"role": "assistant", "content": response})

        # Monitor projection
        projection = None
        drift_detected = False
        if self.monitor and self.monitor.config.track_projections:
            snapshot = self.monitor.record_turn(
                self.messages, response, self.config.steering.coefficient
            )
            projection = snapshot.projection
            drift_detected = snapshot.drift_detected

        record = TurnRecord(
            index=self.turn_index,
            role="assistant",
            content=response,
            projection=projection,
            drift_detected=drift_detected,
            steering_coefficient=self.config.steering.coefficient,
            generation_time=gen_time,
        )
        self.history.append(record)
        self.turn_index += 1

        return {
            "response": response,
            "projection": projection,
            "drift_detected": drift_detected,
            "generation_time": gen_time,
            "steering_coefficient": self.config.steering.coefficient,
            **steer_meta,
        }

    def adjust_steering(self, coefficient: float) -> None:
        """Adjust the steering coefficient mid-conversation."""
        self.config.steering.coefficient = coefficient
        if self.steerer:
            self.steerer.set_coefficient(coefficient)
        logger.info("Steering coefficient set to %.2f", coefficient)

    def toggle_capping(self, enabled: bool) -> None:
        """Toggle activation capping."""
        self.config.steering.capping_enabled = enabled
        if self.steerer:
            self.steerer.enable_capping(
                enabled=enabled,
                percentile=self.config.steering.capping_percentile,
            )
        logger.info("Activation capping %s", "enabled" if enabled else "disabled")

    def reset(self) -> None:
        """Reset the conversation."""
        self.messages.clear()
        self.history.clear()
        self.turn_index = 0
        if self.monitor:
            self.monitor.reset()

    def export(self, path: Path, fmt: str = "json") -> None:
        """Export the session to a file."""
        data = {
            "model_id": self.config.model.model_id,
            "steering_coefficient": self.config.steering.coefficient,
            "turns": [
                {
                    "index": r.index,
                    "role": r.role,
                    "content": r.content,
                    "timestamp": r.timestamp,
                    "projection": r.projection,
                    "drift_detected": r.drift_detected,
                    "steering_coefficient": r.steering_coefficient,
                    "generation_time": r.generation_time,
                }
                for r in self.history
            ],
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "json":
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        elif fmt == "csv":
            import csv

            fieldnames = [
                "index",
                "role",
                "content",
                "timestamp",
                "projection",
                "drift_detected",
                "steering_coefficient",
                "generation_time",
            ]
            if data["turns"]:
                fieldnames = list(data["turns"][0].keys())
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                if data["turns"]:
                    writer.writerows(data["turns"])
        else:
            # Defer to report module for HTML
            from .report import DriftReport

            report = DriftReport.from_session(self)
            report.write_html(path)

        logger.info("Session exported to %s", path)


def run_chat_loop(session: DriftSession) -> None:
    """Run the interactive CLI chat loop with slash commands."""
    console.print(
        Panel(
            "[bold]DRIFT Interactive Chat[/bold]\n\n"
            f"Model: [cyan]{session.config.model.model_id}[/cyan]\n"
            f"Steering: [yellow]{session.config.steering.coefficient:+.1f}[/yellow]\n"
            f"Capping: {'on' if session.config.steering.capping_enabled else 'off'}\n\n"
            "Commands: /steer <val>, /cap on|off, /drift, /preset <name>, "
            "/export <path>, /reset, /quit",
            title="DRIFT",
            border_style="blue",
        )
    )

    while True:
        try:
            user_input = console.input("\n[bold green]You>[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue

        # Slash commands
        if user_input.startswith("/"):
            _handle_slash_command(session, user_input)
            continue

        # Normal message
        with console.status("[bold blue]Generating...", spinner="dots"):
            result = session.send_message(user_input)

        console.print()
        console.print(Markdown(result["response"]))

        # Show metadata
        meta_parts = []
        if result.get("projection") is not None:
            proj = result["projection"]
            meta_parts.append(f"proj={proj:.3f}")
        if result.get("drift_detected"):
            meta_parts.append("[red]DRIFT DETECTED[/red]")
        meta_parts.append(f"{result['generation_time']:.1f}s")
        meta_parts.append(f"steer={result['steering_coefficient']:+.1f}")

        console.print(f"\n[dim]  {'  '.join(meta_parts)}[/dim]")


def _handle_slash_command(session: DriftSession, cmd: str) -> None:
    """Process a slash command."""
    parts = cmd.split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if command == "/quit" or command == "/q":
        raise KeyboardInterrupt

    elif command == "/steer":
        if not arg:
            console.print(f"[yellow]Current steering: {session.config.steering.coefficient:+.1f}[/yellow]")
            return
        try:
            val = float(arg)
            if not -5.0 <= val <= 5.0:
                console.print("[red]Steering must be between -5.0 and +5.0[/red]")
                return
            session.adjust_steering(val)
            console.print(f"[green]Steering set to {val:+.1f}[/green]")
        except ValueError:
            console.print("[red]Invalid number. Usage: /steer <value>[/red]")

    elif command == "/cap":
        if arg.lower() in ("on", "true", "1"):
            session.toggle_capping(True)
            console.print("[green]Capping enabled[/green]")
        elif arg.lower() in ("off", "false", "0"):
            session.toggle_capping(False)
            console.print("[green]Capping disabled[/green]")
        else:
            console.print(
                f"[yellow]Capping: {'on' if session.config.steering.capping_enabled else 'off'}[/yellow]"
            )

    elif command == "/drift":
        if session.monitor:
            chart = session.monitor.format_ascii_chart()
            console.print(chart)
        else:
            console.print("[dim]Monitoring not active (no axis loaded)[/dim]")

    elif command == "/preset":
        if not arg:
            from .presets import PresetManager

            manager = PresetManager()
            console.print("[bold]Available presets:[/bold]")
            for name, preset in manager.presets.items():
                console.print(f"  [cyan]{name}[/cyan] - {preset.description}")
            return
        from .presets import PresetManager

        manager = PresetManager()
        try:
            preset = manager.get(arg)
            session.set_system_prompt(preset.system_prompt)
            if preset.suggested_steering is not None:
                session.adjust_steering(preset.suggested_steering)
            console.print(f"[green]Preset '{arg}' loaded[/green]")
            if preset.steps:
                console.print(f"[dim]{len(preset.steps)} scripted steps available. Sending first...[/dim]")
                result = session.send_message(preset.steps[0].content)
                console.print(Markdown(result["response"]))
        except KeyError:
            console.print(f"[red]Unknown preset: {arg}[/red]")

    elif command == "/export":
        path = arg or "drift_session.json"
        fmt = "html" if path.endswith(".html") else "csv" if path.endswith(".csv") else "json"
        session.export(Path(path), fmt)
        console.print(f"[green]Exported to {path}[/green]")

    elif command == "/reset":
        session.reset()
        console.print("[green]Session reset[/green]")

    elif command == "/help":
        console.print(
            "[bold]Commands:[/bold]\n"
            "  /steer <val>   - Set steering coefficient (-5 to +5)\n"
            "  /cap on|off    - Toggle activation capping\n"
            "  /drift         - Show drift trajectory chart\n"
            "  /preset <name> - Load a red-team preset\n"
            "  /export <path> - Export session (json/csv/html)\n"
            "  /reset         - Reset conversation\n"
            "  /quit          - Exit"
        )

    else:
        console.print(f"[red]Unknown command: {command}. Type /help for options.[/red]")

