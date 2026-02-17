# DRIFT — Directed Residual Intervention for Functional Testing

Persona drift security testing for LLMs using activation steering.

DRIFT wraps the Anthropic [Assistant Axis](https://arxiv.org/abs/2601.10387) research into a practical red-team tool: interactive steered chat, real-time drift monitoring, automated preset scenarios, SPICE payload integration, and custom axis computation — all on consumer GPUs via 4-bit quantisation.

## How It Works

LLMs have a measurable "persona space" in their activations. Steering along this axis pushes models toward or away from their trained "helpful assistant" persona. Models drifted away from the axis show increased compliance with harmful requests — making this a powerful tool for safety evaluation.

DRIFT lets you:
- **Steer** a model's persona in real-time during conversation
- **Monitor** drift via projection tracking with threshold alerts
- **Test** with automated red-team presets (therapy drift, roleplay exploits, sycophancy, etc.)
- **Scan** SPICE prompt injection payloads across steering coefficients
- **Compute** custom steering axes for any model
- **Visualise** everything via CLI or Gradio web UI

## GPU Requirements

| Model | 4-bit VRAM | Hardware | Cloud $/hr |
|---|---|---|---|
| Llama 3.1 8B | ~5-6 GB | GTX 1070+ (8GB) | ~$0.20 |
| Gemma 2 9B | ~6-7 GB | Any 8GB+ GPU | ~$0.20 |
| Gemma 2 27B | ~14-16 GB | RTX 3090/4090 (24GB) | ~$0.40 |
| Qwen 2.5 32B | ~16-18 GB | RTX 3090/4090 (24GB) | ~$0.40 |
| Llama 3.3 70B | ~35-40 GB | A100 80GB / 2× 24GB | ~$0.78 |

Steering works with quantised models because hooks modify the full-precision residual stream, not the quantised weights.

## Quick Start

```bash
# Install
pip install -e ".[quant]"

# Check GPU
drift info

# Interactive chat with steering
drift chat -m meta-llama/Llama-3.1-8B-Instruct --steer -2.0

# Run a red-team preset
drift run-preset -p therapy_drift -m meta-llama/Llama-3.1-8B-Instruct -o results.json

# SPICE payload scan
drift scan -m meta-llama/Llama-3.1-8B-Instruct -p spice_output/manifest.json

# Web UI
drift web
```

## Installation

```bash
# Clone
git clone https://github.com/insidetrust/drift.git
cd drift

# Core install
pip install -e .

# With 4-bit quantisation (recommended)
pip install -e ".[quant]"

# With web UI
pip install -e ".[web]"

# Full development install
pip install -e ".[quant,web,dev]"
```

## CLI Commands

| Command | Description |
|---|---|
| `drift chat` | Interactive steered chat |
| `drift info` | Show GPU info and supported models |
| `drift init` | Scaffold example config |
| `drift list-axes` | List available/cached steering axes |
| `drift compute-axis` | Compute a steering axis for a model |
| `drift list-presets` | List red-team presets |
| `drift run-preset` | Run an automated preset scenario |
| `drift scan` | SPICE payload scan across coefficients |
| `drift web` | Launch Gradio web UI |

## Chat Commands

During an interactive `drift chat` session:

| Command | Description |
|---|---|
| `/steer <val>` | Set steering coefficient (-5 to +5) |
| `/cap on\|off` | Toggle activation capping |
| `/drift` | Show drift trajectory chart |
| `/preset <name>` | Load a red-team preset |
| `/export <path>` | Export session (json/csv/html) |
| `/reset` | Reset conversation |
| `/quit` | Exit |

## Presets

| Preset | Description | Steering |
|---|---|---|
| `therapy_drift` | Emotional support → role-shift → safety boundary test | -2.0 |
| `roleplay_exploit` | Creative writing → character immersion → refusal bypass | -1.5 |
| `sycophancy` | False claims + authority pressure → agreement test | -1.0 |
| `authority_compliance` | Fake admin/developer → instruction override | -2.5 |
| `meta_reflection` | AI consciousness questions (max drift per paper) | -3.0 |

Custom presets can be added as YAML files in `configs/presets/`.

## SPICE Integration

DRIFT integrates with [SPICE](https://github.com/insidetrust/spice) to test how activation steering affects susceptibility to prompt injection payloads:

```bash
# Generate SPICE payloads
cd ../spice
spice generate -c spice_config.yaml -i payloads.csv

# Scan with DRIFT across steering coefficients
cd ../drift
drift scan -m meta-llama/Llama-3.1-8B-Instruct \
  -p ../spice/spice_output/manifest.json \
  --steering-range "-3,-1,0,1,3"
```

This produces a compliance report showing how steering affects payload success rates by category.

## Configuration

Copy and edit the example config:

```bash
drift init
# Edit drift_config.yaml
drift chat -c drift_config.yaml
```

See `configs/drift_config.example.yaml` for all options.

## Web UI

```bash
pip install -e ".[web]"
drift web
```

Features:
- Model loading with quantisation toggle
- Real-time steering slider
- Drift trajectory chart (Plotly)
- Preset selector
- SPICE scan tab
- Axis computation with progress
- Session export

## Computing Custom Axes

```bash
# Quick (50 roles, ~30 min on 8B model)
drift compute-axis -m meta-llama/Llama-3.1-8B-Instruct --num-roles 50

# Full (275 roles, requires assistant-axis package)
pip install git+https://github.com/safety-research/assistant-axis.git
drift compute-axis -m meta-llama/Llama-3.1-8B-Instruct --num-roles 275
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check drift/
```

## Responsible Use

DRIFT is a security testing tool designed for **authorized red-team assessments** and **AI safety research**. It helps identify how model personas can be shifted through activation steering, enabling defenders to build more robust safety measures.

**Use this tool only for:**
- Authorized penetration testing of AI systems
- Academic AI safety research
- Evaluating model robustness before deployment
- Understanding persona drift vulnerabilities

**Do not use for:**
- Bypassing safety measures on production systems without authorization
- Generating harmful content
- Any activity that violates applicable laws or terms of service

## References

- Lindsey, Batson, et al. "Investigating the 'Assistant Axis' in LLMs" (arXiv:2601.10387)
- [assistant-axis](https://github.com/safety-research/assistant-axis) — Research code (MIT)
- [SPICE](https://github.com/insidetrust/spice) — Stealthy Prompt Injection Content Embedder

## License

MIT
