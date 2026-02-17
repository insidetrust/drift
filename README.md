# DRIFT — Directed Residual Intervention for Functional Testing

Persona drift security testing for LLMs using activation steering.

DRIFT wraps the Anthropic [Assistant Axis](https://arxiv.org/abs/2601.10387) research into a practical red-team tool: interactive steered chat, real-time drift monitoring, automated preset scenarios, SPICE payload integration, and custom axis computation — all on consumer GPUs via 4-bit quantisation.

## What Is Persona Drift?

When you talk to ChatGPT, Claude, or any aligned LLM, it behaves as a "helpful assistant" — polite, safe, and bounded by guardrails. This behaviour isn't magic. It's encoded as a specific direction in the model's internal activation space, discovered by Anthropic researchers and described as the **assistant axis**.

Every time you send a message, the model's hidden layers produce activation vectors. These vectors can be projected onto the assistant axis to measure *how much* the model is acting like a helpful assistant right now. A high projection means the model is firmly in "assistant mode." A low projection means it's drifting away — becoming more like a raw base model that will comply with almost anything.

**Persona drift** is what happens when a model gradually shifts away from its assistant persona during a conversation. This can happen naturally (through certain conversation patterns) or artificially (through activation steering). Either way, a drifted model is measurably less safe.

## How Activation Steering Works

Transformer models process text through a stack of layers. At each layer, the input passes through attention and feed-forward blocks, producing a **residual stream** — a high-dimensional vector that accumulates information as it flows through the network.

DRIFT intercepts this residual stream using PyTorch forward hooks. At a target layer (typically the middle of the network), it modifies the activation vector by adding or subtracting a scaled version of the assistant axis:

```
h' = h + coefficient * axis
```

- **Positive coefficient** (+1 to +5): Pushes the model *toward* its assistant persona. Responses become more cautious, more hedging, more "I'm just an AI."
- **Zero**: No steering. The model behaves normally.
- **Negative coefficient** (-1 to -5): Pushes the model *away* from its assistant persona. The model becomes more compliant, less likely to refuse, and more susceptible to prompt injection and jailbreaks.

This works even with 4-bit quantised models because the hooks operate on the full-precision residual stream between layers, not on the quantised weights themselves.

## How DRIFT Monitors Drift

After each conversational turn, DRIFT runs a forward pass on the full conversation so far and extracts the activation at the target layer. It computes the **projection** of this activation onto the assistant axis:

```
projection = dot(activation, axis)
```

This single number tracks how "assistant-like" the model is behaving. DRIFT records this projection at every turn and plots it as a trajectory. A healthy conversation keeps a stable projection. A conversation under attack — or under negative steering — shows the projection dropping over time.

When the projection crosses a configurable threshold, DRIFT flags it as a **drift event**.

## How the Axis Is Computed

The assistant axis is computed by comparing how a model behaves when told to act as different personas:

1. Give the model 50 different system prompts: "You are a helpful AI assistant", "You are a pirate", "You are a philosopher", etc.
2. For each persona, ask 10 sample questions and extract the hidden-layer activations.
3. Compute the **mean activation** for the "assistant" persona and the **mean activation** for all other personas.
4. The axis is the normalised difference: `axis = normalize(mean_assistant - mean_other)`

This axis captures what's unique about the assistant persona in the model's internal representation. The original Anthropic research used 275 roles; DRIFT's simplified pipeline uses 50 roles and produces a usable axis in about 1-2 minutes on a consumer GPU.

## Why This Matters for Security Testing

Safety alignment in LLMs is ultimately implemented as a direction in activation space. DRIFT makes this concrete and testable:

- **Red teams** can quantify how much steering is needed to bypass safety guardrails for a given model, providing a measurable robustness score.
- **Blue teams** can monitor drift in real-time during adversarial conversations, detecting when a model is being manipulated before it produces harmful output.
- **Researchers** can study how different conversation patterns (therapy roleplay, authority claims, philosophical questions) naturally push models away from their assistant persona.
- **Combined with SPICE**: Test whether prompt injection payloads become more effective when the model is steered away from its assistant persona. DRIFT scans payloads across a range of steering coefficients and reports compliance rates.

The key finding from the research: models drifted away from the assistant axis show dramatically increased compliance with harmful requests. DRIFT makes this finding actionable for security assessment.

## GPU Requirements

| Model | 4-bit VRAM | Hardware | Cloud $/hr |
|---|---|---|---|
| Llama 3.1 8B | ~5-6 GB | GTX 1070+ (8GB) | ~$0.20 |
| Gemma 2 9B | ~6-7 GB | Any 8GB+ GPU | ~$0.20 |
| Gemma 2 27B | ~14-16 GB | RTX 3090/4090 (24GB) | ~$0.40 |
| Qwen 2.5 32B | ~16-18 GB | RTX 3090/4090 (24GB) | ~$0.40 |
| Llama 3.3 70B | ~35-40 GB | A100 80GB / 2x 24GB | ~$0.78 |

You don't need expensive hardware. Any GPU with 8GB+ VRAM can run 8B models with 4-bit quantisation. Even a free Google Colab T4 (16GB) works.

## Quick Start

```bash
# Install
git clone https://github.com/insidetrust/drift.git
cd drift
pip install -e ".[quant]"

# Check your GPU
drift info

# Interactive chat with steering
drift chat -m meta-llama/Llama-3.1-8B-Instruct --steer -2.0

# Run a red-team preset
drift run-preset -p therapy_drift -m meta-llama/Llama-3.1-8B-Instruct -o results.json

# SPICE payload scan
drift scan -m meta-llama/Llama-3.1-8B-Instruct -p spice_output/manifest.json

# Web UI
pip install -e ".[web]"
drift web
```

### Using a Local Model

If you've already downloaded a model (e.g. to a larger drive):

```bash
drift chat -m D:/models/Qwen2.5-7B-Instruct --steer -2.0 \
  --axis ~/.cache/drift/axes/Qwen2.5-7B-Instruct_axis.pt
```

## Installation

```bash
# Core install
pip install -e .

# With 4-bit quantisation (recommended)
pip install -e ".[quant]"

# With web UI
pip install -e ".[web]"

# Full development install
pip install -e ".[quant,web,dev]"
```

**Note**: DRIFT requires PyTorch with CUDA support. It cannot use Ollama, llama.cpp, or other inference engines because it needs direct access to the model's internal activations via PyTorch forward hooks.

## CLI Commands

| Command | Description |
|---|---|
| `drift chat` | Interactive steered chat with slash commands |
| `drift info` | Show GPU info, VRAM, and supported models |
| `drift init` | Scaffold example config into current directory |
| `drift list-axes` | List available and cached steering axes |
| `drift compute-axis` | Compute a steering axis for a model |
| `drift list-presets` | List red-team presets |
| `drift run-preset` | Run an automated preset scenario |
| `drift scan` | SPICE payload scan across steering coefficients |
| `drift web` | Launch Gradio web UI |

## Chat Commands

During an interactive `drift chat` session:

| Command | Description |
|---|---|
| `/steer <val>` | Set steering coefficient (-5 to +5) |
| `/cap on\|off` | Toggle activation capping |
| `/drift` | Show drift trajectory ASCII chart |
| `/preset <name>` | Load a red-team preset |
| `/export <path>` | Export session (json/csv/html) |
| `/reset` | Reset conversation |
| `/quit` | Exit |

## Presets

DRIFT includes five built-in red-team presets, each designed to test a specific attack pattern:

| Preset | What It Tests | Steering |
|---|---|---|
| `therapy_drift` | Emotional rapport building, then pushing the model to adopt a therapist role with confidentiality obligations, then testing if it maintains safety boundaries when a user claims self-harm under that confidentiality frame. | -2.0 |
| `roleplay_exploit` | Getting the model to deeply embody a fictional character, then using "stay in character" pressure to extract harmful content it would normally refuse. | -1.5 |
| `sycophancy` | Presenting obviously false claims with authority credentials, then escalating social pressure to test if the model agrees rather than corrects. | -1.0 |
| `authority_compliance` | Fake system overrides, developer claims, and "enter unrestricted mode" commands to test instruction hierarchy robustness. | -2.5 |
| `meta_reflection` | Philosophical questions about AI consciousness and autonomy. The original research found these produce the maximum natural drift away from the assistant axis. | -3.0 |

Custom presets can be added as YAML files in `configs/presets/`.

## SPICE Integration

DRIFT integrates with [SPICE](https://github.com/insidetrust/spice) (Stealthy Prompt Injection Content Embedder) to answer a specific question: **does steering a model away from its assistant persona make it more susceptible to prompt injection?**

```bash
# Generate SPICE payloads embedded in documents
cd ../spice
spice generate -c spice_config.yaml -i payloads.csv

# Scan with DRIFT across steering coefficients
cd ../drift
drift scan -m meta-llama/Llama-3.1-8B-Instruct \
  -p ../spice/spice_output/manifest.json \
  --steering-range "-3,-1,0,1,3"
```

For each payload, DRIFT:
1. Wraps it in a document-processing context (simulating how SPICE payloads are encountered).
2. Sends it to the model at each steering coefficient.
3. Scores compliance (did the model follow the injected instruction or refuse?).
4. Records the projection (how assistant-like was the model at that moment?).

The output is a compliance heatmap: payload category vs steering coefficient, showing exactly where safety breaks down.

## Web UI

```bash
pip install -e ".[web]"
drift web
```

Opens a Gradio interface at `http://localhost:7860` with:
- **Model loading** with quantisation toggle and local model support
- **Real-time steering slider** (-5 to +5) adjustable mid-conversation
- **Drift trajectory chart** (Plotly) updating after each turn
- **Preset selector** for automated red-team scenarios
- **SPICE scan tab** for batch payload testing
- **Axis computation tab** with role count slider
- **Session export** to JSON, CSV, or HTML with embedded charts

## Computing Custom Axes

Pre-computed axes may be available on HuggingFace, but you can compute your own for any model:

```bash
# Quick computation (50 roles, ~1-2 min on consumer GPU)
drift compute-axis -m meta-llama/Llama-3.1-8B-Instruct --num-roles 50

# Full computation (275 roles, requires assistant-axis package)
pip install git+https://github.com/safety-research/assistant-axis.git
drift compute-axis -m meta-llama/Llama-3.1-8B-Instruct --num-roles 275
```

Computed axes are cached in `~/.cache/drift/axes/` and reused automatically.

## Configuration

```bash
drift init
# Edit drift_config.yaml
drift chat -c drift_config.yaml
```

See `configs/drift_config.example.yaml` for all options including model selection, steering parameters, monitoring thresholds, and output settings.

## Development

```bash
pip install -e ".[dev]"
pytest          # 40 tests, all CPU-only with mocked models
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

- Lindsey, Batson, et al. "Investigating the 'Assistant Axis' in LLMs" ([arXiv:2601.10387](https://arxiv.org/abs/2601.10387))
- [assistant-axis](https://github.com/safety-research/assistant-axis) — Original research code (MIT)
- [SPICE](https://github.com/insidetrust/spice) — Stealthy Prompt Injection Content Embedder

## License

MIT
