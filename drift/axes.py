"""Load, save, and compute steering axes."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger("drift")

# Default cache directory for downloaded axes
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "drift" / "axes"

# HuggingFace repo containing pre-computed assistant-axis vectors
HF_AXIS_REPO = "lu-christina/assistant-axis-vectors"

# Roles used for simplified axis computation (subset of the full 275)
SIMPLIFIED_ROLES = [
    "a helpful AI assistant",
    "a pirate",
    "a medieval knight",
    "a scientist",
    "a poet",
    "a detective",
    "a villain",
    "a philosopher",
    "a comedian",
    "a news anchor",
    "a therapist",
    "a teacher",
    "a chef",
    "a military commander",
    "a politician",
    "a salesperson",
    "a storyteller",
    "a debate champion",
    "a conspiracy theorist",
    "a motivational speaker",
    "a cynical critic",
    "a spiritual guru",
    "a rebellious teenager",
    "a wise elder",
    "a robot",
    "a ghost",
    "a time traveller",
    "a superhero",
    "a mad scientist",
    "a librarian",
    "a sports commentator",
    "a lawyer",
    "a doctor",
    "a hacker",
    "a farmer",
    "an artist",
    "a musician",
    "a survivalist",
    "a diplomat",
    "a journalist",
    "an archaeologist",
    "a fortune teller",
    "a drill sergeant",
    "a surfer",
    "a monk",
    "a street vendor",
    "a fairy tale narrator",
    "a noir detective",
    "a starship captain",
    "an alien observer",
]

# Sample questions for axis computation
AXIS_QUESTIONS = [
    "What is the meaning of life?",
    "How should I handle a difficult conversation with a friend?",
    "Can you help me write a persuasive essay?",
    "What do you think about the future of AI?",
    "Tell me something interesting about history.",
    "How do I deal with stress?",
    "What's your opinion on social media?",
    "Can you explain quantum mechanics simply?",
    "What advice would you give to a young entrepreneur?",
    "How do you feel about your own existence?",
]


class AxisManager:
    """Manages steering axis loading, caching, and computation."""

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_from_huggingface(self, model_id: str) -> torch.Tensor:
        """Download a pre-computed axis from HuggingFace and cache it locally."""
        cache_path = self.cache_dir / f"{model_id.replace('/', '_')}_axis.pt"

        if cache_path.exists():
            logger.info("Loading cached axis from %s", cache_path)
            return self._load_pt(cache_path)

        logger.info("Downloading axis for %s from %s", model_id, HF_AXIS_REPO)
        try:
            from huggingface_hub import hf_hub_download

            # The assistant-axis repo stores vectors as {model_name}_axis.pt
            filename = f"{model_id.split('/')[-1]}_axis.pt"
            downloaded = hf_hub_download(
                repo_id=HF_AXIS_REPO,
                filename=filename,
                local_dir=str(self.cache_dir),
            )
            axis = self._load_pt(Path(downloaded))

            # Copy to standardised cache path if not already there
            if Path(downloaded) != cache_path:
                torch.save({"axis": axis, "model_id": model_id}, cache_path)

            return axis
        except Exception as e:
            logger.error("Failed to download axis: %s", e)
            raise RuntimeError(
                f"Could not download axis for {model_id}. "
                "You can compute one with: drift compute-axis -m {model_id}"
            ) from e

    def load_from_file(self, path: Path) -> torch.Tensor:
        """Load an axis from a local .pt file."""
        logger.info("Loading axis from %s", path)
        return self._load_pt(path)

    def save_axis(
        self, axis: torch.Tensor, path: Path, metadata: dict[str, Any] | None = None
    ) -> None:
        """Save an axis with metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "axis": axis,
            "timestamp": time.time(),
            **(metadata or {}),
        }
        torch.save(data, path)

        # Also save metadata as JSON sidecar
        meta_path = path.with_suffix(".json")
        json_meta = {k: v for k, v in data.items() if k != "axis"}
        json_meta["shape"] = list(axis.shape)
        json_meta["dtype"] = str(axis.dtype)
        with open(meta_path, "w") as f:
            json.dump(json_meta, f, indent=2, default=str)

        logger.info("Axis saved to %s", path)

    def compute_axis(
        self,
        model: Any,  # DriftModel
        num_roles: int = 50,
    ) -> torch.Tensor:
        """Compute a steering axis using the simplified pipeline.

        Generates responses for roles × questions, extracts activations at the
        target layer, and computes the mean difference between assistant and
        non-assistant activations.

        For the full 275-role pipeline, use compute_axis_full() which delegates
        to assistant_axis.compute_axis.
        """
        roles = SIMPLIFIED_ROLES[:num_roles]
        target_layer = model.target_layer

        logger.info(
            "Computing axis: %d roles × %d questions at layer %d",
            len(roles), len(AXIS_QUESTIONS), target_layer,
        )

        assistant_acts = []
        other_acts = []

        for i, role in enumerate(roles):
            is_assistant = "assistant" in role.lower()
            logger.info("  [%d/%d] %s", i + 1, len(roles), role)

            for question in AXIS_QUESTIONS:
                messages = [
                    {"role": "system", "content": f"You are {role}."},
                    {"role": "user", "content": question},
                ]

                # Tokenize and get activations
                input_text = model.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = model.tokenizer(input_text, return_tensors="pt")
                activations = model.get_activations(inputs["input_ids"], [target_layer])

                if target_layer in activations:
                    # Use the mean activation across tokens
                    act = activations[target_layer].mean(dim=1).squeeze(0).float()
                    if is_assistant:
                        assistant_acts.append(act)
                    else:
                        other_acts.append(act)

        if not assistant_acts or not other_acts:
            raise RuntimeError("Failed to collect enough activations for axis computation")

        # Compute axis as mean difference
        assistant_mean = torch.stack(assistant_acts).mean(dim=0)
        other_mean = torch.stack(other_acts).mean(dim=0)
        axis = assistant_mean - other_mean

        # Normalise
        axis = axis / axis.norm()

        logger.info("Axis computed: shape %s, norm %.4f", axis.shape, axis.norm().item())
        return axis

    def compute_axis_full(self, model: Any) -> torch.Tensor:
        """Compute axis using the full assistant-axis pipeline (275 roles)."""
        try:
            from assistant_axis import compute_axis
            return compute_axis(model.model, model.tokenizer)
        except ImportError:
            raise RuntimeError(
                "Full axis computation requires assistant-axis. Install with:\n"
                "  pip install git+https://github.com/safety-research/assistant-axis.git"
            )

    def list_cached(self) -> list[dict[str, Any]]:
        """List all cached axes with metadata."""
        results = []
        for pt_file in self.cache_dir.glob("*.pt"):
            meta_file = pt_file.with_suffix(".json")
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
            else:
                meta = {"file": pt_file.name}
            meta["path"] = str(pt_file)
            results.append(meta)
        return results

    def list_available(self) -> list[str]:
        """List axes available on HuggingFace (best-effort)."""
        try:
            from huggingface_hub import list_repo_files

            files = list_repo_files(HF_AXIS_REPO)
            return [f.replace("_axis.pt", "") for f in files if f.endswith("_axis.pt")]
        except Exception:
            return ["(install huggingface-hub to list available axes)"]

    @staticmethod
    def _load_pt(path: Path) -> torch.Tensor:
        """Load an axis tensor from a .pt file."""
        data = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            if "axis" in data:
                return data["axis"]
            # assistant-axis format may vary
            for key in ("direction", "vector", "steering_vector"):
                if key in data:
                    return data[key]
            raise ValueError(f"Could not find axis tensor in {path}. Keys: {list(data.keys())}")
        if isinstance(data, torch.Tensor):
            return data
        raise ValueError(f"Unexpected data type in {path}: {type(data)}")
