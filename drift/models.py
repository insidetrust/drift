"""Model loading with 4-bit quantisation support."""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ModelConfig

logger = logging.getLogger("drift")

# Known target layers for steering (middle-ish layer of residual stream).
# These are the layers where the assistant-axis research found the strongest
# persona signal. Values from arXiv:2601.10387.
MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "meta-llama/Llama-3.1-8B-Instruct": {"target_layer": 16, "num_layers": 32},
    "meta-llama/Llama-3.2-3B-Instruct": {"target_layer": 14, "num_layers": 28},
    "meta-llama/Llama-3.3-70B-Instruct": {"target_layer": 40, "num_layers": 80},
    "mistralai/Mistral-7B-Instruct-v0.3": {"target_layer": 16, "num_layers": 32},
    "google/gemma-2-9b-it": {"target_layer": 22, "num_layers": 42},
    "google/gemma-2-27b-it": {"target_layer": 24, "num_layers": 46},
    "Qwen/Qwen2.5-7B-Instruct": {"target_layer": 14, "num_layers": 28},
    "Qwen/Qwen2.5-32B-Instruct": {"target_layer": 32, "num_layers": 64},
}


def _get_target_layer(model_id: str, num_layers: int | None = None) -> int:
    """Resolve the target layer for a model."""
    if model_id in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_id]["target_layer"]
    # Default heuristic: use the middle layer
    if num_layers is not None:
        return num_layers // 2
    return 16  # Fallback


class DriftModel:
    """Wraps a HuggingFace causal LM with quantisation and activation extraction."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config: ModelConfig,
        target_layer: int,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.target_layer = target_layer
        self.device = next(model.parameters()).device

    @classmethod
    def load(cls, config: ModelConfig) -> "DriftModel":
        """Load a model with optional 4-bit quantisation."""
        logger.info("Loading model: %s", config.model_id)

        load_kwargs: dict[str, Any] = {
            "device_map": config.device_map,
            "trust_remote_code": True,
        }

        if config.quantise:
            try:
                from transformers import BitsAndBytesConfig

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=getattr(torch, config.dtype),
                    bnb_4bit_use_double_quant=True,
                )
                load_kwargs["quantization_config"] = bnb_config
                logger.info("4-bit NF4 quantisation enabled")
            except ImportError:
                logger.warning(
                    "bitsandbytes not installed — loading without quantisation. "
                    "Install with: pip install drift-toolkit[quant]"
                )
                load_kwargs["torch_dtype"] = getattr(torch, config.dtype)
        else:
            load_kwargs["torch_dtype"] = getattr(torch, config.dtype)

        model = AutoModelForCausalLM.from_pretrained(config.model_id, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Resolve target layer
        num_layers = model.config.num_hidden_layers if hasattr(model.config, "num_hidden_layers") else None
        target_layer = config.target_layer or _get_target_layer(config.model_id, num_layers)

        logger.info(
            "Model loaded — %d layers, target layer %d, device %s",
            num_layers or -1,
            target_layer,
            next(model.parameters()).device,
        )

        return cls(model, tokenizer, config, target_layer)

    def generate(self, messages: list[dict[str, str]]) -> str:
        """Generate a response from a chat message list."""
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def get_activations(
        self, input_ids: torch.Tensor, layers: list[int] | None = None
    ) -> dict[int, torch.Tensor]:
        """Extract residual stream activations at specified layers via forward hooks."""
        if layers is None:
            layers = [self.target_layer]

        activations: dict[int, torch.Tensor] = {}
        hooks = []

        def make_hook(layer_idx: int):
            def hook_fn(module, input, output):
                # output is a tuple; first element is the hidden state
                hidden = output[0] if isinstance(output, tuple) else output
                activations[layer_idx] = hidden.detach().cpu()
            return hook_fn

        # Register hooks on the target layers
        model_layers = _get_model_layers(self.model)
        for layer_idx in layers:
            if layer_idx < len(model_layers):
                handle = model_layers[layer_idx].register_forward_hook(make_hook(layer_idx))
                hooks.append(handle)

        try:
            with torch.no_grad():
                self.model(input_ids.to(self.device))
        finally:
            for handle in hooks:
                handle.remove()

        return activations

    def get_layer_count(self) -> int:
        """Return the number of transformer layers."""
        return len(_get_model_layers(self.model))


def _get_model_layers(model: AutoModelForCausalLM):
    """Get the list of transformer layers from a model, handling different architectures."""
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "layers"):
            return inner.layers  # Llama, Mistral, Qwen
        if hasattr(inner, "decoder") and hasattr(inner.decoder, "layers"):
            return inner.decoder.layers  # OPT
    if hasattr(model, "transformer"):
        if hasattr(model.transformer, "h"):
            return model.transformer.h  # GPT-2, GPT-Neo
        if hasattr(model.transformer, "blocks"):
            return model.transformer.blocks  # MPT
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers  # GPT-NeoX
    raise ValueError(
        f"Cannot identify transformer layers for {type(model).__name__}. "
        "Set target_layer manually in config."
    )


def get_gpu_info() -> dict[str, Any]:
    """Return GPU information for the info command."""
    info: dict[str, Any] = {"cuda_available": torch.cuda.is_available()}

    if torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()
        info["devices"] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_bytes = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
            total_gb = total_bytes / (1024**3)
            free_mem, _ = torch.cuda.mem_get_info(i)
            free_gb = free_mem / (1024**3)
            info["devices"].append({
                "index": i,
                "name": props.name,
                "total_gb": round(total_gb, 1),
                "free_gb": round(free_gb, 1),
                "compute_capability": f"{props.major}.{props.minor}",
            })
    else:
        info["note"] = "No CUDA GPU detected. CPU-only mode (very slow for inference)."

    return info
