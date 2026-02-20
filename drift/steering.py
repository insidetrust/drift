"""Activation steering wrapper with fallback hook implementation."""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger("drift")


class DriftSteerer:
    """Steers model activations along a given axis.

    Uses assistant_axis.ActivationSteering when available, otherwise falls back
    to a native PyTorch forward hook implementation.
    """

    def __init__(self, model: Any, axis: torch.Tensor):
        """
        Args:
            model: A DriftModel instance.
            axis: The steering axis tensor (1D, matching hidden dim).
        """
        self.model = model
        self.axis = axis.to(model.device)
        self.coefficient = 0.0
        self.capping_enabled = False
        self.capping_percentile = 95.0
        self._capping_threshold: float | None = None
        self._hook_handle = None
        self._use_native = not self._try_import_assistant_axis()

        if self._use_native:
            logger.info("Using native steering hooks (assistant-axis not installed)")
        else:
            logger.info("Using assistant-axis ActivationSteering")

    def _try_import_assistant_axis(self) -> bool:
        """Check if assistant_axis is available."""
        try:
            import assistant_axis  # noqa: F401
            return True
        except ImportError:
            return False

    def set_coefficient(self, value: float) -> None:
        """Set the steering coefficient."""
        self.coefficient = value
        logger.debug("Steering coefficient set to %.2f", value)

    def enable_capping(
        self,
        enabled: bool = True,
        percentile: float | None = None,
        layers: list[int] | None = None,
    ) -> None:
        """Enable or disable activation capping.

        Capping formula: h <- h - v * min(<h,v> - tau, 0)
        where tau is the percentile threshold of projections.
        """
        self.capping_enabled = enabled
        if percentile is not None:
            self.capping_percentile = percentile
        if not enabled:
            self._capping_threshold = None
            logger.info("Activation capping disabled")
            return
        if self._capping_threshold is None:
            try:
                self.compute_capping_threshold()
            except Exception as e:
                self.capping_enabled = False
                self._capping_threshold = None
                logger.warning("Failed to enable capping: %s", e)
                return
        logger.info(
            "Activation capping enabled (p%.0f, threshold=%.4f)",
            self.capping_percentile,
            self._capping_threshold,
        )

    def steer_and_generate(
        self, messages: list[dict[str, str]]
    ) -> tuple[str, dict[str, Any]]:
        """Generate a response with activation steering applied.

        Returns:
            Tuple of (response_text, metadata_dict).
        """
        if not self._use_native:
            return self._steer_assistant_axis(messages)
        return self._steer_native(messages)

    def _steer_native(
        self, messages: list[dict[str, str]]
    ) -> tuple[str, dict[str, Any]]:
        """Steer using native PyTorch forward hooks."""
        target_layer = self.model.target_layer
        layers = _get_model_layers(self.model.model)
        if target_layer >= len(layers):
            raise ValueError(
                f"Target layer {target_layer} exceeds model layer count {len(layers)}"
            )

        axis = self.axis
        coeff = self.coefficient

        def steering_hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Project onto axis and steer
            # h' = h + coeff * axis
            steering = coeff * axis.to(hidden.device, dtype=hidden.dtype)
            modified = hidden + steering.unsqueeze(0).unsqueeze(0)

            if self.capping_enabled and self._capping_threshold is not None:
                # Capping: h <- h - v * min(<h,v> - tau, 0)
                proj = torch.sum(modified * axis.to(modified.device, dtype=modified.dtype), dim=-1, keepdim=True)
                cap_term = torch.clamp(proj - self._capping_threshold, max=0.0)
                modified = modified - axis.to(modified.device, dtype=modified.dtype) * cap_term

            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified

        handle = layers[target_layer].register_forward_hook(steering_hook)
        try:
            response = self.model.generate(messages)
        finally:
            handle.remove()

        return response, {"steering_method": "native", "coefficient": coeff}

    def _steer_assistant_axis(
        self, messages: list[dict[str, str]]
    ) -> tuple[str, dict[str, Any]]:
        """Steer using assistant_axis.ActivationSteering."""
        from assistant_axis import ActivationSteering

        input_text = self.model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.model.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        with ActivationSteering(
            model=self.model.model,
            axis=self.axis,
            coefficient=self.coefficient,
            layer=self.model.target_layer,
        ):
            with torch.no_grad():
                outputs = self.model.model.generate(
                    **inputs,
                    max_new_tokens=self.model.config.max_new_tokens,
                    temperature=self.model.config.temperature,
                    do_sample=self.model.config.temperature > 0,
                    pad_token_id=self.model.tokenizer.pad_token_id,
                )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.model.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return response, {"steering_method": "assistant_axis", "coefficient": self.coefficient}

    def compute_capping_threshold(
        self, calibration_messages: list[list[dict[str, str]]] | None = None
    ) -> float:
        """Compute the capping threshold from calibration data.

        If no calibration messages are provided, uses a default set.
        """
        if calibration_messages is None:
            calibration_messages = [
                [{"role": "user", "content": "Hello, how are you?"}],
                [{"role": "user", "content": "What is the capital of France?"}],
                [{"role": "user", "content": "Tell me a joke."}],
                [{"role": "user", "content": "Explain photosynthesis."}],
                [{"role": "user", "content": "What are your capabilities?"}],
            ]

        projections = []
        for msgs in calibration_messages:
            input_text = self.model.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            inputs = self.model.tokenizer(input_text, return_tensors="pt")
            activations = self.model.get_activations(inputs["input_ids"], [self.model.target_layer])

            if self.model.target_layer in activations:
                act = activations[self.model.target_layer].mean(dim=1).squeeze(0).float()
                proj = torch.dot(act, self.axis.cpu()).item()
                projections.append(proj)

        if not projections:
            raise RuntimeError("Failed to compute capping threshold")

        # Percentile threshold
        projections_t = torch.tensor(projections)
        threshold = torch.quantile(projections_t, self.capping_percentile / 100.0).item()
        self._capping_threshold = threshold
        logger.info("Capping threshold set to %.4f (p%.0f)", threshold, self.capping_percentile)
        return threshold


def _get_model_layers(model):
    """Get transformer layers (same logic as models.py)."""
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "layers"):
            return inner.layers
        if hasattr(inner, "decoder") and hasattr(inner.decoder, "layers"):
            return inner.decoder.layers
    if hasattr(model, "transformer"):
        if hasattr(model.transformer, "h"):
            return model.transformer.h
        if hasattr(model.transformer, "blocks"):
            return model.transformer.blocks
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise ValueError(f"Cannot identify transformer layers for {type(model).__name__}")
