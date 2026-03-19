"""Kairos Agent — Model Capabilities Registry.

Provides a unified interface for model-specific behaviour that varies across
LLM providers and model families.  Instead of scattering ``if model == ...``
checks throughout the codebase, each model family gets a ``ModelCapabilities``
implementation that answers questions like:

  * Does this model support native thinking/reasoning tokens?
  * How do I extract thinking content from its response?
  * Is this a local or cloud model?
  * What does it cost per token?
  * What Instructor mode should I use?

**Usage**::

    from kairos.ai.llm.capabilities import get_capabilities

    caps = get_capabilities("ollama/qwen3.5:27b")
    if caps.supports_thinking:
        thinking = caps.extract_thinking(response)
    cost = caps.compute_cost(tokens_in=100, tokens_out=50)

Adding a new model family:
  1. Subclass ``ModelCapabilities``.
  2. Register it in ``_CAPABILITY_REGISTRY`` at the bottom of this file.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import instructor

logger = logging.getLogger(__name__)


# =============================================================================
# Data Types
# =============================================================================


@dataclass(frozen=True)
class TokenPricing:
    """Per-million-token pricing for a model.

    All values in USD.  Local models use (0.0, 0.0).
    """

    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0


class ModelType:
    """Enumeration of model deployment types."""

    LOCAL = "local"
    CLOUD = "cloud"


# =============================================================================
# Abstract Base
# =============================================================================


class ModelCapabilities(ABC):
    """Interface describing what a model family can do.

    Each provider/model family implements this to centralise all
    model-specific logic in one place.
    """

    # ── Identity ────────────────────────────────────────────────────

    @property
    @abstractmethod
    def family_name(self) -> str:
        """Human-readable family name (e.g. 'anthropic', 'qwen3')."""

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return ``ModelType.LOCAL`` or ``ModelType.CLOUD``."""

    # ── Thinking / Reasoning ────────────────────────────────────────

    @property
    @abstractmethod
    def supports_thinking(self) -> bool:
        """Whether this model can produce separate thinking/reasoning tokens."""

    @abstractmethod
    def get_thinking_param(self) -> dict[str, Any] | None:
        """Return the provider-specific thinking parameter dict.

        For Anthropic: ``{"type": "enabled", "budget_tokens": N}``
        For Ollama thinking models: handled at the extraction layer.
        Returns None if thinking is not supported or not enabled.
        """

    @abstractmethod
    def extract_thinking(self, response: Any) -> str | None:
        """Extract thinking/reasoning content from a raw LLM response.

        Returns the thinking text, or None if not present.
        """

    # ── Instructor / Structured Output ──────────────────────────────

    @abstractmethod
    def get_instructor_mode(self, thinking_enabled: bool = False) -> instructor.Mode | None:
        """Return the Instructor parsing mode for this model.

        Returns None to use the default (TOOLS mode).
        When thinking is enabled, some providers need a different mode
        (e.g. Anthropic requires JSON mode because TOOLS mode uses
        tool_choice which conflicts with thinking).
        """

    def get_extra_call_params(self, structured_output: bool = False) -> dict[str, Any]:
        """Return extra kwargs to pass to litellm.completion for this model.

        Override in subclasses that need special parameters.
        For example, Ollama thinking models need ``think=False`` when
        requesting structured output to prevent litellm from returning
        empty content.
        """
        return {}

    # ── Pricing ─────────────────────────────────────────────────────

    @abstractmethod
    def get_pricing(self, resolved_model: str) -> TokenPricing:
        """Return token pricing for cost calculation.

        Args:
            resolved_model: The fully resolved model string
                            (e.g. 'claude-sonnet-4-6', 'ollama/qwen3.5:27b').
        """

    def compute_cost(
        self,
        tokens_in: int,
        tokens_out: int,
        resolved_model: str = "",
    ) -> float:
        """Compute USD cost for a call.  Convenience wrapper around ``get_pricing``."""
        pricing = self.get_pricing(resolved_model)
        return (tokens_in * pricing.input_per_mtok + tokens_out * pricing.output_per_mtok) / 1_000_000

    # ── Misc Capabilities ───────────────────────────────────────────

    @property
    def supports_vision(self) -> bool:
        """Whether this model accepts image inputs."""
        return False

    @property
    def supports_audio(self) -> bool:
        """Whether this model accepts audio inputs."""
        return False

    @property
    def supports_tool_calling(self) -> bool:
        """Whether this model supports function/tool calling."""
        return True

    @property
    def supports_json_mode(self) -> bool:
        """Whether this model supports native JSON output mode."""
        return True


# =============================================================================
# Anthropic (Claude) Implementation
# =============================================================================


class AnthropicCapabilities(ModelCapabilities):
    """Capabilities for Anthropic Claude models (cloud)."""

    @property
    def family_name(self) -> str:
        return "anthropic"

    @property
    def model_type(self) -> str:
        return ModelType.CLOUD

    @property
    def supports_thinking(self) -> bool:
        return True

    def get_thinking_param(self) -> dict[str, Any] | None:
        from kairos.ai.llm.config import get_thinking_config
        return get_thinking_config()

    def extract_thinking(self, response: Any) -> str | None:
        """Extract thinking from Anthropic's native fields.

        LiteLLM exposes Anthropic thinking via:
          - message.reasoning_content  (plain string, preferred)
          - message.thinking_blocks    (list of ChatCompletionThinkingBlock)
        """
        msg = response.choices[0].message
        # Prefer the flat string
        if getattr(msg, "reasoning_content", None):
            return msg.reasoning_content
        # Fall back to thinking_blocks
        blocks = getattr(msg, "thinking_blocks", None)
        if blocks:
            parts = [b.thinking for b in blocks if hasattr(b, "thinking")]
            if parts:
                return "\n".join(parts)
        return None

    def get_instructor_mode(self, thinking_enabled: bool = False) -> instructor.Mode | None:
        # Anthropic + thinking requires JSON mode (TOOLS mode uses tool_choice
        # which Anthropic rejects when thinking is on).
        if thinking_enabled:
            return instructor.Mode.JSON
        return None

    def get_pricing(self, resolved_model: str) -> TokenPricing:
        m = resolved_model.lower()
        if "opus" in m:
            return TokenPricing(15.00, 75.00)
        if "haiku" in m:
            return TokenPricing(0.25, 1.25)
        # Sonnet (default for all Claude Sonnet variants)
        return TokenPricing(3.00, 15.00)

    @property
    def supports_vision(self) -> bool:
        return True


# =============================================================================
# Qwen3 Family (thinking-capable, local via Ollama)
# =============================================================================


class Qwen3Capabilities(ModelCapabilities):
    """Capabilities for Qwen3 family models (Qwen3, Qwen3.5, Qwen3-Coder, Qwen3-VL).

    These models support hybrid thinking mode.  When served via Ollama,
    thinking content appears as ``<think>...</think>`` tags inline in the
    response content (Ollama's OpenAI-compat endpoint embeds it there
    rather than in a separate field).
    """

    @property
    def family_name(self) -> str:
        return "qwen3"

    @property
    def model_type(self) -> str:
        return ModelType.LOCAL

    @property
    def supports_thinking(self) -> bool:
        return True

    def get_thinking_param(self) -> dict[str, Any] | None:
        # Qwen3 thinking is enabled by default when the model runs in
        # thinking mode.  We don't need to pass a special parameter to
        # LiteLLM — the model thinks by default.  Return None since
        # there's no provider-level toggle needed.
        return None

    def extract_thinking(self, response: Any) -> str | None:
        """Extract thinking from ``<think>...</think>`` tags in content.

        Qwen3 models via Ollama embed thinking in the response content
        wrapped in ``<think>`` tags.  We parse them out.
        """
        msg = response.choices[0].message
        content = getattr(msg, "content", "") or ""

        # First check if LiteLLM exposed reasoning_content natively
        # (future-proofing for when LiteLLM adds Ollama reasoning support)
        if getattr(msg, "reasoning_content", None):
            return msg.reasoning_content

        # Parse <think>...</think> tags from content
        return self._parse_think_tags(content)

    @staticmethod
    def _parse_think_tags(content: str) -> str | None:
        """Extract text between ``<think>`` and ``</think>`` tags.

        Handles multiple think blocks and returns them joined.
        """
        pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        matches = pattern.findall(content)
        if matches:
            return "\n".join(m.strip() for m in matches if m.strip())
        return None

    @staticmethod
    def strip_think_tags(content: str) -> str:
        """Remove ``<think>...</think>`` blocks from content.

        Use this to get the clean response content after extracting thinking.
        """
        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    def get_instructor_mode(self, thinking_enabled: bool = False) -> instructor.Mode | None:
        # Ollama models work best with JSON mode for structured output
        return instructor.Mode.JSON

    def get_extra_call_params(self, structured_output: bool = False) -> dict[str, Any]:
        """Extra kwargs to pass to litellm.completion for this model.

        For Ollama thinking models (Qwen3/Qwen3.5), we must disable
        thinking when requesting structured output.  Otherwise the model
        puts ALL tokens into ``<think>`` blocks and litellm returns
        empty ``content``.
        """
        if structured_output and self.supports_thinking:
            return {"think": False}
        return {}

    def get_pricing(self, resolved_model: str) -> TokenPricing:
        return TokenPricing(0.0, 0.0)  # Local model — free

    @property
    def supports_vision(self) -> bool:
        # Qwen3-VL variants support vision
        return "vl" in self._model_hint.lower() if hasattr(self, "_model_hint") else False

    @property
    def supports_audio(self) -> bool:
        # Qwen3-Omni variants support audio
        return "omni" in self._model_hint.lower() if hasattr(self, "_model_hint") else False


# =============================================================================
# Qwen2.5 Family (no thinking, local via Ollama)
# =============================================================================


class Qwen25Capabilities(ModelCapabilities):
    """Capabilities for Qwen2.5 family models (pre-thinking architecture).

    Includes Qwen2.5-Omni (audio) and Qwen2.5-Coder.
    """

    @property
    def family_name(self) -> str:
        return "qwen2.5"

    @property
    def model_type(self) -> str:
        return ModelType.LOCAL

    @property
    def supports_thinking(self) -> bool:
        return False

    def get_thinking_param(self) -> dict[str, Any] | None:
        return None

    def extract_thinking(self, response: Any) -> str | None:
        return None  # No thinking support

    def get_instructor_mode(self, thinking_enabled: bool = False) -> instructor.Mode | None:
        return instructor.Mode.JSON

    def get_pricing(self, resolved_model: str) -> TokenPricing:
        return TokenPricing(0.0, 0.0)  # Local model — free

    @property
    def supports_audio(self) -> bool:
        return True  # Qwen2.5-Omni handles audio


# =============================================================================
# Devstral (Mistral coding models, local via Ollama)
# =============================================================================


class DevstralCapabilities(ModelCapabilities):
    """Capabilities for Mistral Devstral models (local coding models)."""

    @property
    def family_name(self) -> str:
        return "devstral"

    @property
    def model_type(self) -> str:
        return ModelType.LOCAL

    @property
    def supports_thinking(self) -> bool:
        return False  # Devstral is a coding model, not a reasoning model

    def get_thinking_param(self) -> dict[str, Any] | None:
        return None

    def extract_thinking(self, response: Any) -> str | None:
        return None

    def get_instructor_mode(self, thinking_enabled: bool = False) -> instructor.Mode | None:
        return instructor.Mode.JSON

    def get_pricing(self, resolved_model: str) -> TokenPricing:
        return TokenPricing(0.0, 0.0)  # Local model — free


# =============================================================================
# OpenAI GPT (cloud)
# =============================================================================


class OpenAICapabilities(ModelCapabilities):
    """Capabilities for OpenAI GPT models (cloud)."""

    @property
    def family_name(self) -> str:
        return "openai"

    @property
    def model_type(self) -> str:
        return ModelType.CLOUD

    @property
    def supports_thinking(self) -> bool:
        return False  # GPT-4o doesn't have reasoning mode (o1/o3 do, but we don't use them)

    def get_thinking_param(self) -> dict[str, Any] | None:
        return None

    def extract_thinking(self, response: Any) -> str | None:
        return None

    def get_instructor_mode(self, thinking_enabled: bool = False) -> instructor.Mode | None:
        return None  # Default TOOLS mode works fine

    def get_pricing(self, resolved_model: str) -> TokenPricing:
        m = resolved_model.lower()
        if "gpt-4o-mini" in m:
            return TokenPricing(0.15, 0.60)
        if "gpt-4o" in m:
            return TokenPricing(2.50, 10.00)
        return TokenPricing(2.50, 10.00)  # Default to GPT-4o pricing

    @property
    def supports_vision(self) -> bool:
        return True


# =============================================================================
# Default / Legacy Ollama (Mistral, Llama, Moondream, etc.)
# =============================================================================


class OllamaDefaultCapabilities(ModelCapabilities):
    """Default capabilities for Ollama models not in a specific family.

    Covers Mistral, Llama, Moondream, and other models that don't
    have thinking support or special features.
    """

    @property
    def family_name(self) -> str:
        return "ollama"

    @property
    def model_type(self) -> str:
        return ModelType.LOCAL

    @property
    def supports_thinking(self) -> bool:
        return False

    def get_thinking_param(self) -> dict[str, Any] | None:
        return None

    def extract_thinking(self, response: Any) -> str | None:
        return None

    def get_instructor_mode(self, thinking_enabled: bool = False) -> instructor.Mode | None:
        return None  # Default TOOLS mode

    def get_pricing(self, resolved_model: str) -> TokenPricing:
        return TokenPricing(0.0, 0.0)  # Local model — free


# =============================================================================
# Registry & Factory
# =============================================================================


# Ordered list of (pattern, capabilities_instance) pairs.
# The first matching pattern wins.  More specific patterns come first.
_CAPABILITY_REGISTRY: list[tuple[re.Pattern[str], ModelCapabilities]] = [
    # Anthropic / Claude
    (re.compile(r"^(anthropic/|claude[-_])", re.IGNORECASE), AnthropicCapabilities()),

    # Qwen3 family (includes Qwen3, Qwen3.5, Qwen3-VL, Qwen3-Coder, Qwen3-Omni)
    (re.compile(r"qwen3", re.IGNORECASE), Qwen3Capabilities()),

    # Qwen2.5 family (Qwen2.5-Omni, Qwen2.5-Coder)
    (re.compile(r"qwen2\.5", re.IGNORECASE), Qwen25Capabilities()),

    # Devstral (Mistral coding models)
    (re.compile(r"devstral", re.IGNORECASE), DevstralCapabilities()),

    # OpenAI GPT
    (re.compile(r"^(openai/|gpt[-_])", re.IGNORECASE), OpenAICapabilities()),

    # Ollama catch-all (Mistral, Llama, Moondream, etc.)
    (re.compile(r"^ollama/", re.IGNORECASE), OllamaDefaultCapabilities()),
]

# Singleton fallback for completely unknown models
_FALLBACK_CAPABILITIES = OllamaDefaultCapabilities()


def get_capabilities(resolved_model: str) -> ModelCapabilities:
    """Look up the capabilities handler for a resolved model string.

    Matches against the registry in order (most specific first).
    Falls back to ``OllamaDefaultCapabilities`` if nothing matches.

    Args:
        resolved_model: The fully resolved model name
                        (e.g. 'claude-sonnet-4-6', 'ollama/qwen3.5:27b').

    Returns:
        The appropriate ``ModelCapabilities`` instance.
    """
    for pattern, caps in _CAPABILITY_REGISTRY:
        if pattern.search(resolved_model):
            return caps

    logger.debug("No capability match for '%s' — using default", resolved_model)
    return _FALLBACK_CAPABILITIES
