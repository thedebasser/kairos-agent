"""Kairos Agent — Centralized LLM Configuration.

Loads llm_config.yaml and exposes helpers that every agent uses to resolve
which model to call.  The ``use_local_llms`` toggle controls whether
local Ollama models are attempted at all.

Key behaviour:
  * ``use_local_llms = true``  → normal local→cloud fallback path.
  * ``use_local_llms = false`` → skip local, go straight to cloud.
  * Training data is **always** captured regardless of toggle.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config path — lives at repository root next to litellm_config.yaml
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).resolve().parents[4] / "llm_config.yaml"


# ---------------------------------------------------------------------------
# Dataclass-light container (plain dict is fine, but helpers are typed)
# ---------------------------------------------------------------------------


class StepConfig:
    """Resolved model configuration for a single pipeline step."""

    def __init__(self, raw: dict[str, Any], use_local: bool, cloud_fallback: bool = True) -> None:
        self._raw = raw
        self._use_local = use_local
        self._cloud_fallback = cloud_fallback

    # Aliases -----------------------------------------------------------------

    @property
    def litellm_alias_local(self) -> str | None:
        return self._raw.get("litellm_alias_local")

    @property
    def litellm_alias_cloud(self) -> str | None:
        return self._raw.get("litellm_alias_cloud")

    @property
    def call_pattern(self) -> str:
        pattern = self._raw.get("call_pattern", "direct")
        # When cloud fallback is disabled, downgrade quality_fallback → direct
        if pattern == "quality_fallback" and not self._cloud_fallback:
            return "direct"
        return pattern

    @property
    def local_model(self) -> str | None:
        return self._raw.get("local_model")

    @property
    def cloud_model(self) -> str | None:
        return self._raw.get("cloud_model")

    # Resolution helpers ------------------------------------------------------

    @property
    def should_try_local(self) -> bool:
        """Whether the local model should be attempted for this step."""
        return self._use_local and self.litellm_alias_local is not None

    def resolve_model(self) -> str:
        """Return the single model alias to use for a *direct* call.

        Resolution order:
        1. litellm_alias_local (when use_local_llms is true)
        2. litellm_alias_cloud (when cloud fallback is on)
        3. litellm_alias_local (last resort, regardless of use_local toggle)
        4. local_model raw value (e.g. ``ollama/glm-4.7-flash:latest``)
        """
        if self.should_try_local:
            return self.litellm_alias_local  # type: ignore[return-value]
        if self.litellm_alias_cloud and self._cloud_fallback:
            return self.litellm_alias_cloud
        # Cloud disabled or not configured — try local alias as last resort
        if self.litellm_alias_local:
            return self.litellm_alias_local
        # No litellm alias at all — fall back to the raw local_model identifier
        if self.local_model:
            return self.local_model
        raise ValueError(
            f"No model configured for step (cloud fallback={'on' if self._cloud_fallback else 'off'}): {self._raw}"
        )

    def resolve_primary_and_fallback(self) -> tuple[str, str]:
        """Return ``(primary, fallback)`` for a quality-fallback call.

        * cloud fallback off → (local_alias, local_alias) — no cloud
        * local enabled      → (local_alias, cloud_alias)
        * local disabled     → (cloud_alias, cloud_alias)
        """
        local = self.litellm_alias_local
        cloud = self.litellm_alias_cloud

        if not self._cloud_fallback:
            # Cloud disabled — both primary and fallback use local
            if local is None:
                raise ValueError(
                    f"Cloud fallback is off but no local model configured: {self._raw}"
                )
            return (local, local)

        if cloud is None:
            raise ValueError(
                f"quality_fallback step has no cloud model: {self._raw}"
            )
        if self.should_try_local:
            return (local, cloud)  # type: ignore[return-value]
        return (cloud, cloud)


# ---------------------------------------------------------------------------
# Singleton loader
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_raw_config() -> dict[str, Any]:
    """Load and cache the raw YAML config."""
    if not _CONFIG_PATH.exists():
        logger.warning("llm_config.yaml not found at %s — using defaults", _CONFIG_PATH)
        return {}
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _reload_config() -> dict[str, Any]:
    """Force-reload config (useful for tests)."""
    _load_raw_config.cache_clear()
    return _load_raw_config()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def use_local_llms() -> bool:
    """Return the global ``use_local_llms`` toggle value."""
    cfg = _load_raw_config()
    return bool(cfg.get("use_local_llms", False))


def enable_cloud_fallback() -> bool:
    """Whether cloud model fallbacks are allowed.

    When ``False``, quality_fallback steps behave as direct (local-only)
    and ``resolve_model()`` never returns a cloud alias.
    """
    cfg = _load_raw_config()
    return bool(cfg.get("enable_cloud_fallback", True))


def always_store_training_data() -> bool:
    """Whether to persist training data even when local is disabled."""
    cfg = _load_raw_config()
    return bool(cfg.get("always_store_training_data", True))


def get_step_config(step_name: str) -> StepConfig:
    """Get the resolved :class:`StepConfig` for a pipeline step.

    Args:
        step_name: Key from the ``steps:`` section of llm_config.yaml
                   (e.g. ``"category_selector"``, ``"simulation_param_adjustment"``).

    Raises:
        KeyError: if the step is not defined in the config.
    """
    cfg = _load_raw_config()
    steps = cfg.get("steps", {})
    if step_name not in steps:
        raise KeyError(
            f"Step '{step_name}' not found in llm_config.yaml. "
            f"Available steps: {list(steps.keys())}"
        )
    return StepConfig(steps[step_name], use_local=use_local_llms(), cloud_fallback=enable_cloud_fallback())


def get_ollama_base_url() -> str:
    """Return the Ollama base URL from config (env var takes precedence)."""
    import os

    env = os.environ.get("OLLAMA_BASE_URL")
    if env:
        return env
    cfg = _load_raw_config()
    return cfg.get("ollama", {}).get("base_url", "http://localhost:11434")


def get_thinking_config() -> dict[str, Any] | None:
    """Return the extended thinking params for Anthropic models, or *None*.

    When enabled, returns ``{"type": "enabled", "budget_tokens": N}``
    ready to pass straight into ``litellm.completion(thinking=...)``.
    """
    cfg = _load_raw_config()
    thinking = cfg.get("thinking", {})
    if not thinking.get("enabled", False):
        return None
    budget = int(thinking.get("budget_tokens", 10_000))
    return {"type": "enabled", "budget_tokens": budget}
