"""Kairos Agent — Application Configuration.

Uses pydantic-settings for type-safe environment variable loading.
All services run on localhost — no auth system for POC.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- Anthropic (Claude) ---
    anthropic_api_key: str = ""

    # --- PostgreSQL ---
    postgres_host: str = "localhost"
    postgres_port: int = 5433
    postgres_db: str = "kairos"
    postgres_user: str = "kairos"
    postgres_password: str = "changeme"
    database_url: str = "postgresql+asyncpg://kairos:changeme@localhost:5433/kairos"
    database_url_sync: str = "postgresql://kairos:changeme@localhost:5433/kairos"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Ollama ---
    ollama_base_url: str = "http://localhost:11434"

    # --- LiteLLM ---
    litellm_config_path: str = "litellm_config.yaml"

    # --- Langfuse ---
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    # --- Discord ---
    discord_webhook_url: str = ""

    # --- Upload-Post ---
    upload_post_api_key: str = ""

    # --- Pipeline Defaults ---
    default_pipeline: str = "physics"
    target_duration_sec: int = 65
    target_duration_min_sec: int = 62
    target_duration_max_sec: int = 68
    target_fps: int = 30
    target_resolution: str = "1080x1920"
    max_simulation_iterations: int = 5

    # --- Sandbox ---
    sandbox_timeout_sec: int = 300
    sandbox_memory_limit: str = "4g"
    sandbox_cpu_limit: int = 2
    sandbox_image: str = "simulation-sandbox:latest"

    # --- Paths ---
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    knowledge_dir: Path = Field(default_factory=lambda: Path("knowledge"))
    music_dir: Path = Field(default_factory=lambda: Path("music"))
    output_dir: Path = Field(default_factory=lambda: Path("output"))

    # --- Cost Alerts ---
    cost_alert_threshold_usd: float = 0.30  # 7-day rolling average


_settings_instance: Settings | None = None


def get_settings() -> Settings:
    """Get application settings (cached singleton)."""
    global _settings_instance  # noqa: PLW0603
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
