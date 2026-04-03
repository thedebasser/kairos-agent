"""Kairos Agent — Application Configuration.

Uses pydantic-settings for type-safe environment variable loading.
All services run on localhost — no auth system for POC.
"""

from __future__ import annotations

import platform
import shutil
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _resolve_ffmpeg(binary: str = "ffmpeg") -> str:
    """Resolve the ffmpeg/ffprobe binary path across platforms.

    Centralised resolution (Finding 5.4 — kairos_architectural_review):
    1. Environment variable FFMPEG_PATH / FFPROBE_PATH
    2. shutil.which() (system PATH)
    3. Platform-specific known locations (Windows WinGet, Chocolatey, etc.)
    4. Falls back to the bare binary name (will fail later with a clear error).
    """
    import os

    env_var = f"{binary.upper()}_PATH"
    env_path = os.environ.get(env_var)
    if env_path and Path(env_path).exists():
        return env_path

    which_path = shutil.which(binary)
    if which_path:
        return which_path

    # Platform-specific known install locations
    if platform.system() == "Windows":
        candidates = [
            Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
            / "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
            / "ffmpeg-8.0.1-full_build/bin" / f"{binary}.exe",
            Path(f"C:/ProgramData/chocolatey/bin/{binary}.exe"),
            Path(f"C:/ffmpeg/bin/{binary}.exe"),
        ]
    else:
        candidates = [
            Path(f"/usr/bin/{binary}"),
            Path(f"/usr/local/bin/{binary}"),
            Path(f"/opt/homebrew/bin/{binary}"),
        ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return binary  # bare name — will produce a clear FileNotFoundError when called


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Anthropic (Claude) ---
    anthropic_api_key: str = ""

    # --- PostgreSQL ---
    postgres_host: str = "localhost"
    postgres_port: int = 5433
    postgres_db: str = "kairos"
    postgres_user: str = "kairos"
    postgres_password: str = "changeme"

    # Derived at validation time — see model_validator below
    database_url: str = ""
    database_url_sync: str = ""

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Ollama ---
    ollama_base_url: str = "http://localhost:11434"

    # --- LiteLLM ---
    litellm_config_path: str = "litellm_config.yaml"

    # --- Langfuse ---
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3001"

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
    sandbox_image: str = "kairos-sandbox:latest"

    # --- Paths (anchored to project_root — Finding 5.5) ---
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    knowledge_dir: Path | None = Field(default=None)
    music_dir: Path | None = Field(default=None)
    output_dir: Path | None = Field(default=None)

    # --- FFmpeg (centralised — Finding 5.4) ---
    ffmpeg_path: str = Field(default="")
    ffprobe_path: str = Field(default="")

    # --- Environment / Theming ---
    freesound_api_key: str = ""  # Optional: enables collision SFX

    # --- Cost Alerts ---
    cost_alert_threshold_usd: float = 0.30  # 7-day rolling average

    # --- Learning Loop (AI review §1-§6) ---
    learning_loop_enabled: bool = Field(
        default=True,
        description=(
            "When True, successful runs auto-populate TrainingExample rows "
            "and category knowledge.  Examples remain gated behind the "
            "'verified' DB flag until an operator promotes them."
        ),
    )

    # --- Cache ---
    max_cache_size_mb: int = 2048  # LRU eviction threshold for global cache

    @model_validator(mode="after")
    def _derive_computed_fields(self) -> "Settings":
        """Derive database URLs from components and anchor paths to project_root.

        Findings 5.5 + config path anchoring + Finding 5.4 FFmpeg consolidation.
        """
        # Derive database_url from components if not explicitly set
        if not self.database_url:
            self.database_url = (
                f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            )
        if not self.database_url_sync:
            self.database_url_sync = (
                f"postgresql://{self.postgres_user}:{self.postgres_password}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            )

        # Anchor relative paths to project_root
        if self.knowledge_dir is None:
            self.knowledge_dir = self.project_root / "knowledge"
        if self.music_dir is None:
            self.music_dir = self.project_root / "assets" / "music"
        if self.output_dir is None:
            self.output_dir = self.project_root / "output"

        # Resolve FFmpeg paths once at startup
        if not self.ffmpeg_path:
            self.ffmpeg_path = _resolve_ffmpeg("ffmpeg")
        if not self.ffprobe_path:
            self.ffprobe_path = _resolve_ffmpeg("ffprobe")

        return self


_settings_instance: Settings | None = None


def get_settings() -> Settings:
    """Get application settings (cached singleton)."""
    global _settings_instance  # noqa: PLW0603
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
