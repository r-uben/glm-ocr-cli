"""Configuration management for GLM-OCR CLI."""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="GLM_OCR_",
        case_sensitive=False,
        # Tolerate stale/removed keys in a user's .env (e.g. the dropped
        # GLM_OCR_OUTPUT_DIR / GLM_OCR_INCLUDE_METADATA fields). pydantic-settings
        # defaults to extra="forbid"; since Settings() runs at import time, a
        # forbidden extra key would raise a ValidationError on EVERY command
        # (even --help) and crash the CLI for any existing user on upgrade.
        extra="ignore",
    )

    # Backend selection
    backend: Literal["ollama", "transformers", "vllm"] = Field(
        default="ollama",
        description="Backend: 'ollama' (local, default), 'transformers', or 'vllm'",
    )

    # Model configuration
    model_name: str = Field(
        default="glm-ocr",
        description="Model name (glm-ocr for Ollama, glm-ocr or zai-org/GLM-OCR for vLLM)",
    )
    ollama_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API URL",
    )
    vllm_base_url: str = Field(
        default="http://localhost:8080/v1",
        description="vLLM OpenAI-compatible API URL",
    )

    # Image preprocessing
    max_dimension: int = Field(
        default=2048,
        description="Maximum image dimension (width or height). Set to 0 to disable.",
    )

    # Output configuration. The output ROOT defaults to <input-parent>/ocr/
    # (computed by ocr-output-contract); it is not a configurable setting.
    extract_images: bool = Field(
        default=False,
        description="Extract and save full-page rasters under images/",
    )

    # Retry configuration
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries for transient backend errors",
    )
    retry_delay: float = Field(
        default=1.0,
        description="Base delay in seconds between retries (exponential backoff)",
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose output",
    )


# Global settings instance
settings = Settings()
