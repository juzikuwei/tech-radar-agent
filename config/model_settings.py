"""Load and validate language-model configuration."""

from dataclasses import dataclass
import os
from pathlib import Path

from config.environment import DEFAULT_ENV_PATH, load_repository_env


@dataclass(frozen=True)
class ModelSettings:
    """Settings shared by model clients and application entry points."""

    api_key: str
    base_url: str
    model: str

    @classmethod
    def from_env(cls, env_path: Path = DEFAULT_ENV_PATH) -> "ModelSettings":
        """Read model settings from the repository's ignored `.env` file."""
        load_repository_env(env_path)
        values = {
            "api_key": os.getenv("LLM_API_KEY", "").strip(),
            "base_url": os.getenv("LLM_BASE_URL", "").strip(),
            "model": os.getenv("LLM_MODEL", "").strip(),
        }
        missing = [name for name, value in values.items() if not value]
        if missing:
            names = ", ".join(f"LLM_{name.upper()}" for name in missing)
            raise ValueError(f"Missing required environment variables: {names}")
        if values["api_key"] == "replace_with_your_deepseek_api_key":
            raise ValueError("Replace the placeholder LLM_API_KEY in .env")
        return cls(**values)
