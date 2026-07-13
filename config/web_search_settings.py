"""Load optional external web-search configuration."""

from dataclasses import dataclass
import os
from pathlib import Path

from config.environment import DEFAULT_ENV_PATH, load_repository_env


@dataclass(frozen=True)
class WebSearchSettings:
    """Settings for the optional research-agent web-search tool."""

    api_key: str

    @classmethod
    def from_env(
        cls, env_path: Path = DEFAULT_ENV_PATH
    ) -> "WebSearchSettings | None":
        """Return settings when TAVILY_API_KEY is set; None disables the tool."""
        load_repository_env(env_path)
        api_key = os.getenv("TAVILY_API_KEY", "").strip()
        if not api_key:
            return None
        return cls(api_key=api_key)
