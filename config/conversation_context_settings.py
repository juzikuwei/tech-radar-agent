"""Load token-budget settings for persistent conversation context."""

from dataclasses import dataclass
import os
from pathlib import Path

from config.environment import DEFAULT_ENV_PATH, load_repository_env


DEFAULT_CONTEXT_TOKEN_THRESHOLD = 12_000
DEFAULT_CONTEXT_TARGET_TOKENS = 8_000


@dataclass(frozen=True)
class ConversationContextSettings:
    """Token thresholds that trigger and size one context compaction."""

    token_threshold: int = DEFAULT_CONTEXT_TOKEN_THRESHOLD
    target_tokens: int = DEFAULT_CONTEXT_TARGET_TOKENS

    def __post_init__(self) -> None:
        if self.token_threshold <= 0:
            raise ValueError("token_threshold must be greater than zero")
        if self.target_tokens <= 0:
            raise ValueError("target_tokens must be greater than zero")
        if self.target_tokens >= self.token_threshold:
            raise ValueError("target_tokens must be less than token_threshold")

    @classmethod
    def from_env(
        cls,
        env_path: Path = DEFAULT_ENV_PATH,
    ) -> "ConversationContextSettings":
        """Read optional context-compaction thresholds from the environment."""
        load_repository_env(env_path)
        return cls(
            token_threshold=_positive_int(
                "CONVERSATION_CONTEXT_TOKEN_THRESHOLD",
                DEFAULT_CONTEXT_TOKEN_THRESHOLD,
            ),
            target_tokens=_positive_int(
                "CONVERSATION_CONTEXT_TARGET_TOKENS",
                DEFAULT_CONTEXT_TARGET_TOKENS,
            ),
        )


def _positive_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value
