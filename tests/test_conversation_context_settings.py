from pathlib import Path

import pytest

from config.conversation_context_settings import ConversationContextSettings


def test_loads_optional_context_thresholds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CONVERSATION_CONTEXT_TOKEN_THRESHOLD", raising=False)
    monkeypatch.delenv("CONVERSATION_CONTEXT_TARGET_TOKENS", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text(
        "CONVERSATION_CONTEXT_TOKEN_THRESHOLD=9000\n"
        "CONVERSATION_CONTEXT_TARGET_TOKENS=6000\n",
        encoding="utf-8",
    )

    settings = ConversationContextSettings.from_env(env_path)

    assert settings.token_threshold == 9000
    assert settings.target_tokens == 6000


def test_rejects_target_at_or_above_threshold() -> None:
    with pytest.raises(ValueError, match="less than"):
        ConversationContextSettings(token_threshold=100, target_tokens=100)
