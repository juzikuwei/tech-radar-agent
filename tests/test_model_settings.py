from pathlib import Path

import pytest

from config.model_settings import ModelSettings


def test_loads_model_settings_from_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text(
        "LLM_API_KEY=test-key\n"
        "LLM_BASE_URL=https://api.deepseek.com\n"
        "LLM_MODEL=deepseek-chat\n",
        encoding="utf-8",
    )

    settings = ModelSettings.from_env(env_path)

    assert settings.api_key == "test-key"
    assert settings.model == "deepseek-chat"


def test_rejects_missing_required_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text(
        "LLM_BASE_URL=https://api.deepseek.com\n"
        "LLM_MODEL=deepseek-chat\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="LLM_API_KEY"):
        ModelSettings.from_env(env_path)
