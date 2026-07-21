import json
from pathlib import Path

import pytest

from config.conversation_context_settings import ConversationContextSettings
from config.model_settings import ModelSettings
from rag.context_compaction import (
    ConversationCompactionError,
    CurrentMessageTooLargeError,
    estimate_context_tokens,
    parse_compaction_summary,
    prepare_conversation_context,
)
from rag.conversation import ConversationTurn
from rag.conversation_store import (
    append_conversation_turn,
    create_conversation,
    get_conversation,
    initialize_conversation_store,
    load_conversation_state,
)
from rag.llm_client import LLMRequestError


MODEL_SETTINGS = ModelSettings("key", "https://example.test", "model")


def make_store(tmp_path: Path) -> tuple[Path, str]:
    database_path = tmp_path / "conversation.db"
    initialize_conversation_store(database_path)
    conversation = create_conversation(database_path)
    return database_path, conversation.conversation_id


def valid_summary(goal: str = "保留长期目标") -> str:
    return json.dumps(
        {
            "user_goals": [goal],
            "confirmed_requirements": ["原始消息永久保留"],
            "decisions": ["达到 Token 阈值后批量压缩"],
            "important_context": [],
            "open_questions": [],
        },
        ensure_ascii=False,
    )


def test_estimates_chinese_and_english_context_tokens() -> None:
    turns = (ConversationTurn("中文目标", "an English response"),)

    assert estimate_context_tokens(None, turns) > 12


def test_rejects_invalid_summary_schema() -> None:
    with pytest.raises(ConversationCompactionError, match="schema"):
        parse_compaction_summary('{"user_goals": []}')


def test_skips_compaction_below_token_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path, conversation_id = make_store(tmp_path)
    append_conversation_turn(
        database_path,
        conversation_id,
        user_message="short question",
        assistant_message="short answer",
    )
    monkeypatch.setattr(
        "rag.context_compaction.generate_text",
        lambda *args, **kwargs: pytest.fail("model should not be called"),
    )

    result = prepare_conversation_context(
        database_path,
        conversation_id,
        settings=ConversationContextSettings(
            token_threshold=1_000,
            target_tokens=500,
        ),
        model_settings=MODEL_SETTINGS,
    )

    assert not result.compacted
    assert len(result.state.uncompacted_turns) == 1


def test_rejects_a_current_message_that_alone_exceeds_the_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path, conversation_id = make_store(tmp_path)
    monkeypatch.setattr(
        "rag.context_compaction.generate_text",
        lambda *args, **kwargs: pytest.fail(
            "an oversized message must be rejected before any model call"
        ),
    )

    with pytest.raises(CurrentMessageTooLargeError):
        prepare_conversation_context(
            database_path,
            conversation_id,
            current_message="问" * 2_000,
            settings=ConversationContextSettings(
                token_threshold=1_000,
                target_tokens=500,
            ),
            model_settings=MODEL_SETTINGS,
        )


def test_compacts_oldest_batch_and_preserves_all_raw_messages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path, conversation_id = make_store(tmp_path)
    for index in range(4):
        append_conversation_turn(
            database_path,
            conversation_id,
            user_message=f"user original {index} " + "目标" * 30,
            assistant_message=f"assistant original {index} " + "回答" * 30,
            paper_ids=(f"paper-{index}",),
        )
    observed_messages: list[list[dict[str, str]]] = []

    def fake_generate(messages: list[dict[str, str]], **_: object) -> str:
        observed_messages.append(messages)
        return valid_summary()

    monkeypatch.setattr("rag.context_compaction.generate_text", fake_generate)

    result = prepare_conversation_context(
        database_path,
        conversation_id,
        settings=ConversationContextSettings(
            token_threshold=120,
            target_tokens=70,
        ),
        model_settings=MODEL_SETTINGS,
    )

    stored = get_conversation(database_path, conversation_id)
    assert result.compacted
    assert result.compacted_turn_count >= 2
    assert result.estimated_tokens_after < result.estimated_tokens_before
    assert [turn.user_message for turn in stored.turns] == [
        f"user original {index} " + "目标" * 30 for index in range(4)
    ]
    request_payload = json.loads(observed_messages[0][1]["content"])
    assert request_payload["turns_to_compact"][0]["user_message"].startswith(
        "user original 0"
    )
    assert "abstract" not in observed_messages[0][1]["content"]
    assert result.state.context_summary is not None
    assert len(result.state.uncompacted_turns) < 4


def test_multiple_batches_commit_only_after_every_summary_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path, conversation_id = make_store(tmp_path)
    for index in range(8):
        append_conversation_turn(
            database_path,
            conversation_id,
            user_message=f"user {index} " + "目标" * 30,
            assistant_message=f"assistant {index} " + "回答" * 30,
        )
    calls = 0

    def fail_second_batch(*_: object, **__: object) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise LLMRequestError("second batch failed")
        return valid_summary()

    monkeypatch.setattr(
        "rag.context_compaction.COMPACTION_BATCH_INPUT_TOKEN_LIMIT",
        250,
    )
    monkeypatch.setattr(
        "rag.context_compaction.generate_text",
        fail_second_batch,
    )

    with pytest.raises(LLMRequestError, match="second batch"):
        prepare_conversation_context(
            database_path,
            conversation_id,
            settings=ConversationContextSettings(
                token_threshold=200,
                target_tokens=100,
            ),
            model_settings=MODEL_SETTINGS,
        )

    state = load_conversation_state(database_path, conversation_id)
    assert calls == 2
    assert state.context_summary is None
    assert state.compacted_through_turn_id == 0
    assert len(state.uncompacted_turns) == 8
