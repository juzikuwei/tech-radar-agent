from pathlib import Path

import pytest

from rag.conversation import MAX_CONVERSATION_TURNS, MAX_STORED_TURNS
from rag.conversation_store import (
    ConversationNotFoundError,
    ConversationTurnLimitError,
    append_conversation_turn,
    create_conversation,
    delete_conversation,
    get_conversation,
    initialize_conversation_store,
    list_conversations,
    load_conversation_state,
)


def make_store(tmp_path: Path) -> Path:
    database_path = tmp_path / "conversations.db"
    initialize_conversation_store(database_path)
    return database_path


def test_creates_lists_loads_and_deletes_conversation(tmp_path: Path) -> None:
    database_path = make_store(tmp_path)
    created = create_conversation(database_path)

    stored_turn = append_conversation_turn(
        database_path,
        created.conversation_id,
        user_message="  如何定位 Agent 故障？  ",
        assistant_message="  从最早失败步骤开始。  ",
        paper_ids=("2607.00001", "2607.00001", "2607.00002"),
    )

    summaries = list_conversations(database_path)
    conversation = get_conversation(database_path, created.conversation_id)
    assert summaries[0].turn_count == 1
    assert summaries[0].title == "如何定位 Agent 故障？"
    assert conversation.summary == summaries[0]
    assert conversation.turns == (stored_turn,)
    assert stored_turn.paper_ids == ("2607.00001", "2607.00002")

    delete_conversation(database_path, created.conversation_id)
    assert list_conversations(database_path) == []
    with pytest.raises(ConversationNotFoundError):
        get_conversation(database_path, created.conversation_id)


def test_loads_only_recent_model_window_and_latest_evidence(tmp_path: Path) -> None:
    database_path = make_store(tmp_path)
    conversation = create_conversation(database_path)
    for index in range(MAX_CONVERSATION_TURNS + 2):
        append_conversation_turn(
            database_path,
            conversation.conversation_id,
            user_message=f"question {index}",
            assistant_message=f"answer {index}",
            paper_ids=(f"paper-{index}",),
        )

    state = load_conversation_state(database_path, conversation.conversation_id)

    assert state.summary.turn_count == MAX_CONVERSATION_TURNS + 2
    assert [turn.user_message for turn in state.recent_turns] == [
        f"question {index}" for index in range(2, MAX_CONVERSATION_TURNS + 2)
    ]
    assert state.active_evidence_ids == (f"paper-{MAX_CONVERSATION_TURNS + 1}",)
    assert state.recent_turns[-1].evidence_ids == (
        f"paper-{MAX_CONVERSATION_TURNS + 1}",
    )


def test_conversation_response_preserves_active_research_evidence(
    tmp_path: Path,
) -> None:
    database_path = make_store(tmp_path)
    conversation = create_conversation(database_path)
    append_conversation_turn(
        database_path,
        conversation.conversation_id,
        user_message="research question",
        assistant_message="research answer",
        paper_ids=("paper-1", "paper-2"),
    )

    direct_turn = append_conversation_turn(
        database_path,
        conversation.conversation_id,
        user_message="谢谢",
        assistant_message="不客气。",
        response_kind="conversation",
    )
    state = load_conversation_state(database_path, conversation.conversation_id)

    assert direct_turn.paper_ids == ()
    assert direct_turn.response_kind == "conversation"
    assert state.active_evidence_ids == ("paper-1", "paper-2")
    assert state.recent_turns[-1].evidence_ids == ()


def test_uses_first_question_as_a_fifty_character_title(tmp_path: Path) -> None:
    database_path = make_store(tmp_path)
    conversation = create_conversation(database_path)
    question = "标题" * 30

    append_conversation_turn(
        database_path,
        conversation.conversation_id,
        user_message=question,
        assistant_message="answer",
    )

    assert get_conversation(database_path, conversation.conversation_id).summary.title == (
        question[:50]
    )


def test_rejects_turn_after_one_hundred_complete_turns(tmp_path: Path) -> None:
    database_path = make_store(tmp_path)
    conversation = create_conversation(database_path)
    for index in range(MAX_STORED_TURNS):
        append_conversation_turn(
            database_path,
            conversation.conversation_id,
            user_message=f"question {index}",
            assistant_message="answer",
        )

    with pytest.raises(ConversationTurnLimitError):
        append_conversation_turn(
            database_path,
            conversation.conversation_id,
            user_message="one too many",
            assistant_message="answer",
        )

    assert get_conversation(
        database_path,
        conversation.conversation_id,
    ).summary.turn_count == MAX_STORED_TURNS


def test_rejects_unknown_conversation_for_state_append_and_delete(
    tmp_path: Path,
) -> None:
    database_path = make_store(tmp_path)

    with pytest.raises(ConversationNotFoundError):
        load_conversation_state(database_path, "missing")
    with pytest.raises(ConversationNotFoundError):
        append_conversation_turn(
            database_path,
            "missing",
            user_message="question",
            assistant_message="answer",
        )
    with pytest.raises(ConversationNotFoundError):
        delete_conversation(database_path, "missing")
