import json
from types import SimpleNamespace

import pytest

from config.model_settings import ModelSettings
from rag.conversation import (
    ConversationDecisionError,
    ConversationTurn,
    bounded_history,
    build_conversation_decision_messages,
    decide_conversation_action,
    parse_conversation_decision,
)
from rag.search import SearchResult


SETTINGS = ModelSettings("key", "https://api.deepseek.com", "deepseek-chat")


def make_paper(arxiv_id: str = "2607.00001") -> SearchResult:
    return SearchResult(
        arxiv_id=arxiv_id,
        title="Failure localization in multi-agent systems",
        document="The method identifies responsible agents and failed steps.",
        entry_url=f"https://arxiv.org/abs/{arxiv_id}",
        primary_category="cs.AI",
        similarity=0.8,
    )


def test_bounded_history_keeps_six_complete_turns() -> None:
    turns = [ConversationTurn(f"q{i}", f"a{i}") for i in range(8)]

    bounded = bounded_history(turns)

    assert [turn.user_message for turn in bounded] == [
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q7",
    ]


def test_decision_prompt_contains_history_question_and_active_evidence() -> None:
    messages = build_conversation_decision_messages(
        "它能定位到具体步骤吗？",
        [ConversationTurn("如何定位负责的 Agent？", "使用故障定位方法。")],
        [make_paper()],
    )
    payload = json.loads(messages[1]["content"])

    assert payload["current_question"] == "它能定位到具体步骤吗？"
    assert payload["conversation_history"][0]["user"] == "如何定位负责的 Agent？"
    assert payload["active_evidence"][0]["arxiv_id"] == "2607.00001"
    assert "不允许直接拒答" in messages[0]["content"]


@pytest.mark.parametrize(
    ("payload", "expected_action"),
    [
        (
            {
                "coverage": "sufficient",
                "next_action": "answer_from_existing",
                "reason": "已有论文直接覆盖步骤定位",
                "standalone_question": "该方法能否定位具体失败步骤？",
                "reusable_arxiv_ids": ["2607.00001"],
                "missing_aspects": [],
                "retrieval_query": None,
            },
            "answer_from_existing",
        ),
        (
            {
                "coverage": "partial",
                "next_action": "retrieve_missing",
                "reason": "缺少代码 Agent 实验",
                "standalone_question": "该方法在代码 Agent 中表现如何？",
                "reusable_arxiv_ids": ["2607.00001"],
                "missing_aspects": ["coding agent evaluation"],
                "retrieval_query": "failure localization coding agent evaluation",
            },
            "retrieve_missing",
        ),
        (
            {
                "coverage": "unrelated",
                "next_action": "fresh_retrieval",
                "reason": "当前问题是新话题",
                "standalone_question": "Cross-encoder 在 RAG 中有什么作用？",
                "reusable_arxiv_ids": [],
                "missing_aspects": ["cross-encoder reranking"],
                "retrieval_query": "cross-encoder reranking in RAG",
            },
            "fresh_retrieval",
        ),
    ],
)
def test_parses_supported_conversation_actions(
    payload: dict[str, object],
    expected_action: str,
) -> None:
    decision = parse_conversation_decision(json.dumps(payload))

    assert decision.next_action == expected_action


def test_rejects_direct_refusal_or_inconsistent_action() -> None:
    with pytest.raises(ConversationDecisionError):
        parse_conversation_decision(
            json.dumps(
                {
                    "coverage": "partial",
                    "next_action": "fresh_retrieval",
                    "reason": "invalid mapping",
                    "standalone_question": "question",
                    "reusable_arxiv_ids": [],
                    "missing_aspects": [],
                    "retrieval_query": "query",
                }
            )
        )


class UnknownEvidenceCompletions:
    def create(self, **_: object) -> object:
        content = json.dumps(
            {
                "coverage": "sufficient",
                "next_action": "answer_from_existing",
                "reason": "claims unknown evidence",
                "standalone_question": "resolved question",
                "reusable_arxiv_ids": ["unknown"],
                "missing_aspects": [],
                "retrieval_query": None,
            }
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def test_decision_cannot_invent_reusable_evidence_ids() -> None:
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=UnknownEvidenceCompletions())
    )

    with pytest.raises(ConversationDecisionError, match="unknown evidence"):
        decide_conversation_action(
            "follow-up",
            [ConversationTurn("q", "a")],
            [make_paper()],
            settings=SETTINGS,
            client=client,
        )
