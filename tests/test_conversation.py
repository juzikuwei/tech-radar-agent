import json
from types import SimpleNamespace

import pytest

from config.model_settings import ModelSettings
from rag.conversation import (
    ConversationDecisionError,
    ConversationTurn,
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


def test_decision_prompt_contains_history_question_and_active_evidence() -> None:
    messages = build_conversation_decision_messages(
        "它能定位到具体步骤吗？",
        [
            ConversationTurn(
                "如何定位负责的 Agent？",
                "使用故障定位方法。",
                ("2607.00001",),
            )
        ],
        [make_paper()],
    )
    payload = json.loads(messages[1]["content"])

    assert payload["current_question"] == "它能定位到具体步骤吗？"
    assert payload["conversation_history"][0]["user"] == "如何定位负责的 Agent？"
    assert payload["conversation_history"][0]["evidence_ids"] == ["2607.00001"]
    assert payload["active_evidence"][0]["arxiv_id"] == "2607.00001"
    assert "respond" in messages[0]["content"]


def test_decision_prompt_contains_summary_and_all_uncompacted_turns() -> None:
    summary = json.dumps(
        {
            "user_goals": ["实现上下文压缩"],
            "confirmed_requirements": [],
            "decisions": [],
            "important_context": [],
            "open_questions": [],
        },
        ensure_ascii=False,
    )
    messages = build_conversation_decision_messages(
        "继续",
        [ConversationTurn(f"q{i}", f"a{i}") for i in range(8)],
        [],
        context_summary=summary,
    )
    payload = json.loads(messages[1]["content"])

    assert payload["conversation_summary"]["user_goals"] == ["实现上下文压缩"]
    assert [turn["user"] for turn in payload["conversation_history"]] == [
        f"q{i}" for i in range(8)
    ]


def test_decision_prompt_does_not_truncate_uncompacted_user_wording() -> None:
    original = "用户原话" * 400
    messages = build_conversation_decision_messages(
        "继续",
        [ConversationTurn(original, "assistant")],
        [],
    )
    payload = json.loads(messages[1]["content"])

    assert payload["conversation_history"][0]["user"] == original


@pytest.mark.parametrize(
    ("payload", "expected_action"),
    [
        (
            {
                "coverage": "not_applicable",
                "next_action": "respond",
                "reason": "用户只是在表达感谢",
                "standalone_question": "用户感谢上一轮回答",
                "reusable_arxiv_ids": [],
                "missing_aspects": [],
                "retrieval_query": None,
            },
            "respond",
        ),
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


class RepairingConversationCompletions:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **_: object) -> object:
        self.calls += 1
        if self.calls == 1:
            payload = {
                "coverage": "none",
                "next_action": "fresh_retrieval",
                "reason": "invalid missing query",
                "standalone_question": "用户在表达不满",
                "reusable_arxiv_ids": [],
                "missing_aspects": [],
                "retrieval_query": None,
            }
        else:
            payload = {
                "coverage": "not_applicable",
                "next_action": "respond",
                "reason": "用户在反馈上一轮回答",
                "standalone_question": "回应用户对上一轮回答的不满",
                "reusable_arxiv_ids": [],
                "missing_aspects": [],
                "retrieval_query": None,
            }
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(payload))
                )
            ]
        )


def test_conversation_decision_repairs_invalid_action_once() -> None:
    completions = RepairingConversationCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    decision = decide_conversation_action(
        "感觉你在乱说",
        [ConversationTurn("question", "answer")],
        [make_paper()],
        settings=SETTINGS,
        client=client,
    )

    assert completions.calls == 2
    assert decision.next_action == "respond"
