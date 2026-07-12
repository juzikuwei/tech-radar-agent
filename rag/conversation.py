"""Conversation state and evidence-aware action decisions for follow-up turns."""

from collections.abc import Sequence
from dataclasses import dataclass
import json
from typing import Any, Literal

from config.model_settings import ModelSettings
from rag.llm_client import StatusCallback, generate_text
from rag.search import SearchResult


MAX_CONVERSATION_TURNS = 6
MAX_ACTIVE_EVIDENCE = 5
HISTORY_USER_CHAR_LIMIT = 1_000
HISTORY_ASSISTANT_CHAR_LIMIT = 2_000
EVIDENCE_ABSTRACT_CHAR_LIMIT = 800

Coverage = Literal["sufficient", "partial", "unrelated", "none"]
NextAction = Literal[
    "answer_from_existing",
    "retrieve_missing",
    "fresh_retrieval",
]

SYSTEM_PROMPT = """你是对话式 arXiv RAG 的证据控制器。

你要根据当前追问、最近对话和上一轮活动论文证据，选择下一步动作。不要回答用户问题。
对话历史只用于理解指代和意图；只有 active_evidence 中的论文标题与摘要可以作为已有事实证据。

动作规则：
1. answer_from_existing：已有论文足以直接回答当前问题，不需要检索。
2. retrieve_missing：已有论文只覆盖一部分，保留有用论文并只检索缺失信息。
3. fresh_retrieval：当前问题是新话题，或没有可复用证据，从新的独立查询开始检索。
4. 不允许直接拒答；证据不足时必须先选择一种检索动作。
5. standalone_question 必须是脱离历史也能理解的完整中文问题。
6. retrieval_query 必须是简短、独立、适合检索英文 arXiv 摘要的英文查询。
7. reusable_arxiv_ids 只能来自 active_evidence。

只输出以下 JSON 对象：
{
  "coverage": "sufficient | partial | unrelated | none",
  "next_action": "answer_from_existing | retrieve_missing | fresh_retrieval",
  "reason": "简短中文理由",
  "standalone_question": "完整中文问题",
  "reusable_arxiv_ids": ["arXiv ID"],
  "missing_aspects": ["缺失方面"],
  "retrieval_query": "英文查询" 或 null
}
"""


class ConversationDecisionError(ValueError):
    """The model returned an unusable conversation action decision."""


@dataclass(frozen=True)
class ConversationTurn:
    """One completed user and assistant exchange."""

    user_message: str
    assistant_message: str


@dataclass(frozen=True)
class ConversationDecision:
    """Structured action chosen from the current evidence state."""

    coverage: Coverage
    next_action: NextAction
    reason: str
    standalone_question: str
    reusable_arxiv_ids: tuple[str, ...]
    missing_aspects: tuple[str, ...]
    retrieval_query: str | None


def bounded_history(
    turns: Sequence[ConversationTurn],
) -> tuple[ConversationTurn, ...]:
    """Return at most the six most recent complete turns."""
    return tuple(turns[-MAX_CONVERSATION_TURNS:])


def build_conversation_decision_messages(
    question: str,
    history: Sequence[ConversationTurn],
    active_evidence: Sequence[SearchResult],
) -> list[dict[str, str]]:
    """Build a bounded prompt for choosing reuse or retrieval."""
    clean_question = question.strip()
    if not clean_question:
        raise ValueError("question must not be empty")

    history_payload = [
        {
            "user": turn.user_message[:HISTORY_USER_CHAR_LIMIT],
            "assistant": turn.assistant_message[:HISTORY_ASSISTANT_CHAR_LIMIT],
        }
        for turn in bounded_history(history)
    ]
    evidence_payload = [
        {
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "abstract": paper.document[:EVIDENCE_ABSTRACT_CHAR_LIMIT],
        }
        for paper in active_evidence[:MAX_ACTIVE_EVIDENCE]
    ]
    payload = {
        "conversation_history": history_payload,
        "current_question": clean_question,
        "active_evidence": evidence_payload,
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, indent=2),
        },
    ]


def parse_conversation_decision(content: str) -> ConversationDecision:
    """Validate a model decision before it controls evidence reuse."""
    try:
        value: Any = json.loads(content)
    except json.JSONDecodeError as error:
        raise ConversationDecisionError(
            "conversation decision is not valid JSON"
        ) from error
    if not isinstance(value, dict):
        raise ConversationDecisionError(
            "conversation decision must be a JSON object"
        )

    coverage = value.get("coverage")
    next_action = value.get("next_action")
    reason = value.get("reason")
    standalone_question = value.get("standalone_question")
    reusable_ids = value.get("reusable_arxiv_ids")
    missing_aspects = value.get("missing_aspects")
    retrieval_query = value.get("retrieval_query")

    allowed_coverages = {"sufficient", "partial", "unrelated", "none"}
    allowed_actions = {
        "answer_from_existing",
        "retrieve_missing",
        "fresh_retrieval",
    }
    if coverage not in allowed_coverages:
        raise ConversationDecisionError("invalid coverage")
    if next_action not in allowed_actions:
        raise ConversationDecisionError("invalid next_action")
    if not isinstance(reason, str) or not reason.strip():
        raise ConversationDecisionError("reason must be a non-empty string")
    if not isinstance(standalone_question, str) or not standalone_question.strip():
        raise ConversationDecisionError(
            "standalone_question must be a non-empty string"
        )
    if not isinstance(reusable_ids, list) or not all(
        isinstance(item, str) and item.strip() for item in reusable_ids
    ):
        raise ConversationDecisionError("reusable_arxiv_ids must be a string list")
    if len(set(reusable_ids)) != len(reusable_ids):
        raise ConversationDecisionError("reusable_arxiv_ids must not contain duplicates")
    if not isinstance(missing_aspects, list) or not all(
        isinstance(item, str) and item.strip() for item in missing_aspects
    ):
        raise ConversationDecisionError("missing_aspects must be a string list")

    expected_coverage = {
        "answer_from_existing": "sufficient",
        "retrieve_missing": "partial",
    }
    required_coverage = expected_coverage.get(next_action)
    if required_coverage is not None and coverage != required_coverage:
        raise ConversationDecisionError(
            f"{next_action} requires {required_coverage} coverage"
        )
    if next_action == "fresh_retrieval" and coverage not in {"unrelated", "none"}:
        raise ConversationDecisionError(
            "fresh_retrieval requires unrelated or none coverage"
        )

    if next_action == "answer_from_existing":
        if not reusable_ids:
            raise ConversationDecisionError(
                "answer_from_existing needs reusable evidence"
            )
        if retrieval_query is not None:
            raise ConversationDecisionError(
                "answer_from_existing must not include a retrieval query"
            )
        clean_retrieval_query = None
    else:
        if not isinstance(retrieval_query, str) or not retrieval_query.strip():
            raise ConversationDecisionError(
                "retrieval actions need a retrieval query"
            )
        clean_retrieval_query = retrieval_query.strip()
        if next_action == "retrieve_missing" and not reusable_ids:
            raise ConversationDecisionError(
                "retrieve_missing needs reusable evidence"
            )
        if next_action == "fresh_retrieval" and reusable_ids:
            raise ConversationDecisionError(
                "fresh_retrieval must not reuse previous evidence"
            )

    return ConversationDecision(
        coverage=coverage,
        next_action=next_action,
        reason=reason.strip(),
        standalone_question=standalone_question.strip(),
        reusable_arxiv_ids=tuple(item.strip() for item in reusable_ids),
        missing_aspects=tuple(item.strip() for item in missing_aspects),
        retrieval_query=clean_retrieval_query,
    )


def decide_conversation_action(
    question: str,
    history: Sequence[ConversationTurn],
    active_evidence: Sequence[SearchResult],
    *,
    settings: ModelSettings,
    client: Any | None = None,
    on_retry: StatusCallback | None = None,
) -> ConversationDecision:
    """Ask DeepSeek whether to reuse, supplement, or replace evidence."""
    messages = build_conversation_decision_messages(
        question,
        history,
        active_evidence,
    )
    content = generate_text(
        messages,
        settings=settings,
        client=client,
        on_retry=on_retry,
        response_format={"type": "json_object"},
        max_tokens=500,
        temperature=0.0,
    )
    decision = parse_conversation_decision(content)
    available_ids = {paper.arxiv_id for paper in active_evidence}
    unknown_ids = set(decision.reusable_arxiv_ids) - available_ids
    if unknown_ids:
        raise ConversationDecisionError(
            f"conversation decision references unknown evidence: {sorted(unknown_ids)}"
        )
    return decision
