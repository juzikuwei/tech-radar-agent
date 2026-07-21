"""Judge retrieved evidence and propose one bounded query rewrite."""

from collections.abc import Sequence
from dataclasses import dataclass
import json
from typing import Any

from config.model_settings import ModelSettings
from rag.llm_client import StatusCallback, UsageCallback, generate_text
from rag.search import SearchResult


DEFAULT_JUDGE_PAPER_COUNT = 3
DEFAULT_ABSTRACT_CHAR_LIMIT = 800

SYSTEM_PROMPT = """你是本地 arXiv RAG 系统的检索质量判断器。

你只判断给定论文标题和摘要是否足以直接支持回答用户问题，不要回答问题本身。
必须输出一个 JSON 对象，并遵守以下规则：
1. sufficient 只有在证据直接覆盖问题核心对象和关系时才为 true。
2. 仅仅出现相同领域词、Agent、RAG 等通用词不代表证据充分。
3. sufficient 为 false 时，指出缺失方面，并生成一个独立、简短的英文检索查询。
4. 改写查询必须保持原问题含义和专有名词，不得加入无关主题。
5. sufficient 为 true 时，rewritten_query 必须为 null。

JSON 字段固定为：
{
  "sufficient": true 或 false,
  "reason": "简短中文理由",
  "missing_aspects": ["缺失方面"],
  "rewritten_query": "英文查询" 或 null
}
"""


class RetrievalDecisionError(ValueError):
    """The model returned an unusable retrieval decision."""


@dataclass(frozen=True)
class RetrievalDecision:
    """Structured decision controlling an optional second retrieval."""

    sufficient: bool
    reason: str
    missing_aspects: tuple[str, ...]
    rewritten_query: str | None


def build_judge_messages(
    question: str,
    papers: Sequence[SearchResult],
    *,
    paper_count: int = DEFAULT_JUDGE_PAPER_COUNT,
    abstract_char_limit: int = DEFAULT_ABSTRACT_CHAR_LIMIT,
) -> list[dict[str, str]]:
    """Build a bounded evidence payload for retrieval-quality judgment."""
    clean_question = question.strip()
    if not clean_question:
        raise ValueError("question must not be empty")
    if paper_count <= 0:
        raise ValueError("paper_count must be greater than zero")
    if abstract_char_limit <= 0:
        raise ValueError("abstract_char_limit must be greater than zero")

    evidence = [
        {
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "abstract": paper.document[:abstract_char_limit],
        }
        for paper in papers[:paper_count]
    ]
    user_content = json.dumps(
        {"question": clean_question, "retrieved_papers": evidence},
        ensure_ascii=False,
        indent=2,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def parse_retrieval_decision(content: str) -> RetrievalDecision:
    """Validate the model's JSON decision before it controls retrieval."""
    try:
        value: Any = json.loads(content)
    except json.JSONDecodeError as error:
        raise RetrievalDecisionError("retrieval decision is not valid JSON") from error
    if not isinstance(value, dict):
        raise RetrievalDecisionError("retrieval decision must be a JSON object")

    sufficient = value.get("sufficient")
    reason = value.get("reason")
    missing_aspects = value.get("missing_aspects")
    rewritten_query = value.get("rewritten_query")

    if not isinstance(sufficient, bool):
        raise RetrievalDecisionError("sufficient must be a boolean")
    if not isinstance(reason, str) or not reason.strip():
        raise RetrievalDecisionError("reason must be a non-empty string")
    if not isinstance(missing_aspects, list) or not all(
        isinstance(item, str) and item.strip() for item in missing_aspects
    ):
        raise RetrievalDecisionError("missing_aspects must be a string list")

    if sufficient:
        if rewritten_query is not None:
            raise RetrievalDecisionError(
                "sufficient decisions must not contain a rewritten query"
            )
        clean_rewritten_query = None
    else:
        if not isinstance(rewritten_query, str) or not rewritten_query.strip():
            raise RetrievalDecisionError(
                "insufficient decisions need a rewritten query"
            )
        clean_rewritten_query = rewritten_query.strip()

    return RetrievalDecision(
        sufficient=sufficient,
        reason=reason.strip(),
        missing_aspects=tuple(item.strip() for item in missing_aspects),
        rewritten_query=clean_rewritten_query,
    )


def judge_retrieval(
    question: str,
    papers: Sequence[SearchResult],
    *,
    settings: ModelSettings,
    client: Any | None = None,
    on_retry: StatusCallback | None = None,
    on_usage: UsageCallback | None = None,
) -> RetrievalDecision:
    """Ask the configured DeepSeek-compatible model for one retrieval decision."""
    messages = build_judge_messages(question, papers)
    content = generate_text(
        messages,
        settings=settings,
        client=client,
        on_retry=on_retry,
        response_format={"type": "json_object"},
        max_tokens=300,
        temperature=0.0,
        on_usage=on_usage,
    )
    decision = parse_retrieval_decision(content)
    if (
        decision.rewritten_query is not None
        and decision.rewritten_query.casefold() == question.strip().casefold()
    ):
        raise RetrievalDecisionError("rewritten query must differ from the question")
    return decision
