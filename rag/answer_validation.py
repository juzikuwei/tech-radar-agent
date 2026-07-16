"""Validate that research answers cite only evidence from the current run."""

from collections.abc import Sequence
from dataclasses import dataclass
import re
from typing import Literal

from rag.search import SearchResult


SAFE_INSUFFICIENT_EVIDENCE_RESPONSE = (
    "当前本地论文库没有检索到足以支持可靠回答的 arXiv 摘要证据。"
    "请补充更具体的技术名称、研究对象、时间范围或比较维度后重试。"
)
SAFE_UNVERIFIED_ANSWER_RESPONSE = (
    "本次生成结果没有通过引用校验，因此我不能可靠地展示其中的研究结论。"
    "请重试，或把问题缩小到更具体的技术点。"
)

ValidationReason = Literal[
    "valid",
    "empty_answer",
    "missing_citation",
    "unknown_citation",
]

_BRACKET_GROUP_PATTERN = re.compile(r"\[([^\]]+)\]")
_CITATION_SEPARATOR_PATTERN = re.compile(r"[,，;；\s]+")
_ARXIV_ID_PATTERN = re.compile(
    r"(?<![\d.])"
    r"(?:\d{2}(?:0[1-9]|1[0-2])\.\d{4,5}"
    r"|[A-Za-z][A-Za-z0-9.-]*/\d{7})"
    r"(?:v\d+)?"
    r"(?![\d.])",
    flags=re.IGNORECASE,
)
_DIRECT_CONVERSATION_PATTERNS = (
    re.compile(
        r"^(?:你好|您好|嗨|hello|hi|谢谢|感谢|多谢|辛苦了|做得好|不错|"
        r"好的|好|明白了|知道了|收到|再见|你是谁|你能做什么|"
        r"介绍一下你自己)[！!。.，,？?\s]*$",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"^(?:感觉|我觉得)?(?:你|这个回答).*(?:不对|有问题|乱说|不错|很好|"
        r"太长|太短)[！!。.？?\s]*$"
    ),
)


@dataclass(frozen=True)
class AnswerValidation:
    """Deterministic citation-validation result for one research answer."""

    is_valid: bool
    cited_ids: tuple[str, ...]
    unknown_citation_ids: tuple[str, ...]
    reason: ValidationReason


def validate_research_answer(
    answer: str,
    evidence: Sequence[SearchResult],
) -> AnswerValidation:
    """Require at least one valid citation and reject every unknown arXiv ID."""
    if not answer.strip():
        return AnswerValidation(False, (), (), "empty_answer")

    evidence_by_id = {paper.arxiv_id: paper for paper in evidence}
    cited_ids: list[str] = []

    for group in _BRACKET_GROUP_PATTERN.findall(answer):
        for token in _CITATION_SEPARATOR_PATTERN.split(group.strip()):
            clean_token = token.strip()
            if not clean_token:
                continue
            if clean_token in evidence_by_id:
                cited_ids.append(clean_token)

    unknown_ids = [
        match.group(0)
        for match in _ARXIV_ID_PATTERN.finditer(answer)
        if match.group(0) not in evidence_by_id
    ]

    unique_cited_ids = tuple(dict.fromkeys(cited_ids))
    unique_unknown_ids = tuple(dict.fromkeys(unknown_ids))
    if unique_unknown_ids:
        return AnswerValidation(
            False,
            unique_cited_ids,
            unique_unknown_ids,
            "unknown_citation",
        )
    if not unique_cited_ids:
        return AnswerValidation(False, (), (), "missing_citation")
    return AnswerValidation(True, unique_cited_ids, (), "valid")


def select_cited_papers(
    validation: AnswerValidation,
    evidence: Sequence[SearchResult],
) -> tuple[SearchResult, ...]:
    """Return validated evidence in first-citation order."""
    if not validation.is_valid:
        return ()
    evidence_by_id = {paper.arxiv_id: paper for paper in evidence}
    return tuple(
        evidence_by_id[paper_id]
        for paper_id in validation.cited_ids
        if paper_id in evidence_by_id
    )


def is_direct_conversation_message(message: str) -> bool:
    """Allow conservative greetings, feedback, and assistant-meta questions."""
    clean_message = " ".join(message.split())
    return any(pattern.fullmatch(clean_message) for pattern in _DIRECT_CONVERSATION_PATTERNS)
