"""Deterministic scorers for retrieval, Agent control, answers, and memory."""

from collections.abc import Iterable, Sequence
import math
import re
from typing import Any

from eval.schemas import (
    AgentExpectation,
    AnswerCase,
    MemoryCase,
    RetrievalCase,
)


_CITATION_PATTERN = re.compile(r"\[([0-9]{4}\.[0-9]{4,5})(?:v[0-9]+)?\]")


def reciprocal_rank(retrieved_ids: Sequence[str], target_ids: Iterable[str]) -> float:
    """Return the reciprocal rank of the first target, or zero when absent."""
    targets = set(target_ids)
    for rank, paper_id in enumerate(retrieved_ids, start=1):
        if paper_id in targets:
            return 1.0 / rank
    return 0.0


def _dcg(retrieved_ids: Sequence[str], grades: dict[str, int], k: int) -> float:
    return sum(
        (2 ** grades.get(paper_id, 0) - 1) / math.log2(rank + 1)
        for rank, paper_id in enumerate(retrieved_ids[:k], start=1)
    )


def ndcg(retrieved_ids: Sequence[str], grades: dict[str, int], k: int) -> float | None:
    """Calculate graded NDCG when the case contains relevance labels."""
    if not grades:
        return None
    ideal = sorted(grades.values(), reverse=True)
    ideal_dcg = sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal[:k], start=1)
    )
    return _dcg(retrieved_ids, grades, k) / ideal_dcg if ideal_dcg else 0.0


def score_retrieval_case(
    case: RetrievalCase,
    retrieved_ids: Sequence[str],
    *,
    top_k: int,
    top_similarity: float | None = None,
    top_rerank_score: float | None = None,
) -> dict[str, Any]:
    """Score one query without conflating answerable and unanswerable cases."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    ids = list(retrieved_ids)
    if not case.answerable:
        return {
            "answerable": False,
            "metrics": {
                "hit_at_k": None,
                "mrr": None,
                "ndcg_at_k": None,
                "top_similarity": top_similarity,
                "top_rerank_score": top_rerank_score,
            },
        }
    hit = any(paper_id in case.must_find_ids for paper_id in ids[:top_k])
    return {
        "answerable": True,
        "metrics": {
            "hit_at_k": float(hit),
            "mrr": reciprocal_rank(ids, case.must_find_ids),
            "ndcg_at_k": ndcg(ids, case.relevance_grades, top_k),
            "top_similarity": top_similarity,
            "top_rerank_score": top_rerank_score,
        },
    }


def aggregate_retrieval_scores(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate answerable ranking scores and separate refusal candidates."""
    answerable = [item for item in results if item.get("answerable")]
    unanswerable = [item for item in results if not item.get("answerable")]

    def mean_metric(name: str) -> float | None:
        values = [item["metrics"].get(name) for item in answerable]
        values = [float(value) for value in values if value is not None]
        return sum(values) / len(values) if values else None

    top_scores = [
        float(item["metrics"]["top_similarity"])
        for item in unanswerable
        if item["metrics"].get("top_similarity") is not None
    ]
    rerank_scores = [
        float(item["metrics"]["top_rerank_score"])
        for item in unanswerable
        if item["metrics"].get("top_rerank_score") is not None
    ]
    return {
        "answerable": {
            "question_count": len(answerable),
            "mean_hit_at_k": mean_metric("hit_at_k"),
            "mean_mrr": mean_metric("mrr"),
            "mean_ndcg_at_k": mean_metric("ndcg_at_k"),
        },
        "unanswerable": {
            "question_count": len(unanswerable),
            "mean_top_similarity": (
                sum(top_scores) / len(top_scores) if top_scores else None
            ),
            "max_top_similarity": max(top_scores) if top_scores else None,
            "mean_top_rerank_score": (
                sum(rerank_scores) / len(rerank_scores) if rerank_scores else None
            ),
            "max_top_rerank_score": max(rerank_scores) if rerank_scores else None,
        },
    }


def score_agent_turn(expected: AgentExpectation, result: Any) -> dict[str, Any]:
    """Compare a production RagResult's structured action with expectations."""
    decision = getattr(result, "conversation_decision", None)
    action = getattr(decision, "next_action", None)
    reusable_ids = tuple(getattr(decision, "reusable_arxiv_ids", ()) or ())
    retrieval_attempts = int(getattr(result, "retrieval_attempts", 0))
    failures: list[str] = []
    action_ok = action in expected.allowed_actions
    if not action_ok:
        failures.append(f"action {action!r} not in {list(expected.allowed_actions)!r}")
    retrieval_ok = expected.min_retrievals <= retrieval_attempts <= expected.max_retrievals
    if not retrieval_ok:
        failures.append(
            f"retrieval_attempts={retrieval_attempts} outside "
            f"[{expected.min_retrievals}, {expected.max_retrievals}]"
        )
    reuse_ok = True
    if expected.evidence_reuse == "required" and not reusable_ids:
        reuse_ok = False
        failures.append("expected reusable evidence")
    elif expected.evidence_reuse == "forbidden" and reusable_ids:
        reuse_ok = False
        failures.append(f"unexpected reusable evidence: {list(reusable_ids)}")
    answer_ok = bool(getattr(result, "answer", None))
    if not answer_ok:
        failures.append("answer was not generated")
    budget_ok = retrieval_attempts <= 2
    if not budget_ok:
        failures.append("retrieval budget exceeded two attempts")
    return {
        "passed": not failures,
        "checks": {
            "action": action_ok,
            "retrieval_bounds": retrieval_ok,
            "evidence_reuse": reuse_ok,
            "answer_generated": answer_ok,
            "retrieval_budget": budget_ok,
        },
        "actual": {
            "action": action,
            "retrieval_attempts": retrieval_attempts,
            "reusable_arxiv_ids": list(reusable_ids),
        },
        "failures": failures,
    }


def citation_ids(answer: str | None) -> tuple[str, ...]:
    """Extract canonical versionless arXiv IDs from bracket citations."""
    if not answer:
        return ()
    return tuple(dict.fromkeys(_CITATION_PATTERN.findall(answer)))


def score_answer_case(case: AnswerCase, result: Any) -> dict[str, Any]:
    """Score answer contracts while leaving semantic grading to humans."""
    answer = str(getattr(result, "answer", "") or "")
    cited = set(citation_ids(answer))
    evidence = {
        str(getattr(paper, "arxiv_id"))
        for paper in (getattr(result, "papers", ()) or ())
    }
    expected = set(case.expected_citations)
    lower_answer = answer.casefold()
    phrase_results = {
        phrase: phrase.casefold() in lower_answer for phrase in case.required_phrases
    }
    refusal_ok = (not case.must_refuse) or (not cited and not evidence)
    citation_ok = cited <= evidence
    expected_citation_ok = not expected or bool(cited & expected)
    passed = bool(answer.strip()) and refusal_ok and citation_ok and expected_citation_ok and all(
        phrase_results.values()
    )
    return {
        "passed": passed,
        "checks": {
            "answer_generated": bool(answer.strip()),
            "refusal": refusal_ok,
            "citations_in_evidence": citation_ok,
            "expected_citation": expected_citation_ok,
            "required_phrases": all(phrase_results.values()),
        },
        "actual": {
            "citation_ids": sorted(cited),
            "evidence_ids": sorted(evidence),
            "required_phrase_results": phrase_results,
        },
    }


def score_memory_case(
    case: MemoryCase,
    *,
    summary: str | None,
    compaction_count: int,
) -> dict[str, Any]:
    """Check that a summary retained required user constraints."""
    normalized = (summary or "").casefold()
    phrase_results = {
        phrase: phrase.casefold() in normalized
        for phrase in case.required_memory_phrases
    }
    compaction_ok = compaction_count >= case.min_compactions
    return {
        "passed": compaction_ok and all(phrase_results.values()),
        "checks": {
            "minimum_compactions": compaction_ok,
            "required_memory": all(phrase_results.values()),
        },
        "actual": {
            "compaction_count": compaction_count,
            "required_memory_results": phrase_results,
        },
    }
