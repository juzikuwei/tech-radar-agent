"""Dependency-injected runners for repeatable local evaluation suites."""

from collections.abc import Callable, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from eval.schemas import (
    AgentCase,
    AgentTurn,
    AnswerCase,
    MemoryCase,
    RetrievalCase,
)
from eval.scoring import (
    aggregate_retrieval_scores,
    score_agent_turn,
    score_answer_case,
    score_memory_case,
    score_retrieval_case,
)
from rag.conversation import ConversationTurn


class SearchFn(Protocol):
    """Search boundary used by retrieval evaluations."""

    def __call__(self, query: str, top_k: int) -> Sequence[Any]: ...


class TurnFn(Protocol):
    """Production conversation boundary used by Agent evaluations."""

    def __call__(
        self,
        question: str,
        history: tuple[ConversationTurn, ...],
        active_evidence: tuple[Any, ...],
    ) -> Any: ...


class MemoryFn(Protocol):
    """Boundary that executes a memory case and returns summary metadata."""

    def __call__(self, case: MemoryCase) -> tuple[str | None, int]: ...


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def run_retrieval_cases(
    cases: Sequence[RetrievalCase],
    *,
    search: SearchFn,
    top_k: int,
) -> dict[str, Any]:
    """Run retrieval cases and return per-query plus aggregate scores."""
    results: list[dict[str, Any]] = []
    for case in cases:
        started = perf_counter()
        found = list(search(case.query, top_k))
        duration_ms = (perf_counter() - started) * 1_000
        ids = [str(getattr(item, "arxiv_id")) for item in found]
        top_similarity = (
            getattr(found[0], "similarity", None) if found else None
        )
        top_rerank_score = (
            getattr(found[0], "rerank_score", None) if found else None
        )
        score = score_retrieval_case(
            case,
            ids,
            top_k=top_k,
            top_similarity=(
                None if top_similarity is None else float(top_similarity)
            ),
            top_rerank_score=(
                None if top_rerank_score is None else float(top_rerank_score)
            ),
        )
        results.append(
            {
                "id": case.id,
                "query": case.query,
                "tags": list(case.tags),
                "retrieved_ids": ids,
                "duration_ms": duration_ms,
                **score,
            }
        )
    return {
        "suite": "retrieval",
        "top_k": top_k,
        "summary": aggregate_retrieval_scores(results),
        "cases": results,
    }


def _history_after_result(
    history: tuple[ConversationTurn, ...],
    result: Any,
) -> tuple[ConversationTurn, ...]:
    answer = getattr(result, "answer", None)
    if not answer:
        return history
    evidence_ids = tuple(
        str(getattr(paper, "arxiv_id"))
        for paper in (getattr(result, "papers", ()) or ())
    )
    return (*history, ConversationTurn(str(getattr(result, "question", "")), answer, evidence_ids))


def run_agent_cases(
    cases: Sequence[AgentCase],
    *,
    run_turn: TurnFn,
) -> dict[str, Any]:
    """Run multi-turn Agent scenarios against the production turn function."""
    reports: list[dict[str, Any]] = []
    for case in cases:
        history: tuple[ConversationTurn, ...] = ()
        active_evidence: tuple[Any, ...] = ()
        turn_reports: list[dict[str, Any]] = []
        for index, turn in enumerate(case.turns, start=1):
            started = perf_counter()
            result = run_turn(turn.question, history, active_evidence)
            duration_ms = (perf_counter() - started) * 1_000
            evaluation = (
                {
                    "passed": bool(getattr(result, "answer", None)),
                    "checks": {
                        "answer_generated": bool(getattr(result, "answer", None))
                    },
                    "failures": (
                        []
                        if getattr(result, "answer", None)
                        else ["answer was not generated"]
                    ),
                }
                if turn.expected is None
                else score_agent_turn(turn.expected, result)
            )
            turn_reports.append(
                {
                    "turn": index,
                    "question": turn.question,
                    "duration_ms": duration_ms,
                    "evaluation": evaluation,
                    "trace_stages": [
                        str(getattr(event, "stage"))
                        for event in (getattr(result, "trace", ()) or ())
                    ],
                }
            )
            history = _history_after_result(history, result)
            active_evidence = tuple(getattr(result, "papers", ()) or ())
        passed = all(
            report["evaluation"]["passed"]
            for report in turn_reports
        )
        reports.append({"id": case.id, "tags": list(case.tags), "passed": passed, "turns": turn_reports})
    return {
        "suite": "agent",
        "summary": {
            "case_count": len(reports),
            "passed": sum(bool(item["passed"]) for item in reports),
            "pass_rate": (
                sum(bool(item["passed"]) for item in reports) / len(reports)
                if reports
                else None
            ),
        },
        "cases": reports,
    }


def run_answer_cases(
    cases: Sequence[AnswerCase],
    *,
    run_turn: TurnFn,
) -> dict[str, Any]:
    """Run answer/refusal cases and apply deterministic contract checks."""
    reports: list[dict[str, Any]] = []
    for case in cases:
        started = perf_counter()
        result = run_turn(case.question, (), ())
        duration_ms = (perf_counter() - started) * 1_000
        reports.append(
            {
                "id": case.id,
                "question": case.question,
                "duration_ms": duration_ms,
                "evaluation": score_answer_case(case, result),
            }
        )
    passed = sum(bool(item["evaluation"]["passed"]) for item in reports)
    return {
        "suite": "answer",
        "summary": {
            "case_count": len(reports),
            "passed": passed,
            "pass_rate": passed / len(reports) if reports else None,
        },
        "cases": reports,
    }


def run_memory_cases(
    cases: Sequence[MemoryCase],
    *,
    run_memory: MemoryFn,
) -> dict[str, Any]:
    """Run memory cases through an injected compaction-aware adapter."""
    reports: list[dict[str, Any]] = []
    for case in cases:
        started = perf_counter()
        summary, compaction_count = run_memory(case)
        duration_ms = (perf_counter() - started) * 1_000
        reports.append(
            {
                "id": case.id,
                "turn_count": len(case.turns),
                "duration_ms": duration_ms,
                "evaluation": score_memory_case(
                    case,
                    summary=summary,
                    compaction_count=compaction_count,
                ),
            }
        )
    passed = sum(bool(item["evaluation"]["passed"]) for item in reports)
    return {
        "suite": "memory",
        "summary": {
            "case_count": len(reports),
            "passed": passed,
            "pass_rate": passed / len(reports) if reports else None,
        },
        "cases": reports,
    }


def write_report(report: dict[str, Any], path: Path) -> None:
    """Write a JSON report while keeping generated data outside version control."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
