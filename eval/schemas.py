"""Versioned case schemas used by the local evaluation runner."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal


class EvalDataError(ValueError):
    """Raised when a tracked evaluation dataset is malformed."""


@dataclass(frozen=True)
class RetrievalCase:
    """One query with anchor papers and optional graded relevance labels."""

    id: str
    query: str
    answerable: bool
    must_find_ids: tuple[str, ...]
    relevance_grades: dict[str, int]
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentExpectation:
    """Deterministic expectations for one conversational Agent turn."""

    allowed_actions: tuple[str, ...]
    min_retrievals: int
    max_retrievals: int
    evidence_reuse: Literal["required", "forbidden", "optional"]


@dataclass(frozen=True)
class AgentTurn:
    """One user message and its optional control-flow expectation."""

    question: str
    expected: AgentExpectation | None = None


@dataclass(frozen=True)
class AgentCase:
    """A multi-turn scenario evaluated against the production trace."""

    id: str
    turns: tuple[AgentTurn, ...]
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnswerCase:
    """A final-answer case with lightweight, reviewable checks."""

    id: str
    question: str
    answerable: bool
    expected_citations: tuple[str, ...] = ()
    required_phrases: tuple[str, ...] = ()
    must_refuse: bool = False
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class MemoryCase:
    """A case for checking memory retained after one or more compactions."""

    id: str
    turns: tuple[str, ...]
    required_memory_phrases: tuple[str, ...]
    min_compactions: int = 1
    tags: tuple[str, ...] = ()


def _read_dataset(path: Path, key: str) -> tuple[int, list[dict[str, Any]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvalDataError(f"cannot read {path}: {error}") from error
    if not isinstance(payload, dict):
        raise EvalDataError(f"{path} must contain a JSON object")
    version = payload.get("dataset_version")
    cases = payload.get(key)
    if not isinstance(version, int) or version <= 0:
        raise EvalDataError(f"{path} needs a positive integer dataset_version")
    if not isinstance(cases, list) or not cases:
        raise EvalDataError(f"{path} needs a non-empty {key} list")
    return version, cases


def _clean_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvalDataError(f"{field} must be a non-empty string")
    return value.strip()


def _string_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise EvalDataError(f"{field} must be a list of non-empty strings")
    return tuple(dict.fromkeys(item.strip() for item in value))


def _tags(value: object, *, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    return _string_tuple(value, field=field)


def load_retrieval_cases(path: Path) -> tuple[int, tuple[RetrievalCase, ...]]:
    """Load and validate retrieval cases."""
    version, raw_cases = _read_dataset(path, "cases")
    parsed: list[RetrievalCase] = []
    seen: set[str] = set()
    for raw in raw_cases:
        if not isinstance(raw, dict):
            raise EvalDataError("retrieval cases must be JSON objects")
        case_id = _clean_string(raw.get("id"), field="retrieval case id")
        if case_id in seen:
            raise EvalDataError(f"duplicate retrieval case id: {case_id}")
        seen.add(case_id)
        query = _clean_string(raw.get("query"), field=f"{case_id}.query")
        answerable = raw.get("answerable")
        if not isinstance(answerable, bool):
            raise EvalDataError(f"{case_id}.answerable must be boolean")
        must_find_ids = _string_tuple(
            raw.get("must_find_ids", []), field=f"{case_id}.must_find_ids"
        )
        if answerable and not must_find_ids:
            raise EvalDataError(
                f"answerable retrieval case {case_id} needs must_find_ids"
            )
        raw_grades = raw.get("relevance_grades", {})
        if not isinstance(raw_grades, dict):
            raise EvalDataError(f"{case_id}.relevance_grades must be an object")
        grades: dict[str, int] = {}
        for paper_id, grade in raw_grades.items():
            if (
                not isinstance(paper_id, str)
                or not paper_id.strip()
                or not isinstance(grade, int)
                or isinstance(grade, bool)
                or grade <= 0
            ):
                raise EvalDataError(f"{case_id}.relevance_grades is invalid")
            grades[paper_id.strip()] = grade
        parsed.append(
            RetrievalCase(
                id=case_id,
                query=query,
                answerable=answerable,
                must_find_ids=must_find_ids,
                relevance_grades=grades,
                tags=_tags(raw.get("tags"), field=f"{case_id}.tags"),
            )
        )
    return version, tuple(parsed)


def _parse_expectation(raw: object, *, field: str) -> AgentExpectation:
    if not isinstance(raw, dict):
        raise EvalDataError(f"{field} must be an object")
    actions = _string_tuple(raw.get("allowed_actions"), field=f"{field}.allowed_actions")
    allowed = {"respond", "answer_from_existing", "retrieve_missing", "fresh_retrieval"}
    if not set(actions) <= allowed:
        raise EvalDataError(f"{field}.allowed_actions contains an unknown action")
    minimum = raw.get("min_retrievals")
    maximum = raw.get("max_retrievals")
    if (
        not isinstance(minimum, int)
        or isinstance(minimum, bool)
        or not isinstance(maximum, int)
        or isinstance(maximum, bool)
        or minimum < 0
        or maximum < minimum
        or maximum > 2
    ):
        raise EvalDataError(f"{field} retrieval bounds are invalid")
    reuse = raw.get("evidence_reuse")
    if reuse not in {"required", "forbidden", "optional"}:
        raise EvalDataError(f"{field}.evidence_reuse is invalid")
    return AgentExpectation(actions, minimum, maximum, reuse)


def load_agent_cases(path: Path) -> tuple[int, tuple[AgentCase, ...]]:
    """Load and validate conversational Agent scenarios."""
    version, raw_cases = _read_dataset(path, "cases")
    parsed: list[AgentCase] = []
    seen: set[str] = set()
    for raw in raw_cases:
        if not isinstance(raw, dict):
            raise EvalDataError("agent cases must be JSON objects")
        case_id = _clean_string(raw.get("id"), field="agent case id")
        if case_id in seen:
            raise EvalDataError(f"duplicate agent case id: {case_id}")
        seen.add(case_id)
        raw_turns = raw.get("turns")
        if not isinstance(raw_turns, list) or not raw_turns:
            raise EvalDataError(f"{case_id}.turns must be non-empty")
        turns: list[AgentTurn] = []
        for index, raw_turn in enumerate(raw_turns, start=1):
            if not isinstance(raw_turn, dict):
                raise EvalDataError(f"{case_id}.turns[{index}] must be an object")
            question = _clean_string(
                raw_turn.get("question"), field=f"{case_id}.turns[{index}].question"
            )
            expected = raw_turn.get("expected")
            turns.append(
                AgentTurn(
                    question,
                    None
                    if expected is None
                    else _parse_expectation(
                        expected, field=f"{case_id}.turns[{index}].expected"
                    ),
                )
            )
        parsed.append(
            AgentCase(
                id=case_id,
                turns=tuple(turns),
                tags=_tags(raw.get("tags"), field=f"{case_id}.tags"),
            )
        )
    return version, tuple(parsed)


def load_answer_cases(path: Path) -> tuple[int, tuple[AnswerCase, ...]]:
    """Load and validate final-answer cases."""
    version, raw_cases = _read_dataset(path, "cases")
    parsed: list[AnswerCase] = []
    seen: set[str] = set()
    for raw in raw_cases:
        if not isinstance(raw, dict):
            raise EvalDataError("answer cases must be JSON objects")
        case_id = _clean_string(raw.get("id"), field="answer case id")
        if case_id in seen:
            raise EvalDataError(f"duplicate answer case id: {case_id}")
        seen.add(case_id)
        answerable = raw.get("answerable")
        if not isinstance(answerable, bool):
            raise EvalDataError(f"{case_id}.answerable must be boolean")
        must_refuse = raw.get("must_refuse", not answerable)
        if not isinstance(must_refuse, bool):
            raise EvalDataError(f"{case_id}.must_refuse must be boolean")
        parsed.append(
            AnswerCase(
                id=case_id,
                question=_clean_string(raw.get("question"), field=f"{case_id}.question"),
                answerable=answerable,
                expected_citations=_string_tuple(
                    raw.get("expected_citations", []),
                    field=f"{case_id}.expected_citations",
                ),
                required_phrases=_string_tuple(
                    raw.get("required_phrases", []),
                    field=f"{case_id}.required_phrases",
                ),
                must_refuse=must_refuse,
                tags=_tags(raw.get("tags"), field=f"{case_id}.tags"),
            )
        )
    return version, tuple(parsed)


def load_memory_cases(path: Path) -> tuple[int, tuple[MemoryCase, ...]]:
    """Load and validate long-conversation memory cases."""
    version, raw_cases = _read_dataset(path, "cases")
    parsed: list[MemoryCase] = []
    seen: set[str] = set()
    for raw in raw_cases:
        if not isinstance(raw, dict):
            raise EvalDataError("memory cases must be JSON objects")
        case_id = _clean_string(raw.get("id"), field="memory case id")
        if case_id in seen:
            raise EvalDataError(f"duplicate memory case id: {case_id}")
        seen.add(case_id)
        min_compactions = raw.get("min_compactions", 1)
        if (
            not isinstance(min_compactions, int)
            or isinstance(min_compactions, bool)
            or min_compactions < 1
        ):
            raise EvalDataError(f"{case_id}.min_compactions must be positive")
        parsed.append(
            MemoryCase(
                id=case_id,
                turns=_string_tuple(raw.get("turns"), field=f"{case_id}.turns"),
                required_memory_phrases=_string_tuple(
                    raw.get("required_memory_phrases"),
                    field=f"{case_id}.required_memory_phrases",
                ),
                min_compactions=min_compactions,
                tags=_tags(raw.get("tags"), field=f"{case_id}.tags"),
            )
        )
    return version, tuple(parsed)

