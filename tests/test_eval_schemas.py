from pathlib import Path

import pytest

from eval.schemas import (
    EvalDataError,
    load_agent_cases,
    load_answer_cases,
    load_memory_cases,
    load_retrieval_cases,
)


CASES = Path("eval/cases")


def test_loads_all_tracked_eval_datasets() -> None:
    retrieval_version, retrieval = load_retrieval_cases(CASES / "retrieval.json")
    agent_version, agent = load_agent_cases(CASES / "agent.json")
    answer_version, answer = load_answer_cases(CASES / "answer.json")
    memory_version, memory = load_memory_cases(CASES / "memory.json")

    assert retrieval_version == 2
    assert len(retrieval) == 13
    assert agent_version == 2
    assert len(agent) == 4
    assert answer_version == 1
    assert len(answer) == 4
    assert memory_version == 1
    assert len(memory) == 3


def test_retrieval_loader_rejects_answerable_case_without_anchor(tmp_path: Path) -> None:
    path = tmp_path / "retrieval.json"
    path.write_text(
        '{"dataset_version": 1, "cases": [{"id": "x", "query": "q", '
        '"answerable": true, "must_find_ids": [], "relevance_grades": {}}]}',
        encoding="utf-8",
    )

    with pytest.raises(EvalDataError, match="must_find_ids"):
        load_retrieval_cases(path)


def test_agent_loader_rejects_retrieval_budget_above_production_limit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "agent.json"
    path.write_text(
        '{"dataset_version": 1, "cases": [{"id": "x", "turns": [{'
        '"question": "q", "expected": {"allowed_actions": ["respond"], '
        '"min_retrievals": 0, "max_retrievals": 3, '
        '"evidence_reuse": "optional"}}]}]}',
        encoding="utf-8",
    )

    with pytest.raises(EvalDataError, match="retrieval bounds"):
        load_agent_cases(path)

