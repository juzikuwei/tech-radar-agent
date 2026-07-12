import json
from pathlib import Path

import pytest

from eval.run_agent_eval import (
    aggregate_scenarios,
    evaluate_expected_behavior,
    load_and_validate_scenarios,
    run_scenario,
)
from rag.application import RagResult
from rag.conversation import ConversationDecision, ConversationTurn
from rag.search import SearchResult


def make_paper(arxiv_id: str = "2607.00001") -> SearchResult:
    return SearchResult(
        arxiv_id=arxiv_id,
        title="Paper",
        document="Evidence",
        entry_url=f"https://arxiv.org/abs/{arxiv_id}",
        primary_category="cs.AI",
        similarity=0.8,
    )


def make_result(
    *,
    action: str | None,
    retrieval_attempts: int,
    reusable_ids: tuple[str, ...] = (),
    answer: str | None = "answer",
) -> RagResult:
    decision = None
    if action is not None:
        coverage = {
            "answer_from_existing": "sufficient",
            "retrieve_missing": "partial",
            "fresh_retrieval": "unrelated",
        }[action]
        decision = ConversationDecision(
            coverage=coverage,  # type: ignore[arg-type]
            next_action=action,  # type: ignore[arg-type]
            reason="test decision",
            standalone_question="standalone",
            reusable_arxiv_ids=reusable_ids,
            missing_aspects=(),
            retrieval_query=None if action == "answer_from_existing" else "query",
        )
    return RagResult(
        question="question",
        papers=(make_paper(),),
        answer=answer,
        generation_error=None if answer is not None else "failed",
        retrieval_attempts=retrieval_attempts,
        conversation_decision=decision,
    )


def test_loads_repository_agent_scenarios() -> None:
    dataset = load_and_validate_scenarios(Path("eval/agent_scenarios.json"))

    assert dataset["dataset_version"] == 1
    assert len(dataset["scenarios"]) == 3


def test_rejects_invalid_retrieval_bounds(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "id": "bad",
                        "turns": [
                            {
                                "question": "q",
                                "expected": {
                                    "allowed_actions": ["fresh_retrieval"],
                                    "min_retrievals": 2,
                                    "max_retrievals": 1,
                                    "evidence_reuse": "forbidden",
                                },
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="retrieval bounds"):
        load_and_validate_scenarios(path)


def test_evaluates_action_retrieval_and_required_reuse() -> None:
    expected = {
        "allowed_actions": ["answer_from_existing"],
        "min_retrievals": 0,
        "max_retrievals": 0,
        "evidence_reuse": "required",
    }

    evaluation = evaluate_expected_behavior(
        expected,
        make_result(
            action="answer_from_existing",
            retrieval_attempts=0,
            reusable_ids=("2607.00001",),
        ),
    )

    assert evaluation["passed"] is True
    assert all(evaluation["checks"].values())


def test_reports_forbidden_reuse_and_wrong_action() -> None:
    expected = {
        "allowed_actions": ["fresh_retrieval"],
        "min_retrievals": 1,
        "max_retrievals": 2,
        "evidence_reuse": "forbidden",
    }

    evaluation = evaluate_expected_behavior(
        expected,
        make_result(
            action="retrieve_missing",
            retrieval_attempts=1,
            reusable_ids=("2607.00001",),
        ),
    )

    assert evaluation["passed"] is False
    assert evaluation["checks"]["action"] is False
    assert evaluation["checks"]["evidence_reuse"] is False


def test_scenario_updates_history_and_active_evidence_between_turns() -> None:
    observed: list[tuple[int, tuple[str, ...]]] = []

    def fake_run_turn(
        question: str,
        history: tuple[ConversationTurn, ...],
        active_evidence: tuple[SearchResult, ...],
    ) -> RagResult:
        observed.append(
            (len(history), tuple(paper.arxiv_id for paper in active_evidence))
        )
        if not history:
            return make_result(action=None, retrieval_attempts=1)
        return make_result(
            action="answer_from_existing",
            retrieval_attempts=0,
            reusable_ids=("2607.00001",),
        )

    scenario = {
        "id": "two-turn",
        "turns": [
            {"question": "first"},
            {
                "question": "follow-up",
                "expected": {
                    "allowed_actions": ["answer_from_existing"],
                    "min_retrievals": 0,
                    "max_retrievals": 0,
                    "evidence_reuse": "required",
                },
            },
        ],
    }

    report = run_scenario(scenario, run_turn=fake_run_turn)

    assert observed == [(0, ()), (1, ("2607.00001",))]
    assert report["passed"] is True


def test_aggregate_reports_action_accuracy_and_retrieval_rate() -> None:
    scenario_reports = [
        {
            "passed": True,
            "turns": [
                {
                    "actual": {"retrieval_attempts": 1},
                    "evaluation": None,
                },
                {
                    "expected": {
                        "allowed_actions": ["answer_from_existing"],
                    },
                    "actual": {"retrieval_attempts": 0},
                    "evaluation": {
                        "passed": True,
                        "checks": {"action": True},
                    },
                },
            ],
        },
        {
            "passed": False,
            "turns": [
                {
                    "expected": {
                        "allowed_actions": ["fresh_retrieval"],
                    },
                    "actual": {"retrieval_attempts": 1},
                    "evaluation": {
                        "passed": False,
                        "checks": {"action": False},
                    },
                }
            ],
        },
    ]

    summary = aggregate_scenarios(scenario_reports)

    assert summary["scenario_pass_rate"] == pytest.approx(0.5)
    assert summary["action_accuracy"] == pytest.approx(0.5)
    assert summary["average_retrievals_per_turn"] == pytest.approx(2 / 3)
