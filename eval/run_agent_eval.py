"""Evaluate real conversational Agentic RAG actions against scenario expectations."""

import argparse
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any

from config.environment import load_repository_env

# Hugging Face reads endpoint settings while its modules are imported.
load_repository_env()

from config.model_settings import ModelSettings
from rag.application import RagResult, run_rag
from rag.conversation import ConversationTurn, MAX_CONVERSATION_TURNS
from rag.embedder import DEFAULT_MODEL_NAME, E5Embedder
from rag.keyword_search import DEFAULT_DATABASE_PATH, ensure_keyword_index
from rag.reranker import DEFAULT_RERANKER_MODEL, CrossEncoderReranker
from rag.search import SearchResult
from rag.vector_store import (
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION_NAME,
    get_persistent_collection,
)


DEFAULT_SCENARIOS_PATH = Path("eval/agent_scenarios.json")
DEFAULT_OUTPUT_PATH = Path("eval/results/agent_baseline.json")
ALLOWED_ACTIONS = {
    "answer_from_existing",
    "retrieve_missing",
    "fresh_retrieval",
}
EVIDENCE_REUSE_POLICIES = {"required", "forbidden", "optional"}

RunTurn = Callable[
    [str, tuple[ConversationTurn, ...], tuple[SearchResult, ...]],
    RagResult,
]


def load_and_validate_scenarios(path: Path) -> dict[str, Any]:
    """Load a small scenario dataset and validate its expected actions."""
    dataset = json.loads(path.read_text(encoding="utf-8"))
    scenarios = dataset.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("agent evaluation needs a non-empty scenarios list")

    seen_ids: set[str] = set()
    for scenario in scenarios:
        scenario_id = scenario.get("id")
        if not isinstance(scenario_id, str) or not scenario_id.strip():
            raise ValueError("every scenario needs a non-empty id")
        if scenario_id in seen_ids:
            raise ValueError(f"duplicate scenario id: {scenario_id}")
        seen_ids.add(scenario_id)

        turns = scenario.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"scenario {scenario_id} needs non-empty turns")
        for turn_index, turn in enumerate(turns, start=1):
            question = turn.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(
                    f"scenario {scenario_id} turn {turn_index} needs a question"
                )
            expected = turn.get("expected")
            if expected is not None:
                _validate_expectation(expected, scenario_id, turn_index)

    return dataset


def _validate_expectation(
    expected: object,
    scenario_id: str,
    turn_index: int,
) -> None:
    if not isinstance(expected, dict):
        raise ValueError(
            f"scenario {scenario_id} turn {turn_index} expected must be an object"
        )
    allowed_actions = expected.get("allowed_actions")
    if not isinstance(allowed_actions, list) or not allowed_actions:
        raise ValueError(
            f"scenario {scenario_id} turn {turn_index} needs allowed_actions"
        )
    if not all(action in ALLOWED_ACTIONS for action in allowed_actions):
        raise ValueError(
            f"scenario {scenario_id} turn {turn_index} has invalid actions"
        )
    if len(set(allowed_actions)) != len(allowed_actions):
        raise ValueError(
            f"scenario {scenario_id} turn {turn_index} has duplicate actions"
        )

    minimum = expected.get("min_retrievals")
    maximum = expected.get("max_retrievals")
    if (
        not isinstance(minimum, int)
        or isinstance(minimum, bool)
        or not isinstance(maximum, int)
        or isinstance(maximum, bool)
        or minimum < 0
        or maximum > 2
        or minimum > maximum
    ):
        raise ValueError(
            f"scenario {scenario_id} turn {turn_index} has invalid retrieval bounds"
        )

    reuse_policy = expected.get("evidence_reuse")
    if reuse_policy not in EVIDENCE_REUSE_POLICIES:
        raise ValueError(
            f"scenario {scenario_id} turn {turn_index} has invalid reuse policy"
        )


def evaluate_expected_behavior(
    expected: dict[str, Any],
    result: RagResult,
) -> dict[str, Any]:
    """Compare one real RagResult with deterministic control expectations."""
    decision = result.conversation_decision
    actual_action = decision.next_action if decision is not None else None
    reusable_ids = list(decision.reusable_arxiv_ids) if decision is not None else []
    failures: list[str] = []

    action_passed = actual_action in expected["allowed_actions"]
    if not action_passed:
        failures.append(
            f"action {actual_action!r} not in {expected['allowed_actions']}"
        )

    retrieval_passed = (
        expected["min_retrievals"]
        <= result.retrieval_attempts
        <= expected["max_retrievals"]
    )
    if not retrieval_passed:
        failures.append(
            "retrieval_attempts "
            f"{result.retrieval_attempts} outside "
            f"[{expected['min_retrievals']}, {expected['max_retrievals']}]"
        )

    reuse_policy = expected["evidence_reuse"]
    reuse_passed = True
    if reuse_policy == "required" and not reusable_ids:
        reuse_passed = False
        failures.append("expected reusable evidence but none was selected")
    elif reuse_policy == "forbidden" and reusable_ids:
        reuse_passed = False
        failures.append(f"unexpected reusable evidence: {reusable_ids}")

    generation_passed = result.answer is not None
    if not generation_passed:
        failures.append(
            f"answer generation failed: {result.generation_error or 'no answer'}"
        )

    budget_passed = result.retrieval_attempts <= 2
    if not budget_passed:
        failures.append("retrieval budget exceeded two attempts")

    return {
        "passed": not failures,
        "failures": failures,
        "checks": {
            "action": action_passed,
            "retrieval_bounds": retrieval_passed,
            "evidence_reuse": reuse_passed,
            "answer_generated": generation_passed,
            "retrieval_budget": budget_passed,
        },
    }


def run_scenario(
    scenario: dict[str, Any],
    *,
    run_turn: RunTurn,
) -> dict[str, Any]:
    """Run one scenario with the same six-turn and active-evidence updates as UI."""
    history: tuple[ConversationTurn, ...] = ()
    active_evidence: tuple[SearchResult, ...] = ()
    turn_reports: list[dict[str, Any]] = []
    scenario_errors: list[str] = []

    for turn_index, turn in enumerate(scenario["turns"], start=1):
        question = str(turn["question"]).strip()
        started_at = perf_counter()
        try:
            result = run_turn(question, history, active_evidence)
        except Exception as error:
            scenario_errors.append(
                f"turn {turn_index} raised {type(error).__name__}: {error}"
            )
            turn_reports.append(
                {
                    "turn": turn_index,
                    "question": question,
                    "passed": False,
                    "error": str(error),
                }
            )
            break

        duration_ms = (perf_counter() - started_at) * 1_000
        decision = result.conversation_decision
        expected = turn.get("expected")
        evaluation = (
            evaluate_expected_behavior(expected, result)
            if isinstance(expected, dict)
            else None
        )
        turn_passed = (
            evaluation["passed"]
            if evaluation is not None
            else result.answer is not None
        )
        turn_reports.append(
            {
                "turn": turn_index,
                "question": question,
                "expected": expected,
                "actual": {
                    "action": decision.next_action if decision else None,
                    "coverage": decision.coverage if decision else None,
                    "reason": decision.reason if decision else None,
                    "standalone_question": result.standalone_question,
                    "retrieval_query": decision.retrieval_query if decision else None,
                    "reusable_arxiv_ids": (
                        list(decision.reusable_arxiv_ids) if decision else []
                    ),
                    "retrieval_attempts": result.retrieval_attempts,
                    "paper_ids": [paper.arxiv_id for paper in result.papers],
                    "answer_generated": result.answer is not None,
                    "generation_error": result.generation_error,
                    "trace_stages": [event.stage for event in result.trace],
                    "duration_ms": duration_ms,
                },
                "evaluation": evaluation,
                "passed": turn_passed,
            }
        )

        if result.answer is not None:
            history = (
                *history,
                ConversationTurn(question, result.answer),
            )[-MAX_CONVERSATION_TURNS:]
            active_evidence = result.papers

    scenario_passed = not scenario_errors and all(
        bool(turn.get("passed")) for turn in turn_reports
    )
    return {
        "id": scenario["id"],
        "description": scenario.get("description"),
        "passed": scenario_passed,
        "errors": scenario_errors,
        "turns": turn_reports,
    }


def aggregate_scenarios(
    scenario_reports: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate deterministic action and retrieval metrics."""
    evaluated_turns = [
        turn
        for scenario in scenario_reports
        for turn in scenario["turns"]
        if turn.get("evaluation") is not None
    ]
    executed_turns = [
        turn
        for scenario in scenario_reports
        for turn in scenario["turns"]
        if "actual" in turn
    ]
    action_hits = sum(
        bool(turn["evaluation"]["checks"]["action"])
        for turn in evaluated_turns
    )
    retrieval_counts = [
        int(turn["actual"]["retrieval_attempts"])
        for turn in executed_turns
    ]
    unnecessary_retrievals = sum(
        turn["expected"]["allowed_actions"] == ["answer_from_existing"]
        and int(turn["actual"]["retrieval_attempts"]) > 0
        for turn in evaluated_turns
    )
    expected_reuse_turns = sum(
        turn["expected"]["allowed_actions"] == ["answer_from_existing"]
        for turn in evaluated_turns
    )
    return {
        "scenario_count": len(scenario_reports),
        "passed_scenarios": sum(bool(report["passed"]) for report in scenario_reports),
        "scenario_pass_rate": (
            sum(bool(report["passed"]) for report in scenario_reports)
            / len(scenario_reports)
            if scenario_reports
            else 0.0
        ),
        "evaluated_turn_count": len(evaluated_turns),
        "passed_turn_count": sum(
            bool(turn["evaluation"]["passed"]) for turn in evaluated_turns
        ),
        "action_accuracy": (
            action_hits / len(evaluated_turns) if evaluated_turns else 0.0
        ),
        "average_retrievals_per_turn": (
            fmean(retrieval_counts) if retrieval_counts else 0.0
        ),
        "unnecessary_retrieval_rate": (
            unnecessary_retrievals / expected_reuse_turns
            if expected_reuse_turns
            else 0.0
        ),
        "retrieval_budget_violations": sum(
            count > 2 for count in retrieval_counts
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build arguments for the real Agent behavior baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chroma-path", type=Path, default=DEFAULT_CHROMA_PATH)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    return parser


def main() -> int:
    """Run every scenario against the current real Agent and write a report."""
    args = build_parser().parse_args()
    if args.top_k <= 0:
        print("Evaluation failed: top-k must be greater than zero")
        return 1

    try:
        dataset = load_and_validate_scenarios(args.scenarios)
        settings = ModelSettings.from_env()
        ensure_keyword_index(args.database)
        collection = get_persistent_collection(args.chroma_path, args.collection)
        if collection.count() == 0:
            raise ValueError("ChromaDB collection is empty")
        embedder = E5Embedder(args.embedding_model)
        reranker = CrossEncoderReranker(args.reranker_model)

        def run_turn(
            question: str,
            history: tuple[ConversationTurn, ...],
            active_evidence: tuple[SearchResult, ...],
        ) -> RagResult:
            return run_rag(
                question,
                top_k=args.top_k,
                collection=collection,
                embedder=embedder,
                reranker=reranker,
                database_path=args.database,
                settings=settings,
                conversation_history=history,
                active_evidence=active_evidence,
            )

        scenario_reports = [
            run_scenario(scenario, run_turn=run_turn)
            for scenario in dataset["scenarios"]
        ]
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_version": dataset.get("dataset_version"),
            "model": settings.model,
            "base_url": settings.base_url,
            "embedding_model": args.embedding_model,
            "reranker_model": args.reranker_model,
            "collection_size": collection.count(),
            "top_k": args.top_k,
            "summary": aggregate_scenarios(scenario_reports),
            "scenarios": scenario_reports,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Evaluation failed: {error}")
        return 1

    summary = report["summary"]
    for scenario in scenario_reports:
        status = "PASS" if scenario["passed"] else "FAIL"
        print(f"{status} {scenario['id']}")
    print(f"Scenarios: {summary['scenario_count']}")
    print(f"Scenario pass rate: {summary['scenario_pass_rate']:.2%}")
    print(f"Action accuracy: {summary['action_accuracy']:.2%}")
    print(
        "Average retrievals per turn: "
        f"{summary['average_retrievals_per_turn']:.2f}"
    )
    print(
        "Unnecessary retrieval rate: "
        f"{summary['unnecessary_retrieval_rate']:.2%}"
    )
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
