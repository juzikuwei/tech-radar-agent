import pytest

from eval.run_retrieval_eval import (
    aggregate_results,
    calculate_question_metrics,
)


def test_calculate_question_metrics_supports_strict_and_relaxed_labels() -> None:
    metrics = calculate_question_metrics(
        ["direct", "partial", "noise-1", "noise-2", "noise-3"],
        relevant_ids={"direct"},
        partially_relevant_ids={"partial"},
        answerable=True,
    )

    assert metrics["strict_precision"] == pytest.approx(0.2)
    assert metrics["relaxed_precision"] == pytest.approx(0.4)
    assert metrics["strict_recall"] == pytest.approx(1.0)
    assert metrics["relaxed_recall"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(1.0)


def test_mrr_uses_first_directly_relevant_rank() -> None:
    metrics = calculate_question_metrics(
        ["noise-1", "partial", "noise-2", "direct"],
        relevant_ids={"direct"},
        partially_relevant_ids={"partial"},
        answerable=True,
    )

    assert metrics["mrr"] == pytest.approx(0.25)


def test_unanswerable_question_does_not_calculate_recall() -> None:
    metrics = calculate_question_metrics(
        ["noise-1", "noise-2"],
        relevant_ids=set(),
        partially_relevant_ids=set(),
        answerable=False,
    )

    assert all(value is None for value in metrics.values())


def test_aggregate_results_separates_unanswerable_scores() -> None:
    answerable_metrics = {
        "strict_precision": 0.2,
        "relaxed_precision": 0.4,
        "strict_recall": 1.0,
        "relaxed_recall": 1.0,
        "mrr": 1.0,
    }
    results = [
        {
            "answerable": True,
            "metrics": answerable_metrics,
            "top_similarity": 0.83,
        },
        {
            "answerable": False,
            "metrics": {key: None for key in answerable_metrics},
            "top_similarity": 0.79,
        },
        {
            "answerable": False,
            "metrics": {key: None for key in answerable_metrics},
            "top_similarity": 0.81,
        },
    ]

    summary = aggregate_results(results)

    assert summary["answerable"]["mean_strict_precision"] == pytest.approx(0.2)
    assert summary["unanswerable"]["mean_top_similarity"] == pytest.approx(0.8)
    assert summary["unanswerable"]["max_top_similarity"] == pytest.approx(0.81)
