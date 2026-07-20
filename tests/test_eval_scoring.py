from types import SimpleNamespace

from eval.schemas import AgentExpectation, AnswerCase, MemoryCase, RetrievalCase
from eval.scoring import (
    aggregate_retrieval_scores,
    citation_ids,
    score_agent_turn,
    score_answer_case,
    score_memory_case,
    score_retrieval_case,
)


def test_retrieval_scores_anchor_hit_mrr_and_ndcg() -> None:
    case = RetrievalCase(
        id="q",
        query="query",
        answerable=True,
        must_find_ids=("a",),
        relevance_grades={"a": 3, "b": 1},
    )

    score = score_retrieval_case(case, ["b", "a"], top_k=2)

    assert score["metrics"]["hit_at_k"] == 1.0
    assert score["metrics"]["mrr"] == 0.5
    assert score["metrics"]["ndcg_at_k"] is not None


def test_unanswerable_retrieval_is_not_counted_as_recall() -> None:
    case = RetrievalCase(
        id="q",
        query="query",
        answerable=False,
        must_find_ids=(),
        relevance_grades={},
    )

    score = score_retrieval_case(
        case,
        ["a"],
        top_k=5,
        top_similarity=0.72,
        top_rerank_score=4.2,
    )
    summary = aggregate_retrieval_scores([score])

    assert score["metrics"]["mrr"] is None
    assert summary["answerable"]["question_count"] == 0
    assert summary["unanswerable"]["max_top_similarity"] == 0.72
    assert summary["unanswerable"]["max_top_rerank_score"] == 4.2


def test_agent_scoring_checks_action_reuse_and_budget() -> None:
    result = SimpleNamespace(
        answer="回答",
        retrieval_attempts=0,
        conversation_decision=SimpleNamespace(
            next_action="answer_from_existing",
            reusable_arxiv_ids=("a",),
        ),
    )
    expected = AgentExpectation(
        allowed_actions=("answer_from_existing",),
        min_retrievals=0,
        max_retrievals=0,
        evidence_reuse="required",
    )

    assert score_agent_turn(expected, result)["passed"] is True


def test_answer_scoring_rejects_unknown_citation_and_accepts_refusal() -> None:
    answerable = AnswerCase(
        id="a",
        question="q",
        answerable=True,
        expected_citations=("2607.07989",),
    )
    valid = SimpleNamespace(
        answer="结论 [2607.07989]",
        papers=(SimpleNamespace(arxiv_id="2607.07989"),),
    )
    invalid = SimpleNamespace(
        answer="结论 [2607.00001]",
        papers=(SimpleNamespace(arxiv_id="2607.07989"),),
    )
    refusal = AnswerCase(
        id="r",
        question="q",
        answerable=False,
        must_refuse=True,
    )

    assert score_answer_case(answerable, valid)["passed"] is True
    assert score_answer_case(answerable, invalid)["passed"] is False
    assert score_answer_case(refusal, SimpleNamespace(answer="证据不足", papers=()))[
        "passed"
    ] is True
    assert citation_ids("a [2607.07989v2] b [2607.07989]") == ("2607.07989",)


def test_memory_scoring_requires_compaction_and_constraint_retention() -> None:
    case = MemoryCase(
        id="m",
        turns=("first", "second"),
        required_memory_phrases=("本地 arXiv", "不要网页"),
        min_compactions=2,
    )

    score = score_memory_case(
        case,
        summary="保留本地 arXiv 证据，并记录不要网页作为引用来源。",
        compaction_count=2,
    )

    assert score["passed"] is True
