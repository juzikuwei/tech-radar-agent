from types import SimpleNamespace

import numpy as np

from config.model_settings import ModelSettings
from rag.answer_validation import (
    SAFE_INSUFFICIENT_EVIDENCE_RESPONSE,
    SAFE_UNVERIFIED_ANSWER_RESPONSE,
)
from rag.application import answer_from_results, run_rag
from rag.conversation import (
    ConversationDecision,
    ConversationDecisionError,
    ConversationTurn,
)
from rag.llm_client import LLMRequestError
from rag.retrieval_judge import RetrievalDecision, RetrievalDecisionError
from rag.search import SearchResult


class FakeCompletions:
    def create(self, **options: object) -> object:
        assert "response_format" not in options
        content = "A supported result. [2607.00001]"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def test_pipeline_connects_prompt_and_model_text_output() -> None:
    result = SearchResult(
        arxiv_id="2607.00001",
        title="A paper",
        document="A supported abstract.",
        entry_url="https://arxiv.org/abs/2607.00001",
        primary_category="cs.AI",
        similarity=0.9,
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )

    answer = answer_from_results(
        "What is supported?",
        [result],
        settings=ModelSettings("key", "https://example.test", "model"),
        client=client,
    )

    assert answer == "A supported result. [2607.00001]"


class FakeCollection:
    def count(self) -> int:
        return 1

    def query(self, **options: object) -> dict[str, object]:
        return {
            "ids": [["2607.00001"]],
            "documents": [["A supported abstract."]],
            "metadatas": [[{
                "title": "A paper",
                "entry_url": "https://arxiv.org/abs/2607.00001",
                "primary_category": "cs.AI",
            }]],
            "distances": [[0.1]],
        }


class FakeEmbedder:
    def encode_query(self, query: str) -> object:
        import numpy as np

        return np.asarray([1.0, 0.0])


class FailingCompletions:
    def create(self, **options: object) -> object:
        raise RuntimeError("provider failed")


def test_run_rag_preserves_papers_when_generation_fails(monkeypatch: object) -> None:
    def fail_generation(*args: object, **options: object) -> str:
        raise LLMRequestError("provider failed")

    monkeypatch.setattr("rag.application.answer_from_results", fail_generation)

    result = run_rag(
        "What is supported?",
        top_k=1,
        collection=FakeCollection(),
        embedder=FakeEmbedder(),
        settings=ModelSettings("key", "https://example.test", "model"),
    )

    assert result.answer is None
    assert result.generation_error == "provider failed"
    assert [paper.arxiv_id for paper in result.papers] == ["2607.00001"]


def make_search_result(arxiv_id: str, document: str) -> SearchResult:
    return SearchResult(
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        document=document,
        entry_url=f"https://arxiv.org/abs/{arxiv_id}",
        primary_category="cs.AI",
        similarity=0.8,
        rerank_score=0.5,
    )


class OriginalQuestionReranker:
    model_name = "fake-reranker"

    def score(self, query: str, documents: list[str]) -> np.ndarray:
        assert query == "original question"
        weights = {"initial A": 0.2, "shared B": 0.7, "rewritten C": 0.9}
        return np.asarray([weights[document] for document in documents])


def test_insufficient_evidence_triggers_one_rewrite_and_original_rerank(
    monkeypatch: object,
) -> None:
    calls: list[str] = []

    def fake_hybrid_search(query: str, **_: object) -> list[SearchResult]:
        calls.append(query)
        if query == "original question":
            return [
                make_search_result("A", "initial A"),
                make_search_result("B", "shared B"),
            ]
        assert query == "rewritten query"
        return [
            make_search_result("B", "shared B"),
            make_search_result("C", "rewritten C"),
        ]

    monkeypatch.setattr("rag.application.hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(
        "rag.application.judge_retrieval",
        lambda *args, **kwargs: RetrievalDecision(
            sufficient=False,
            reason="missing direct evidence",
            missing_aspects=("specific mechanism",),
            rewritten_query="rewritten query",
        ),
    )
    monkeypatch.setattr(
        "rag.application.answer_from_results",
        lambda question, results, **kwargs: "grounded answer [C] [B]",
    )

    result = run_rag(
        "original question",
        top_k=2,
        collection=object(),
        embedder=object(),
        reranker=OriginalQuestionReranker(),
        settings=ModelSettings("key", "https://example.test", "model"),
    )

    assert calls == ["original question", "rewritten query"]
    assert result.retrieval_attempts == 2
    assert [paper.arxiv_id for paper in result.papers] == ["C", "B"]
    assert result.answer == "grounded answer [C] [B]"
    assert [event.stage for event in result.trace] == [
        "retrieval_judgment",
        "final_union_rerank",
        "answer_generation",
        "answer_validation",
    ]


def test_sufficient_evidence_skips_second_retrieval(monkeypatch: object) -> None:
    calls: list[str] = []

    def fake_hybrid_search(query: str, **_: object) -> list[SearchResult]:
        calls.append(query)
        return [make_search_result("A", "initial A")]

    monkeypatch.setattr("rag.application.hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(
        "rag.application.judge_retrieval",
        lambda *args, **kwargs: RetrievalDecision(
            sufficient=True,
            reason="direct evidence exists",
            missing_aspects=(),
            rewritten_query=None,
        ),
    )
    monkeypatch.setattr(
        "rag.application.answer_from_results",
        lambda question, results, **kwargs: "grounded answer [A]",
    )

    result = run_rag(
        "original question",
        top_k=1,
        collection=object(),
        embedder=object(),
        reranker=OriginalQuestionReranker(),
        settings=ModelSettings("key", "https://example.test", "model"),
    )

    assert calls == ["original question"]
    assert result.retrieval_attempts == 1
    assert result.retrieval_decision is not None
    assert result.retrieval_decision.sufficient is True
    assert [event.stage for event in result.trace] == [
        "retrieval_judgment",
        "answer_generation",
        "answer_validation",
    ]


def test_invalid_judge_output_falls_back_to_initial_results(
    monkeypatch: object,
) -> None:
    monkeypatch.setattr(
        "rag.application.hybrid_search",
        lambda *args, **kwargs: [make_search_result("A", "initial A")],
    )

    def fail_judgment(*args: object, **kwargs: object) -> RetrievalDecision:
        raise RetrievalDecisionError("invalid decision")

    monkeypatch.setattr("rag.application.judge_retrieval", fail_judgment)
    monkeypatch.setattr(
        "rag.application.answer_from_results",
        lambda question, results, **kwargs: "fallback answer [A]",
    )

    result = run_rag(
        "original question",
        top_k=1,
        collection=object(),
        embedder=object(),
        reranker=OriginalQuestionReranker(),
        settings=ModelSettings("key", "https://example.test", "model"),
    )

    assert result.answer == "fallback answer [A]"
    assert result.retrieval_attempts == 1
    assert result.retrieval_decision_error == "invalid decision"
    assert result.trace[0].stage == "retrieval_judgment"
    assert result.trace[0].status == "failed"


class FollowupReranker:
    model_name = "followup-reranker"

    def score(self, query: str, documents: list[str]) -> np.ndarray:
        weights = {
            "old direct evidence": 0.9,
            "old partial evidence": 0.5,
            "new missing evidence": 0.95,
            "fresh topic evidence": 0.85,
        }
        return np.asarray([weights[document] for document in documents])


HISTORY = (ConversationTurn("first question", "first answer"),)


def test_followup_feedback_responds_without_retrieval(monkeypatch: object) -> None:
    old = make_search_result("OLD", "old direct evidence")
    monkeypatch.setattr(
        "rag.application.decide_conversation_action",
        lambda *args, **kwargs: ConversationDecision(
            coverage="not_applicable",
            next_action="respond",
            reason="the user is giving feedback",
            standalone_question="回应用户反馈",
            reusable_arxiv_ids=(),
            missing_aspects=(),
            retrieval_query=None,
        ),
    )
    monkeypatch.setattr(
        "rag.application.hybrid_search",
        lambda *args, **kwargs: pytest.fail("feedback must not trigger retrieval"),
    )
    monkeypatch.setattr(
        "rag.application.generate_conversational_response",
        lambda *args, **kwargs: "你说得对。你希望我先检查哪一个结论？",
    )

    result = run_rag(
        "感觉你在乱说",
        top_k=5,
        collection=object(),
        embedder=object(),
        reranker=FollowupReranker(),
        settings=ModelSettings("key", "https://example.test", "model"),
        conversation_history=HISTORY,
        active_evidence=(old,),
    )

    assert result.answer == "你说得对。你希望我先检查哪一个结论？"
    assert result.papers == ()
    assert result.retrieval_attempts == 0
    assert result.response_kind == "conversation"
    assert [event.stage for event in result.trace] == [
        "conversation_evidence_decision",
        "conversation_response",
    ]


def test_invalid_followup_decision_requests_clarification_without_retrieval(
    monkeypatch: object,
) -> None:
    monkeypatch.setattr(
        "rag.application.decide_conversation_action",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ConversationDecisionError("invalid action")
        ),
    )
    monkeypatch.setattr(
        "rag.application.hybrid_search",
        lambda *args, **kwargs: pytest.fail(
            "invalid conversation decision must not retrieve raw user text"
        ),
    )

    result = run_rag(
        "感觉你在乱说",
        top_k=5,
        collection=object(),
        embedder=object(),
        reranker=FollowupReranker(),
        settings=ModelSettings("key", "https://example.test", "model"),
        conversation_history=HISTORY,
    )

    assert result.answer is not None
    assert "请明确告诉我" in result.answer
    assert result.retrieval_attempts == 0
    assert result.response_kind == "conversation"
    assert [event.stage for event in result.trace] == [
        "conversation_evidence_decision",
        "conversation_clarification",
    ]


def test_followup_reuses_sufficient_evidence_without_retrieval(
    monkeypatch: object,
) -> None:
    old = make_search_result("OLD", "old direct evidence")
    monkeypatch.setattr(
        "rag.application.decide_conversation_action",
        lambda *args, **kwargs: ConversationDecision(
            coverage="sufficient",
            next_action="answer_from_existing",
            reason="old evidence directly answers the follow-up",
            standalone_question="resolved follow-up",
            reusable_arxiv_ids=("OLD",),
            missing_aspects=(),
            retrieval_query=None,
        ),
    )
    monkeypatch.setattr(
        "rag.application.hybrid_search",
        lambda *args, **kwargs: pytest.fail("retrieval should be skipped"),
    )
    monkeypatch.setattr(
        "rag.application.judge_retrieval",
        lambda *args, **kwargs: pytest.fail("retrieval judge should be skipped"),
    )
    monkeypatch.setattr(
        "rag.application.answer_from_results",
        lambda question, results, **kwargs: "answer from old evidence [OLD]",
    )

    result = run_rag(
        "它能定位具体步骤吗？",
        top_k=5,
        collection=object(),
        embedder=object(),
        reranker=FollowupReranker(),
        settings=ModelSettings("key", "https://example.test", "model"),
        conversation_history=HISTORY,
        active_evidence=(old,),
    )

    assert result.retrieval_attempts == 0
    assert [paper.arxiv_id for paper in result.papers] == ["OLD"]
    assert [event.stage for event in result.trace] == [
        "conversation_evidence_decision",
        "active_evidence_rerank",
        "answer_generation",
        "answer_validation",
    ]


def test_followup_retrieves_only_missing_evidence_and_combines_it(
    monkeypatch: object,
) -> None:
    old = make_search_result("OLD", "old partial evidence")
    new = make_search_result("NEW", "new missing evidence")
    calls: list[str] = []

    monkeypatch.setattr(
        "rag.application.decide_conversation_action",
        lambda *args, **kwargs: ConversationDecision(
            coverage="partial",
            next_action="retrieve_missing",
            reason="coding-agent evaluation is missing",
            standalone_question="resolved follow-up",
            reusable_arxiv_ids=("OLD",),
            missing_aspects=("coding agent evaluation",),
            retrieval_query="targeted missing evidence query",
        ),
    )

    def fake_hybrid_search(query: str, **_: object) -> list[SearchResult]:
        calls.append(query)
        return [new]

    monkeypatch.setattr("rag.application.hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(
        "rag.application.judge_retrieval",
        lambda *args, **kwargs: RetrievalDecision(
            sufficient=True,
            reason="combined evidence is sufficient",
            missing_aspects=(),
            rewritten_query=None,
        ),
    )
    monkeypatch.setattr(
        "rag.application.answer_from_results",
        lambda question, results, **kwargs: "combined answer [NEW] [OLD]",
    )

    result = run_rag(
        "这种方法在代码 Agent 中表现如何？",
        top_k=5,
        collection=object(),
        embedder=object(),
        reranker=FollowupReranker(),
        settings=ModelSettings("key", "https://example.test", "model"),
        conversation_history=HISTORY,
        active_evidence=(old,),
    )

    assert calls == ["targeted missing evidence query"]
    assert result.retrieval_attempts == 1
    assert [paper.arxiv_id for paper in result.papers] == ["NEW", "OLD"]
    assert result.retrieval_decision is not None
    assert result.retrieval_decision.sufficient is True


def test_new_topic_discards_previous_evidence(monkeypatch: object) -> None:
    old = make_search_result("OLD", "old direct evidence")
    fresh = make_search_result("FRESH", "fresh topic evidence")
    monkeypatch.setattr(
        "rag.application.decide_conversation_action",
        lambda *args, **kwargs: ConversationDecision(
            coverage="unrelated",
            next_action="fresh_retrieval",
            reason="the user changed topics",
            standalone_question="new standalone question",
            reusable_arxiv_ids=(),
            missing_aspects=("new topic",),
            retrieval_query="fresh topic query",
        ),
    )
    monkeypatch.setattr(
        "rag.application.hybrid_search",
        lambda query, **kwargs: [fresh],
    )
    monkeypatch.setattr(
        "rag.application.judge_retrieval",
        lambda *args, **kwargs: RetrievalDecision(
            sufficient=True,
            reason="fresh evidence is sufficient",
            missing_aspects=(),
            rewritten_query=None,
        ),
    )
    monkeypatch.setattr(
        "rag.application.answer_from_results",
        lambda question, results, **kwargs: "fresh answer [FRESH]",
    )

    result = run_rag(
        "Cross-encoder 有什么作用？",
        top_k=5,
        collection=object(),
        embedder=object(),
        reranker=FollowupReranker(),
        settings=ModelSettings("key", "https://example.test", "model"),
        conversation_history=HISTORY,
        active_evidence=(old,),
    )

    assert [paper.arxiv_id for paper in result.papers] == ["FRESH"]
    assert result.conversation_decision is not None
    assert result.conversation_decision.next_action == "fresh_retrieval"


def test_zero_results_return_a_deterministic_grounded_refusal() -> None:
    class EmptyCollection:
        def count(self) -> int:
            return 0

    result = run_rag(
        "没有相关资料的问题",
        top_k=5,
        collection=EmptyCollection(),  # type: ignore[arg-type]
        embedder=FakeEmbedder(),
        settings=ModelSettings("key", "https://example.test", "model"),
    )

    assert result.answer == SAFE_INSUFFICIENT_EVIDENCE_RESPONSE
    assert result.generation_error is None
    assert result.papers == ()
    assert result.trace[-1].stage == "answer_validation"
    assert result.trace[-1].details["reason"] == "no_evidence"


def test_unknown_citation_is_replaced_with_a_safe_refusal(
    monkeypatch: object,
) -> None:
    monkeypatch.setattr(
        "rag.application.answer_from_results",
        lambda *args, **kwargs: "错误结论 [9912.99999]。",
    )

    result = run_rag(
        "What is supported?",
        top_k=1,
        collection=FakeCollection(),
        embedder=FakeEmbedder(),
        settings=ModelSettings("key", "https://example.test", "model"),
    )

    assert result.answer == SAFE_UNVERIFIED_ANSWER_RESPONSE
    assert result.papers == ()
    assert result.generation_error is None
    assert result.trace[-1].status == "failed"
    assert result.trace[-1].details["unknown_citation_ids"] == ["9912.99999"]
