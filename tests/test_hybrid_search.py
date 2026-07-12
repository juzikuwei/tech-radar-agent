import numpy as np

from rag.execution_trace import TraceRecorder
from rag.hybrid_search import hybrid_search, reciprocal_rank_fusion
from rag.search import SearchResult


def make_result(
    arxiv_id: str,
    *,
    similarity: float | None = None,
    keyword_score: float | None = None,
) -> SearchResult:
    return SearchResult(
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        document=f"Document {arxiv_id}",
        entry_url=f"https://arxiv.org/abs/{arxiv_id}",
        primary_category="cs.AI",
        similarity=similarity,
        keyword_score=keyword_score,
    )


def test_rrf_rewards_candidates_found_by_both_retrievers() -> None:
    dense = [make_result("A", similarity=0.9), make_result("B", similarity=0.8)]
    keyword = [
        make_result("B", keyword_score=10.0),
        make_result("C", keyword_score=8.0),
    ]

    fused = reciprocal_rank_fusion(dense, keyword)

    assert [result.arxiv_id for result in fused] == ["B", "A", "C"]
    assert fused[0].similarity == 0.8
    assert fused[0].keyword_score == 10.0


class FakeReranker:
    model_name = "fake-cross-encoder"

    def score(self, query: str, documents: list[str]) -> np.ndarray:
        assert query == "specific question"
        return np.asarray([0.1 if "A" in document else 0.9 for document in documents])


def test_hybrid_search_uses_reranker_for_final_order(monkeypatch: object) -> None:
    monkeypatch.setattr(
        "rag.hybrid_search.search_collection",
        lambda *args, **kwargs: [make_result("A", similarity=0.9)],
    )
    monkeypatch.setattr(
        "rag.hybrid_search.search_keyword_papers",
        lambda *args, **kwargs: [make_result("B", keyword_score=12.0)],
    )

    trace = TraceRecorder()
    results = hybrid_search(
        "specific question",
        top_k=2,
        collection=object(),
        embedder=object(),
        reranker=FakeReranker(),
        trace=trace,
    )

    assert [result.arxiv_id for result in results] == ["B", "A"]
    assert results[0].rerank_score == 0.9
    assert [event.stage for event in trace.events] == [
        "dense_retrieval",
        "keyword_retrieval",
        "rank_fusion",
        "candidate_rerank",
    ]
