"""Combine dense and BM25 retrieval, then rerank the fused candidates."""

from dataclasses import replace
from pathlib import Path

from chromadb.api.models.Collection import Collection

from rag.execution_trace import TraceRecorder, start_timer
from rag.keyword_search import DEFAULT_DATABASE_PATH, search_keyword_papers
from rag.reranker import Reranker
from rag.search import QueryEmbedder, SearchResult, search_collection


DEFAULT_DENSE_CANDIDATES = 30
DEFAULT_KEYWORD_CANDIDATES = 30
DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    dense_results: list[SearchResult],
    keyword_results: list[SearchResult],
    *,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[SearchResult]:
    """Merge incomparable retrieval scores using reciprocal ranks."""
    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than zero")

    candidates: dict[str, SearchResult] = {}
    fusion_scores: dict[str, float] = {}

    for results in (dense_results, keyword_results):
        for rank, result in enumerate(results, start=1):
            fusion_scores[result.arxiv_id] = fusion_scores.get(
                result.arxiv_id, 0.0
            ) + 1.0 / (rrf_k + rank)
            existing = candidates.get(result.arxiv_id)
            if existing is None:
                candidates[result.arxiv_id] = result
                continue
            candidates[result.arxiv_id] = replace(
                existing,
                similarity=(
                    existing.similarity
                    if existing.similarity is not None
                    else result.similarity
                ),
                keyword_score=(
                    existing.keyword_score
                    if existing.keyword_score is not None
                    else result.keyword_score
                ),
            )

    fused = [
        replace(result, fusion_score=fusion_scores[paper_id])
        for paper_id, result in candidates.items()
    ]
    return sorted(
        fused,
        key=lambda result: (-float(result.fusion_score or 0.0), result.arxiv_id),
    )


def hybrid_search(
    query: str,
    *,
    top_k: int,
    collection: Collection,
    embedder: QueryEmbedder,
    reranker: Reranker,
    database_path: Path = DEFAULT_DATABASE_PATH,
    dense_top_k: int = DEFAULT_DENSE_CANDIDATES,
    keyword_top_k: int = DEFAULT_KEYWORD_CANDIDATES,
    trace: TraceRecorder | None = None,
    retrieval_round: int = 1,
) -> list[SearchResult]:
    """Retrieve with dense and BM25 search, then Cross-encode the candidates."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if dense_top_k <= 0 or keyword_top_k <= 0:
        raise ValueError("candidate counts must be greater than zero")

    started_at = start_timer()
    dense_results = search_collection(
        query,
        top_k=dense_top_k,
        collection=collection,
        embedder=embedder,
    )
    if trace is not None:
        trace.record(
            stage="dense_retrieval",
            label="E5 向量召回",
            started_at=started_at,
            details={
                "round": retrieval_round,
                "query": query,
                "result_count": len(dense_results),
                "top_arxiv_ids": [
                    result.arxiv_id for result in dense_results[:5]
                ],
            },
        )

    started_at = start_timer()
    keyword_results = search_keyword_papers(
        query,
        top_k=keyword_top_k,
        database_path=database_path,
    )
    if trace is not None:
        trace.record(
            stage="keyword_retrieval",
            label="BM25 关键词召回",
            started_at=started_at,
            details={
                "round": retrieval_round,
                "query": query,
                "result_count": len(keyword_results),
                "top_arxiv_ids": [
                    result.arxiv_id for result in keyword_results[:5]
                ],
            },
        )

    started_at = start_timer()
    fused = reciprocal_rank_fusion(dense_results, keyword_results)
    if trace is not None:
        trace.record(
            stage="rank_fusion",
            label="RRF 候选融合与去重",
            started_at=started_at,
            details={
                "round": retrieval_round,
                "dense_count": len(dense_results),
                "keyword_count": len(keyword_results),
                "unique_candidate_count": len(fused),
            },
        )
    if not fused:
        return []

    started_at = start_timer()
    scores = reranker.score(query, [result.document for result in fused])
    if len(scores) != len(fused):
        raise ValueError("reranker returned an unexpected number of scores")

    reranked = [
        replace(result, rerank_score=float(score))
        for result, score in zip(fused, scores)
    ]
    reranked.sort(
        key=lambda result: (-float(result.rerank_score or 0.0), result.arxiv_id)
    )
    final_results = reranked[: min(top_k, len(reranked))]
    if trace is not None:
        trace.record(
            stage="candidate_rerank",
            label="Cross-encoder 候选重排",
            started_at=started_at,
            details={
                "round": retrieval_round,
                "candidate_count": len(fused),
                "result_count": len(final_results),
                "top_arxiv_ids": [
                    result.arxiv_id for result in final_results
                ],
            },
        )
    return final_results
