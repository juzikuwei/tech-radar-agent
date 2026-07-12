"""Framework-independent similarity ranking helpers."""

import numpy as np


def build_embedding_text(title: str, abstract: str) -> str:
    """Build the exact paper text represented by one embedding vector."""
    return f"{title}\n{abstract}"


def rank_normalized_embeddings(
    query_embedding: np.ndarray,
    document_embeddings: np.ndarray,
    *,
    top_k: int,
) -> list[tuple[int, float]]:
    """Rank normalized document vectors by cosine similarity."""
    if query_embedding.ndim != 1:
        raise ValueError("query_embedding must be one-dimensional")
    if document_embeddings.ndim != 2:
        raise ValueError("document_embeddings must be two-dimensional")
    if document_embeddings.shape[1] != query_embedding.shape[0]:
        raise ValueError("query and document embedding dimensions must match")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if document_embeddings.shape[0] == 0:
        return []

    scores = document_embeddings @ query_embedding
    result_count = min(top_k, document_embeddings.shape[0])
    ranked_indices = np.argsort(scores)[::-1][:result_count]
    return [(int(index), float(scores[index])) for index in ranked_indices]
