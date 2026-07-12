import numpy as np
import pytest

from rag.similarity import build_embedding_text, rank_normalized_embeddings


def test_build_embedding_text_uses_title_and_abstract() -> None:
    assert build_embedding_text("A title", "An abstract") == "A title\nAn abstract"


def test_rank_normalized_embeddings_returns_highest_score_first() -> None:
    query = np.array([1.0, 0.0])
    documents = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.8, 0.6],
        ]
    )

    ranked = rank_normalized_embeddings(query, documents, top_k=2)

    assert ranked[0] == (1, pytest.approx(1.0))
    assert ranked[1] == (2, pytest.approx(0.8))


def test_rank_normalized_embeddings_rejects_dimension_mismatch() -> None:
    query = np.array([1.0, 0.0])
    documents = np.array([[1.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="dimensions must match"):
        rank_normalized_embeddings(query, documents, top_k=1)
