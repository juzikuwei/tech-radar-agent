"""Cross-encoder boundary for precise ranking of a small candidate set."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from sentence_transformers import CrossEncoder


DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
DEFAULT_RERANK_BATCH_SIZE = 16


class Reranker(Protocol):
    """Minimal scoring contract needed by hybrid retrieval."""

    model_name: str

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray:
        """Return one relevance score for each query-document pair."""


class CrossEncoderReranker:
    """Lazily load a local multilingual Cross-encoder for candidate ranking."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: CrossEncoder | None = None

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray:
        """Score candidate documents jointly with the user query."""
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("query must not be empty")
        if not documents:
            return np.asarray([], dtype=float)
        if any(not document.strip() for document in documents):
            raise ValueError("documents must not contain empty text")

        scores = self._get_model().predict(
            [(clean_query, document) for document in documents],
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(scores, dtype=float).reshape(-1)

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(
                self.model_name,
                local_files_only=True,
            )
        return self._model
