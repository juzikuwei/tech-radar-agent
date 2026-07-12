"""Multilingual E5 embedding boundary."""

from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-small"


class E5Embedder:
    """Generate normalized query and passage vectors using E5 prefixes."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    def encode_query(self, query: str) -> np.ndarray:
        """Encode one user query as a normalized vector."""
        if not query.strip():
            raise ValueError("query must not be empty")
        vectors = self._get_model().encode(
            [f"query: {query}"],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors[0]

    def encode_passages(self, passages: Sequence[str]) -> np.ndarray:
        """Encode normalized paper texts as normalized vectors."""
        if not passages:
            raise ValueError("passages must not be empty")
        if any(not passage.strip() for passage in passages):
            raise ValueError("passages must not contain empty text")
        return self._get_model().encode(
            [f"passage: {passage}" for passage in passages],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(passages) > 20,
        )

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(
                self.model_name,
                local_files_only=True,
            )
        return self._model
