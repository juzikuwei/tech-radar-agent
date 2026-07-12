import numpy as np

from rag.reranker import CrossEncoderReranker


class FakeCrossEncoder:
    def predict(self, pairs: list[tuple[str, str]], **options: object) -> np.ndarray:
        assert pairs == [("query", "first"), ("query", "second")]
        assert options["batch_size"] == 2
        return np.asarray([0.2, 0.8])


def test_cross_encoder_scores_query_document_pairs() -> None:
    reranker = CrossEncoderReranker("fake-model", batch_size=2)
    reranker._model = FakeCrossEncoder()  # type: ignore[assignment]

    scores = reranker.score("query", ["first", "second"])

    assert scores.tolist() == [0.2, 0.8]
