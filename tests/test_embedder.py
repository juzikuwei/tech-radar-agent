from rag import embedder as embedder_module


class FakeSentenceTransformer:
    """Record model-loading arguments without loading real weights."""

    def __init__(self, model_name: str, **options: object) -> None:
        self.model_name = model_name
        self.options = options


def test_embedder_loads_model_from_local_cache_only(monkeypatch: object) -> None:
    monkeypatch.setattr(
        embedder_module,
        "SentenceTransformer",
        FakeSentenceTransformer,
    )

    embedder = embedder_module.E5Embedder("test-model")
    model = embedder._get_model()

    assert model.model_name == "test-model"
    assert model.options == {"local_files_only": True}
