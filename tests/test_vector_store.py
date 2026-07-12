from collections.abc import Sequence
from uuid import uuid4

import chromadb
import numpy as np
import pytest

from rag.indexer import sync_papers_to_collection
from rag.search import search_collection


class FakeEmbedder:
    def __init__(self, model_name: str = "fake-e5-v1") -> None:
        self.model_name = model_name
        self.encoded_passages: list[str] = []
        self.batch_sizes: list[int] = []

    def encode_passages(self, passages: Sequence[str]) -> np.ndarray:
        self.batch_sizes.append(len(passages))
        vectors = []
        for passage in passages:
            self.encoded_passages.append(passage)
            if "FAIL" in passage:
                raise RuntimeError("simulated embedding failure")
            if "memory" in passage.lower():
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return np.asarray(vectors, dtype=float)

    def encode_query(self, query: str) -> np.ndarray:
        if "memory" in query.lower():
            return np.asarray([1.0, 0.0], dtype=float)
        return np.asarray([0.0, 1.0], dtype=float)


def make_collection():
    client = chromadb.EphemeralClient()
    return client.get_or_create_collection(
        f"test_papers_{uuid4().hex}",
        configuration={"hnsw": {"space": "cosine"}},
    )


def make_paper(
    *,
    arxiv_id: str,
    title: str,
    content_hash: str,
    version: int = 1,
) -> dict[str, str]:
    return {
        "arxiv_id": arxiv_id,
        "versioned_arxiv_id": f"{arxiv_id}v{version}",
        "title": title,
        "abstract": f"Abstract for {title}",
        "content_hash": content_hash,
        "primary_category": "cs.AI",
        "published_at": "2026-07-01T00:00:00+00:00",
        "updated_at": f"2026-07-{version:02d}T00:00:00+00:00",
        "entry_url": f"https://arxiv.org/abs/{arxiv_id}v{version}",
    }


def test_sync_adds_then_skips_unchanged_papers() -> None:
    collection = make_collection()
    embedder = FakeEmbedder()
    papers = [
        make_paper(arxiv_id="2607.00001", title="Memory agent", content_hash="a"),
        make_paper(arxiv_id="2607.00002", title="Energy agent", content_hash="b"),
    ]

    first = sync_papers_to_collection(papers, collection, embedder)
    encoded_after_first = len(embedder.encoded_passages)
    second = sync_papers_to_collection(papers, collection, embedder)

    assert first.added == 2
    assert second.unchanged == 2
    assert len(embedder.encoded_passages) == encoded_after_first
    assert collection.count() == 2


def test_sync_embeds_changed_papers_in_configured_batches() -> None:
    collection = make_collection()
    embedder = FakeEmbedder()
    papers = [
        make_paper(
            arxiv_id=f"2607.{index:05d}",
            title=f"Paper {index}",
            content_hash=str(index),
        )
        for index in range(5)
    ]

    stats = sync_papers_to_collection(
        papers,
        collection,
        embedder,
        batch_size=2,
    )

    assert stats.added == 5
    assert embedder.batch_sizes == [2, 2, 1]
    assert collection.count() == 5


def test_sync_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        sync_papers_to_collection([], make_collection(), FakeEmbedder(), batch_size=0)


def test_sync_replaces_vector_when_content_hash_changes() -> None:
    collection = make_collection()
    embedder = FakeEmbedder()
    original = make_paper(
        arxiv_id="2607.00001", title="Memory agent", content_hash="old"
    )
    changed = make_paper(
        arxiv_id="2607.00001",
        title="Energy agent",
        content_hash="new",
        version=2,
    )

    sync_papers_to_collection([original], collection, embedder)
    stats = sync_papers_to_collection([changed], collection, embedder)
    stored = collection.get(ids=["2607.00001"], include=["metadatas"])

    assert stats.updated == 1
    assert collection.count() == 1
    assert stored["metadatas"][0]["content_hash"] == "new"


def test_sync_updates_metadata_without_reembedding() -> None:
    collection = make_collection()
    embedder = FakeEmbedder()
    original = make_paper(
        arxiv_id="2607.00001", title="Memory agent", content_hash="same"
    )
    revised_metadata = make_paper(
        arxiv_id="2607.00001",
        title="Memory agent",
        content_hash="same",
        version=2,
    )
    sync_papers_to_collection([original], collection, embedder)
    encoded_after_first = len(embedder.encoded_passages)

    stats = sync_papers_to_collection([revised_metadata], collection, embedder)

    assert stats.metadata_updated == 1
    assert len(embedder.encoded_passages) == encoded_after_first


def test_sync_rebuilds_vector_when_model_changes() -> None:
    collection = make_collection()
    paper = make_paper(
        arxiv_id="2607.00001", title="Memory agent", content_hash="same"
    )
    sync_papers_to_collection([paper], collection, FakeEmbedder("model-v1"))

    stats = sync_papers_to_collection(
        [paper], collection, FakeEmbedder("model-v2")
    )

    assert stats.updated == 1


def test_sync_keeps_successful_papers_when_one_fails() -> None:
    collection = make_collection()
    papers = [
        make_paper(arxiv_id="2607.00001", title="Memory agent", content_hash="a"),
        make_paper(arxiv_id="2607.00002", title="FAIL paper", content_hash="b"),
    ]

    stats = sync_papers_to_collection(papers, collection, FakeEmbedder())

    assert stats.added == 1
    assert stats.failed == 1
    assert stats.failure_messages[0].startswith("2607.00002:")
    assert collection.count() == 1


def test_search_returns_nearest_paper_first() -> None:
    collection = make_collection()
    embedder = FakeEmbedder()
    papers = [
        make_paper(arxiv_id="2607.00001", title="Memory agent", content_hash="a"),
        make_paper(arxiv_id="2607.00002", title="Energy agent", content_hash="b"),
    ]
    sync_papers_to_collection(papers, collection, embedder)

    results = search_collection(
        "memory research",
        top_k=2,
        collection=collection,
        embedder=embedder,
    )

    assert results[0].arxiv_id == "2607.00001"
    assert results[0].similarity == 1.0
