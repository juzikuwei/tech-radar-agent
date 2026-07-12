"""Search the persistent ChromaDB paper index."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from chromadb.api.models.Collection import Collection

from rag.embedder import DEFAULT_MODEL_NAME, E5Embedder
from rag.vector_store import (
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION_NAME,
    get_persistent_collection,
)


class QueryEmbedder(Protocol):
    """Minimal query embedding contract needed by persistent search."""

    def encode_query(self, query: str) -> np.ndarray:
        """Encode one user query."""


@dataclass(frozen=True)
class SearchResult:
    """One ranked paper returned from ChromaDB."""

    arxiv_id: str
    title: str
    document: str
    entry_url: str
    primary_category: str
    similarity: float | None
    keyword_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None


def search_collection(
    query: str,
    *,
    top_k: int,
    collection: Collection,
    embedder: QueryEmbedder,
) -> list[SearchResult]:
    """Return the nearest papers using a normalized E5 query vector."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    collection_size = collection.count()
    if collection_size == 0:
        return []

    query_embedding = embedder.encode_query(query)
    raw_results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(top_k, collection_size),
        include=["documents", "metadatas", "distances"],
    )

    ids = raw_results["ids"][0]
    documents = (raw_results["documents"] or [[]])[0]
    metadatas = (raw_results["metadatas"] or [[]])[0]
    distances = (raw_results["distances"] or [[]])[0]

    return [
        SearchResult(
            arxiv_id=paper_id,
            title=str(metadata["title"]),
            document=document,
            entry_url=str(metadata["entry_url"]),
            primary_category=str(metadata["primary_category"]),
            similarity=1.0 - float(distance),
        )
        for paper_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        )
    ]


def build_parser() -> argparse.ArgumentParser:
    """Build arguments for persistent semantic search."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--chroma-path", type=Path, default=DEFAULT_CHROMA_PATH)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    return parser


def main() -> int:
    """Query the persistent paper collection and print ranked results."""
    args = build_parser().parse_args()
    collection = get_persistent_collection(args.chroma_path, args.collection)
    embedder = E5Embedder(args.model)
    results = search_collection(
        args.query,
        top_k=args.top_k,
        collection=collection,
        embedder=embedder,
    )

    if not results:
        print("No indexed papers found")
        return 1

    print(f"Query: {args.query}")
    for rank, result in enumerate(results, start=1):
        print()
        print(
            f"{rank}. similarity={result.similarity:.4f} "
            f"arxiv_id={result.arxiv_id}"
        )
        print(result.title)
        print(result.entry_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
