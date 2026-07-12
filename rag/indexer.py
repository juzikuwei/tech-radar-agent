"""Incrementally synchronize SQLite papers into ChromaDB."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from chromadb.api.models.Collection import Collection

from config.environment import load_repository_env

# Load shared runtime settings before importing the embedding boundary.
load_repository_env()

from ingestion.repository import load_papers_for_embedding
from rag.embedder import DEFAULT_MODEL_NAME, E5Embedder
from rag.similarity import build_embedding_text
from rag.vector_store import (
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION_NAME,
    get_persistent_collection,
)


DEFAULT_DATABASE_PATH = Path("data/tech_radar.db")
DEFAULT_BATCH_SIZE = 32


class PassageEmbedder(Protocol):
    """Minimal embedding contract needed by the index synchronizer."""

    model_name: str

    def encode_passages(self, passages: Sequence[str]) -> np.ndarray:
        """Encode one or more passage strings."""


@dataclass(frozen=True)
class SyncStats:
    """Observable result of one idempotent ChromaDB synchronization."""

    added: int = 0
    updated: int = 0
    metadata_updated: int = 0
    unchanged: int = 0
    failed: int = 0
    failure_messages: tuple[str, ...] = ()


def sync_papers_to_collection(
    papers: Sequence[dict[str, str]],
    collection: Collection,
    embedder: PassageEmbedder,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> SyncStats:
    """Synchronize papers with batched embedding and safe per-paper fallback."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if not papers:
        return SyncStats()

    paper_ids = [paper["arxiv_id"] for paper in papers]
    stored = collection.get(ids=paper_ids, include=["metadatas"])
    stored_metadata = {
        paper_id: metadata or {}
        for paper_id, metadata in zip(stored["ids"], stored["metadatas"] or [])
    }

    added = 0
    updated = 0
    metadata_updated = 0
    unchanged = 0
    failed = 0
    failure_messages: list[str] = []
    vector_jobs: list[tuple[dict[str, str], dict[str, str], str, bool]] = []

    for paper in papers:
        paper_id = paper["arxiv_id"]
        existing = stored_metadata.get(paper_id)
        metadata = _build_metadata(paper, embedder.model_name)
        document = build_embedding_text(paper["title"], paper["abstract"])

        vector_changed = (
            existing is None
            or existing.get("content_hash") != paper["content_hash"]
            or existing.get("embedding_model") != embedder.model_name
        )

        if vector_changed:
            vector_jobs.append((paper, metadata, document, existing is None))
            continue

        try:
            if _metadata_changed(existing, metadata):
                collection.update(ids=[paper_id], metadatas=[metadata])
                metadata_updated += 1
            else:
                unchanged += 1
        except Exception as exc:
            failed += 1
            failure_messages.append(f"{paper_id}: {exc}")

    for start in range(0, len(vector_jobs), batch_size):
        batch = vector_jobs[start : start + batch_size]
        documents = [job[2] for job in batch]
        try:
            embeddings = embedder.encode_passages(documents)
        except Exception:
            embeddings = None

        if embeddings is None:
            for paper, metadata, document, is_new in batch:
                try:
                    embedding = embedder.encode_passages([document])[0]
                    collection.upsert(
                        ids=[paper["arxiv_id"]],
                        embeddings=[embedding.tolist()],
                        documents=[document],
                        metadatas=[metadata],
                    )
                    if is_new:
                        added += 1
                    else:
                        updated += 1
                except Exception as exc:
                    failed += 1
                    failure_messages.append(f"{paper['arxiv_id']}: {exc}")
            continue

        try:
            collection.upsert(
                ids=[job[0]["arxiv_id"] for job in batch],
                embeddings=[embedding.tolist() for embedding in embeddings],
                documents=documents,
                metadatas=[job[1] for job in batch],
            )
            added += sum(1 for job in batch if job[3])
            updated += sum(1 for job in batch if not job[3])
        except Exception:
            for (paper, metadata, document, is_new), embedding in zip(
                batch, embeddings
            ):
                try:
                    collection.upsert(
                        ids=[paper["arxiv_id"]],
                        embeddings=[embedding.tolist()],
                        documents=[document],
                        metadatas=[metadata],
                    )
                    if is_new:
                        added += 1
                    else:
                        updated += 1
                except Exception as exc:
                    failed += 1
                    failure_messages.append(f"{paper['arxiv_id']}: {exc}")

    return SyncStats(
        added=added,
        updated=updated,
        metadata_updated=metadata_updated,
        unchanged=unchanged,
        failed=failed,
        failure_messages=tuple(failure_messages),
    )


def _build_metadata(paper: dict[str, str], model_name: str) -> dict[str, str]:
    return {
        "content_hash": paper["content_hash"],
        "embedding_model": model_name,
        "versioned_arxiv_id": paper["versioned_arxiv_id"],
        "primary_category": paper["primary_category"],
        "published_at": paper["published_at"],
        "updated_at": paper["updated_at"],
        "entry_url": paper["entry_url"],
        "title": paper["title"],
    }


def _metadata_changed(
    existing: dict[str, object],
    current: dict[str, str],
) -> bool:
    return any(existing.get(key) != value for key, value in current.items())


def build_parser() -> argparse.ArgumentParser:
    """Build arguments for the persistent index synchronization command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--chroma-path", type=Path, default=DEFAULT_CHROMA_PATH)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser


def main() -> int:
    """Synchronize SQLite current state into a persistent Chroma collection."""
    args = build_parser().parse_args()
    if not args.database.exists():
        print(f"Database not found: {args.database}")
        return 1

    papers = load_papers_for_embedding(args.database)
    if not papers:
        print("No papers found in the database")
        return 1

    collection = get_persistent_collection(args.chroma_path, args.collection)
    embedder = E5Embedder(args.model)
    stats = sync_papers_to_collection(
        papers,
        collection,
        embedder,
        batch_size=args.batch_size,
    )

    print(f"Added: {stats.added}")
    print(f"Updated: {stats.updated}")
    print(f"Metadata updated: {stats.metadata_updated}")
    print(f"Unchanged: {stats.unchanged}")
    print(f"Failed: {stats.failed}")
    for failure_message in stats.failure_messages:
        print(f"Failure: {failure_message}")
    print(f"Collection size: {collection.count()}")
    return 1 if stats.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
