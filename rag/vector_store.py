"""Persistent ChromaDB collection configuration."""

from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings


DEFAULT_CHROMA_PATH = Path("data/chroma")
DEFAULT_COLLECTION_NAME = "tech_radar_papers"


def get_persistent_collection(
    path: Path = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Collection:
    """Open the persistent cosine-similarity paper collection."""
    path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(path),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=collection_name,
        configuration={"hnsw": {"space": "cosine"}},
    )
