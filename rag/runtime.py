"""Shared long-lived resources used by API and MCP adapters."""

from dataclasses import dataclass
from pathlib import Path

from chromadb.api.models.Collection import Collection

from config.environment import load_repository_env

# Hugging Face reads endpoint settings while its modules are imported.
load_repository_env()

from config.model_settings import ModelSettings
from rag.embedder import E5Embedder
from rag.keyword_search import DEFAULT_DATABASE_PATH, ensure_keyword_index
from rag.reranker import CrossEncoderReranker, Reranker
from rag.vector_store import get_persistent_collection
from rag.web_search import WebSearchClient, load_web_search_client


@dataclass(frozen=True)
class RagRuntime:
    """Persistent stores and models shared within one server process."""

    collection: Collection
    embedder: E5Embedder
    reranker: Reranker
    settings: ModelSettings
    database_path: Path = DEFAULT_DATABASE_PATH
    web_search_client: WebSearchClient | None = None


def load_rag_runtime() -> RagRuntime:
    """Load the production stores and local models once."""
    ensure_keyword_index(DEFAULT_DATABASE_PATH)
    return RagRuntime(
        collection=get_persistent_collection(),
        embedder=E5Embedder(),
        reranker=CrossEncoderReranker(),
        settings=ModelSettings.from_env(),
        web_search_client=load_web_search_client(),
    )
