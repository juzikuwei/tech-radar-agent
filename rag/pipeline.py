"""Run the first end-to-end retrieval-augmented answer pipeline."""

import argparse
from pathlib import Path
import sys

from config.environment import load_repository_env

# Hugging Face reads endpoint settings while its modules are imported.
load_repository_env()

from config.model_settings import ModelSettings
from rag.application import run_rag
from rag.embedder import DEFAULT_MODEL_NAME, E5Embedder
from rag.keyword_search import DEFAULT_DATABASE_PATH
from rag.llm_client import RetryNotice
from rag.reranker import DEFAULT_RERANKER_MODEL, CrossEncoderReranker
from rag.vector_store import (
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION_NAME,
    get_persistent_collection,
)

def _print_retry(notice: RetryNotice) -> None:
    print(
        f"Reconnecting in {notice.wait_seconds:g}s "
        f"(attempt {notice.next_attempt}/{notice.max_attempts}, "
        f"reason: {notice.reason})"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments for one grounded question."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chroma-path", type=Path, default=DEFAULT_CHROMA_PATH)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    return parser


def main() -> int:
    """Retrieve papers, call DeepSeek, and print its grounded answer."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    args = build_parser().parse_args()
    settings = ModelSettings.from_env()
    collection = get_persistent_collection(args.chroma_path, args.collection)
    result = run_rag(
        args.question,
        top_k=args.top_k,
        collection=collection,
        embedder=E5Embedder(args.embedding_model),
        settings=settings,
        reranker=CrossEncoderReranker(args.reranker_model),
        database_path=args.database,
        on_retry=_print_retry,
    )
    if not result.papers:
        print("No papers were retrieved; the question cannot be answered.")
        return 1

    if result.retrieval_decision_error is not None:
        print(
            "Retrieval judge failed; using the first retrieval: "
            f"{result.retrieval_decision_error}"
        )
    elif result.retrieval_decision is not None:
        decision = result.retrieval_decision
        print(
            "Retrieval decision: "
            f"sufficient={decision.sufficient} reason={decision.reason}"
        )
        if decision.rewritten_query is not None:
            print(f"Rewritten query: {decision.rewritten_query}")
    print(f"Retrieval attempts: {result.retrieval_attempts}")
    print()

    print("Retrieved papers:")
    for rank, paper in enumerate(result.papers, start=1):
        dense_score = (
            f" dense={paper.similarity:.4f}"
            if paper.similarity is not None
            else ""
        )
        keyword_score = (
            f" bm25={paper.keyword_score:.4f}"
            if paper.keyword_score is not None
            else ""
        )
        rerank_score = (
            f" rerank={paper.rerank_score:.4f}"
            if paper.rerank_score is not None
            else ""
        )
        print(
            f"{rank}. {paper.arxiv_id}{dense_score}{keyword_score}"
            f"{rerank_score} {paper.title}"
        )

    print()
    if result.generation_error is not None:
        print(f"Answer generation failed: {result.generation_error}")
        return 1
    print(result.answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
