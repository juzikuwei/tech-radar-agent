"""Run one Chinese-to-English semantic search against local papers."""

import argparse
from pathlib import Path

from ingestion.repository import load_papers_for_embedding
from rag.embedder import DEFAULT_MODEL_NAME, E5Embedder
from rag.similarity import build_embedding_text, rank_normalized_embeddings


DEFAULT_DATABASE_PATH = Path("data/tech_radar.db")
DEFAULT_QUERY = "Agent 的长期记忆有哪些研究？"


def build_parser() -> argparse.ArgumentParser:
    """Build arguments for the in-memory similarity experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    return parser


def main() -> int:
    """Embed the current SQLite papers and print a ranked result list."""
    args = build_parser().parse_args()

    if not args.database.exists():
        print(f"Database not found: {args.database}")
        return 1
    if args.top_k <= 0:
        print("top-k must be greater than zero")
        return 1

    papers = load_papers_for_embedding(args.database)
    if not papers:
        print("No papers found in the database")
        return 1

    passages = [
        build_embedding_text(paper["title"], paper["abstract"])
        for paper in papers
    ]
    embedder = E5Embedder(args.model)
    document_embeddings = embedder.encode_passages(passages)
    query_embedding = embedder.encode_query(args.query)
    ranked = rank_normalized_embeddings(
        query_embedding,
        document_embeddings,
        top_k=args.top_k,
    )

    print(f"Query: {args.query}")
    print(f"Model: {args.model}")
    for rank, (paper_index, score) in enumerate(ranked, start=1):
        paper = papers[paper_index]
        print()
        print(f"{rank}. score={score:.4f} arxiv_id={paper['arxiv_id']}")
        print(paper["title"])
        print(paper["entry_url"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
