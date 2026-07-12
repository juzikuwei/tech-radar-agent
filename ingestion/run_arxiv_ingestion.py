"""Command-line entry point for one complete arXiv snapshot."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from config.arxiv_queries import (
    DEFAULT_QUERY_NAME,
    get_arxiv_query,
    load_arxiv_queries,
)
from ingestion.arxiv_fetcher import ArxivFetchError, fetch_papers
from ingestion.normalizer import to_paper_record
from ingestion.snapshot import write_jsonl_snapshot


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments for a small, manually triggered batch."""
    parser = argparse.ArgumentParser(description=__doc__)
    query_group = parser.add_mutually_exclusive_group()
    query_group.add_argument("--query")
    query_group.add_argument("--query-name", default=DEFAULT_QUERY_NAME)
    parser.add_argument("--list-queries", action="store_true")
    parser.add_argument("--max-results", type=int, default=3)
    parser.add_argument("--page-size", type=int, default=25)
    parser.add_argument("--delay-seconds", type=float, default=3.0)
    parser.add_argument("--num-retries", type=int, default=3)
    parser.add_argument("--output", type=Path)
    return parser


def default_output_path(fetched_at: datetime) -> Path:
    """Create a timestamped path so completed snapshots remain reproducible."""
    timestamp = fetched_at.strftime("%Y%m%dT%H%M%SZ")
    return Path("data/raw") / f"arxiv_{timestamp}.jsonl"


def main() -> int:
    """Fetch, normalize, and atomically publish one arXiv batch."""
    args = build_parser().parse_args()
    if args.list_queries:
        for name, query in load_arxiv_queries().items():
            print(f"{name}: {query}")
        return 0

    try:
        query = args.query or get_arxiv_query(args.query_name)
    except (OSError, ValueError) as exc:
        print(f"Ingestion failed: {exc}")
        return 1

    fetched_at = datetime.now(timezone.utc)
    output_path = args.output or default_output_path(fetched_at)

    try:
        papers = fetch_papers(
            query,
            args.max_results,
            page_size=args.page_size,
            delay_seconds=args.delay_seconds,
            num_retries=args.num_retries,
        )
    except (ValueError, ArxivFetchError) as exc:
        print(f"Ingestion failed: {exc}")
        return 1

    records = [
        to_paper_record(paper, query=query, fetched_at=fetched_at)
        for paper in papers
    ]
    count = write_jsonl_snapshot(records, output_path)
    print(f"Saved {count} papers to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
