"""Small readable demo of the current arXiv ingestion pipeline."""

from datetime import datetime, timezone
import json

from ingestion.arxiv_fetcher import fetch_papers
from ingestion.normalizer import to_paper_record


QUERY = "cat:cs.AI AND all:agent"


papers = fetch_papers(QUERY, max_results=3)

for index, paper in enumerate(papers, start=1):
    record = to_paper_record(
        paper,
        query=QUERY,
        fetched_at=datetime.now(timezone.utc),
    )
    print(f"Paper {index}")
    print(json.dumps(record, ensure_ascii=False, indent=2))
