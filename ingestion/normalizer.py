"""Convert arXiv library objects into stable internal paper records."""

from datetime import datetime
import hashlib
import re

import arxiv


VERSION_SUFFIX_PATTERN = re.compile(r"v\d+$")


def normalize_text(text: str) -> str:
    """Collapse line breaks and repeated whitespace without changing meaning."""
    return " ".join(text.split())


def get_versionless_arxiv_id(paper: arxiv.Result) -> str:
    """Return a stable identity shared by all revisions of one paper."""
    return VERSION_SUFFIX_PATTERN.sub("", paper.get_short_id())


def build_content_hash(title: str, abstract: str) -> str:
    """Fingerprint the exact normalized content used by future embeddings."""
    embedding_content = f"{title}\n{abstract}"
    return hashlib.sha256(embedding_content.encode("utf-8")).hexdigest()


def to_paper_record(
    paper: arxiv.Result,
    *,
    query: str,
    fetched_at: datetime,
) -> dict[str, object]:
    """Convert one arXiv result into a JSON-serializable paper record."""
    title = normalize_text(paper.title)
    abstract = normalize_text(paper.summary)

    return {
        "arxiv_id": get_versionless_arxiv_id(paper),
        "versioned_arxiv_id": paper.get_short_id(),
        "raw_title": paper.title,
        "raw_abstract": paper.summary,
        "title": title,
        "abstract": abstract,
        "content_hash": build_content_hash(title, abstract),
        "authors": [author.name for author in paper.authors],
        "categories": list(paper.categories),
        "primary_category": paper.primary_category,
        "published_at": paper.published.isoformat(),
        "updated_at": paper.updated.isoformat(),
        "entry_url": paper.entry_id,
        "pdf_url": paper.pdf_url,
        "query": query,
        "fetched_at": fetched_at.isoformat(),
    }
