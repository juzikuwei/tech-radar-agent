"""Network boundary for fetching papers from arXiv."""

import arxiv


class ArxivFetchError(RuntimeError):
    """Raised when arXiv cannot complete a paper query."""


def fetch_papers(
    query: str,
    max_results: int = 3,
    *,
    page_size: int = 25,
    delay_seconds: float = 3.0,
    num_retries: int = 3,
) -> list[arxiv.Result]:
    """Fetch a complete batch of papers or raise an arXiv boundary error."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if max_results <= 0:
        raise ValueError("max_results must be greater than zero")
    if delay_seconds < 0:
        raise ValueError("delay_seconds must not be negative")
    if page_size <= 0:
        raise ValueError("page_size must be greater than zero")
    if num_retries < 0:
        raise ValueError("num_retries must not be negative")

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    client = arxiv.Client(
        page_size=min(max_results, page_size),
        delay_seconds=delay_seconds,
        num_retries=num_retries,
    )

    try:
        return list(client.results(search))
    except Exception as exc:
        raise ArxivFetchError(
            f"arXiv query failed: {query!r}; "
            f"cause={type(exc).__name__}: {exc}"
        ) from exc
