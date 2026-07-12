"""Load named arXiv query definitions from repository configuration."""

import json
from pathlib import Path


DEFAULT_QUERIES_PATH = Path(__file__).with_name("arxiv_queries.json")
DEFAULT_QUERY_NAME = "agent_core"


def load_arxiv_queries(path: Path = DEFAULT_QUERIES_PATH) -> dict[str, str]:
    """Return validated query names and non-empty arXiv expressions."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("arXiv query configuration must be a non-empty object")

    queries: dict[str, str] = {}
    for name, query in payload.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("arXiv query names must be non-empty strings")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"arXiv query {name!r} must be a non-empty string")
        queries[name.strip()] = query.strip()

    if DEFAULT_QUERY_NAME not in queries:
        raise ValueError(f"missing default arXiv query: {DEFAULT_QUERY_NAME}")
    return queries


def get_arxiv_query(name: str, path: Path = DEFAULT_QUERIES_PATH) -> str:
    """Resolve one named query or report the available names."""
    queries = load_arxiv_queries(path)
    try:
        return queries[name]
    except KeyError as error:
        available = ", ".join(sorted(queries))
        raise ValueError(
            f"unknown arXiv query name {name!r}; available: {available}"
        ) from error
