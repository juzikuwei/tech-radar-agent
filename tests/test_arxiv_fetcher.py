import pytest

from ingestion.arxiv_fetcher import fetch_papers


@pytest.mark.parametrize(
    ("query", "max_results"),
    [
        ("", 3),
        ("   ", 3),
        ("all:agent", 0),
        ("all:agent", -1),
    ],
)
def test_fetch_papers_rejects_invalid_input(query: str, max_results: int) -> None:
    with pytest.raises(ValueError):
        fetch_papers(query, max_results)


def test_fetch_papers_rejects_invalid_page_size() -> None:
    with pytest.raises(ValueError, match="page_size"):
        fetch_papers("all:agent", 3, page_size=0)
