from datetime import datetime, timezone
from types import SimpleNamespace

from ingestion.normalizer import normalize_text, to_paper_record


def make_paper(*, title: str, summary: str, short_id: str = "2607.08716v2"):
    paper = SimpleNamespace(
        title=title,
        summary=summary,
        authors=[SimpleNamespace(name="Ada Example")],
        categories=["cs.AI", "cs.CL"],
        primary_category="cs.AI",
        published=datetime(2026, 7, 9, tzinfo=timezone.utc),
        updated=datetime(2026, 7, 10, tzinfo=timezone.utc),
        entry_id=f"https://arxiv.org/abs/{short_id}",
        pdf_url=f"https://arxiv.org/pdf/{short_id}",
        get_short_id=lambda: short_id,
    )
    return paper


def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text("Agent\n  systems\tlearn") == "Agent systems learn"


def test_paper_record_uses_versionless_identity() -> None:
    record = to_paper_record(
        make_paper(title="Agent", summary="Summary"),
        query="all:agent",
        fetched_at=datetime(2026, 7, 11, tzinfo=timezone.utc),
    )

    assert record["arxiv_id"] == "2607.08716"
    assert record["versioned_arxiv_id"] == "2607.08716v2"


def test_paper_record_keeps_raw_and_normalized_text() -> None:
    record = to_paper_record(
        make_paper(title="Agent\n  systems", summary="One\t summary"),
        query="all:agent",
        fetched_at=datetime(2026, 7, 11, tzinfo=timezone.utc),
    )

    assert record["raw_title"] == "Agent\n  systems"
    assert record["raw_abstract"] == "One\t summary"
    assert record["title"] == "Agent systems"
    assert record["abstract"] == "One summary"


def test_content_hash_ignores_whitespace_only_changes() -> None:
    fetched_at = datetime(2026, 7, 11, tzinfo=timezone.utc)
    first = to_paper_record(
        make_paper(title="Agent  systems", summary="One\nsummary"),
        query="all:agent",
        fetched_at=fetched_at,
    )
    second = to_paper_record(
        make_paper(title="Agent systems", summary="One summary"),
        query="all:agent",
        fetched_at=fetched_at,
    )

    assert first["content_hash"] == second["content_hash"]
