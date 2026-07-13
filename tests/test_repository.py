from copy import deepcopy
from pathlib import Path
import sqlite3

import pytest

from ingestion.repository import (
    SnapshotImportError,
    get_paper_count,
    import_jsonl_snapshot,
    load_papers_by_arxiv_ids,
    load_papers_for_embedding,
)
from ingestion.snapshot import write_jsonl_snapshot


def make_record(
    *,
    arxiv_id: str,
    version: int = 1,
    title: str = "Agent systems",
    content_hash: str = "hash-v1",
    updated_at: str = "2026-07-09T00:00:00+00:00",
    fetched_at: str = "2026-07-10T00:00:00+00:00",
) -> dict[str, object]:
    versioned_id = f"{arxiv_id}v{version}"
    return {
        "arxiv_id": arxiv_id,
        "versioned_arxiv_id": versioned_id,
        "raw_title": title,
        "raw_abstract": "Raw abstract",
        "title": title,
        "abstract": "Normalized abstract",
        "content_hash": content_hash,
        "authors": ["Ada Example"],
        "categories": ["cs.AI"],
        "primary_category": "cs.AI",
        "published_at": "2026-07-01T00:00:00+00:00",
        "updated_at": updated_at,
        "entry_url": f"https://arxiv.org/abs/{versioned_id}",
        "pdf_url": f"https://arxiv.org/pdf/{versioned_id}",
        "query": "all:agent",
        "fetched_at": fetched_at,
    }


def write_snapshot(path: Path, records: list[dict[str, object]]) -> None:
    write_jsonl_snapshot(records, path)


def fetch_paper(database_path: Path, arxiv_id: str) -> sqlite3.Row | None:
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        return connection.execute(
            "SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)
        ).fetchone()


def count_papers(database_path: Path) -> int:
    with sqlite3.connect(database_path) as connection:
        return connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0]


def test_importing_same_snapshot_twice_is_idempotent(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "papers.jsonl"
    database_path = tmp_path / "papers.db"
    records = [make_record(arxiv_id="2607.00001"), make_record(arxiv_id="2607.00002")]
    write_snapshot(snapshot_path, records)

    first = import_jsonl_snapshot(snapshot_path, database_path)
    second = import_jsonl_snapshot(snapshot_path, database_path)

    assert (first.inserted, first.updated, first.unchanged) == (2, 0, 0)
    assert (second.inserted, second.updated, second.unchanged) == (0, 0, 2)
    assert count_papers(database_path) == 2


def test_newer_revision_updates_existing_paper(tmp_path: Path) -> None:
    database_path = tmp_path / "papers.db"
    first_snapshot = tmp_path / "v1.jsonl"
    second_snapshot = tmp_path / "v2.jsonl"
    write_snapshot(first_snapshot, [make_record(arxiv_id="2607.00001")])
    write_snapshot(
        second_snapshot,
        [
            make_record(
                arxiv_id="2607.00001",
                version=2,
                title="Updated agent systems",
                content_hash="hash-v2",
                updated_at="2026-07-11T00:00:00+00:00",
                fetched_at="2026-07-11T01:00:00+00:00",
            )
        ],
    )

    import_jsonl_snapshot(first_snapshot, database_path)
    stats = import_jsonl_snapshot(second_snapshot, database_path)
    paper = fetch_paper(database_path, "2607.00001")

    assert stats.updated == 1
    assert paper is not None
    assert paper["versioned_arxiv_id"] == "2607.00001v2"
    assert paper["content_hash"] == "hash-v2"
    assert paper["first_seen_at"] == "2026-07-10T00:00:00+00:00"
    assert paper["last_seen_at"] == "2026-07-11T01:00:00+00:00"


def test_older_revision_cannot_downgrade_current_state(tmp_path: Path) -> None:
    database_path = tmp_path / "papers.db"
    newer_snapshot = tmp_path / "v2.jsonl"
    older_snapshot = tmp_path / "v1.jsonl"
    write_snapshot(
        newer_snapshot,
        [
            make_record(
                arxiv_id="2607.00001",
                version=2,
                content_hash="hash-v2",
                updated_at="2026-07-11T00:00:00+00:00",
            )
        ],
    )
    write_snapshot(
        older_snapshot,
        [
            make_record(
                arxiv_id="2607.00001",
                version=1,
                content_hash="hash-v1",
                updated_at="2026-07-09T00:00:00+00:00",
                fetched_at="2026-07-12T00:00:00+00:00",
            )
        ],
    )

    import_jsonl_snapshot(newer_snapshot, database_path)
    stats = import_jsonl_snapshot(older_snapshot, database_path)
    paper = fetch_paper(database_path, "2607.00001")

    assert stats.skipped_older == 1
    assert paper is not None
    assert paper["versioned_arxiv_id"] == "2607.00001v2"
    assert paper["content_hash"] == "hash-v2"
    assert paper["last_seen_at"] == "2026-07-12T00:00:00+00:00"


def test_invalid_record_rolls_back_the_whole_snapshot(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "invalid.jsonl"
    database_path = tmp_path / "papers.db"
    valid = make_record(arxiv_id="2607.00001")
    invalid = deepcopy(make_record(arxiv_id="2607.00002"))
    del invalid["abstract"]
    write_snapshot(snapshot_path, [valid, invalid])

    with pytest.raises(SnapshotImportError):
        import_jsonl_snapshot(snapshot_path, database_path)

    assert count_papers(database_path) == 0


def test_ambiguous_same_timestamp_rolls_back_new_rows(tmp_path: Path) -> None:
    database_path = tmp_path / "papers.db"
    initial_snapshot = tmp_path / "initial.jsonl"
    conflicting_snapshot = tmp_path / "conflict.jsonl"
    original = make_record(arxiv_id="2607.00001")
    conflicting = make_record(arxiv_id="2607.00001", content_hash="different-hash")
    write_snapshot(initial_snapshot, [original])
    write_snapshot(
        conflicting_snapshot,
        [make_record(arxiv_id="2607.00002"), conflicting],
    )
    import_jsonl_snapshot(initial_snapshot, database_path)

    with pytest.raises(SnapshotImportError):
        import_jsonl_snapshot(conflicting_snapshot, database_path)

    assert count_papers(database_path) == 1
    assert fetch_paper(database_path, "2607.00002") is None


def test_load_papers_for_embedding_returns_current_normalized_text(tmp_path: Path) -> None:
    database_path = tmp_path / "papers.db"
    snapshot_path = tmp_path / "papers.jsonl"
    write_snapshot(
        snapshot_path,
        [make_record(arxiv_id="2607.00001", title="Normalized title")],
    )
    import_jsonl_snapshot(snapshot_path, database_path)

    papers = load_papers_for_embedding(database_path)

    assert len(papers) == 1
    assert papers[0]["arxiv_id"] == "2607.00001"
    assert papers[0]["title"] == "Normalized title"
    assert papers[0]["abstract"] == "Normalized abstract"


def test_loads_requested_papers_in_id_order_and_reports_count(tmp_path: Path) -> None:
    database_path = tmp_path / "papers.db"
    snapshot_path = tmp_path / "papers.jsonl"
    write_snapshot(
        snapshot_path,
        [
            make_record(arxiv_id="2607.00001", title="First paper"),
            make_record(arxiv_id="2607.00002", title="Second paper"),
        ],
    )
    import_jsonl_snapshot(snapshot_path, database_path)

    papers = load_papers_by_arxiv_ids(
        database_path,
        ["2607.00002", "2607.00001", "2607.99999"],
    )

    assert [paper["arxiv_id"] for paper in papers] == [
        "2607.00002",
        "2607.00001",
    ]
    assert papers[0]["title"] == "Second paper"
    assert get_paper_count(database_path) == 2
