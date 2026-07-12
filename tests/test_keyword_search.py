from pathlib import Path
import sqlite3

from ingestion.repository import initialize_database
from rag.keyword_search import ensure_keyword_index, search_keyword_papers


def insert_paper(
    database_path: Path,
    *,
    arxiv_id: str,
    title: str,
    abstract: str,
) -> None:
    """Insert one minimal valid paper into a test database."""
    with sqlite3.connect(database_path) as connection:
        initialize_database(connection)
        connection.execute(
            """
            INSERT INTO papers (
                arxiv_id, versioned_arxiv_id, raw_title, raw_abstract,
                title, abstract, content_hash, authors_json, categories_json,
                primary_category, published_at, updated_at, entry_url, pdf_url,
                first_seen_at, last_seen_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, '[]', '[]', 'cs.AI', ?, ?, ?, ?, ?, ?)
            """,
            (
                arxiv_id,
                f"{arxiv_id}v1",
                title,
                abstract,
                title,
                abstract,
                f"hash-{arxiv_id}",
                "2026-07-01T00:00:00+00:00",
                "2026-07-01T00:00:00+00:00",
                f"https://arxiv.org/abs/{arxiv_id}",
                f"https://arxiv.org/pdf/{arxiv_id}",
                "2026-07-01T00:00:00+00:00",
                "2026-07-01T00:00:00+00:00",
            ),
        )
        connection.commit()


def test_bm25_ranks_rare_exact_title_term_first(tmp_path: Path) -> None:
    database_path = tmp_path / "papers.db"
    insert_paper(
        database_path,
        arxiv_id="2607.00001",
        title="SkillCenter for Autonomous Agents",
        abstract="A source-grounded skill library.",
    )
    insert_paper(
        database_path,
        arxiv_id="2607.00002",
        title="General Agent Memory",
        abstract="Memory for long-running autonomous systems.",
    )

    results = search_keyword_papers(
        "SkillCenter quality gating",
        top_k=2,
        database_path=database_path,
    )

    assert results[0].arxiv_id == "2607.00001"
    assert results[0].keyword_score is not None
    assert results[0].similarity is None


def test_keyword_index_tracks_later_paper_updates(tmp_path: Path) -> None:
    database_path = tmp_path / "papers.db"
    insert_paper(
        database_path,
        arxiv_id="2607.00001",
        title="General Agent Study",
        abstract="A general abstract.",
    )
    first_stats = ensure_keyword_index(database_path)

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE papers SET title = ? WHERE arxiv_id = ?",
            ("Prismata Prompt Injection Defense", "2607.00001"),
        )
        connection.commit()

    results = search_keyword_papers(
        "Prismata",
        top_k=1,
        database_path=database_path,
    )

    assert first_stats.rebuilt is True
    assert results[0].arxiv_id == "2607.00001"
