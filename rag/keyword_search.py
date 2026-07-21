"""SQLite FTS5 keyword retrieval over normalized paper titles and abstracts."""

from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3

from rag.search import SearchResult
from rag.sqlite_utils import enable_wal_mode, open_connection


# Anchored to the repository root so services started from another working
# directory never silently create a second, empty database.
DEFAULT_DATABASE_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "tech_radar.db"
)
TITLE_WEIGHT = 4.0
ABSTRACT_WEIGHT = 1.0

FTS_SCHEMA_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
    title,
    abstract,
    content='papers',
    content_rowid='rowid',
    tokenize='porter unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS papers_fts_after_insert
AFTER INSERT ON papers BEGIN
    INSERT INTO papers_fts(rowid, title, abstract)
    VALUES (new.rowid, new.title, new.abstract);
END;

CREATE TRIGGER IF NOT EXISTS papers_fts_after_delete
AFTER DELETE ON papers BEGIN
    INSERT INTO papers_fts(papers_fts, rowid, title, abstract)
    VALUES ('delete', old.rowid, old.title, old.abstract);
END;

CREATE TRIGGER IF NOT EXISTS papers_fts_after_update
AFTER UPDATE OF title, abstract ON papers BEGIN
    INSERT INTO papers_fts(papers_fts, rowid, title, abstract)
    VALUES ('delete', old.rowid, old.title, old.abstract);
    INSERT INTO papers_fts(rowid, title, abstract)
    VALUES (new.rowid, new.title, new.abstract);
END;
"""


@dataclass(frozen=True)
class KeywordIndexStats:
    """Observable state after ensuring the FTS5 index exists."""

    paper_count: int
    indexed_count: int
    rebuilt: bool


def ensure_keyword_index(
    database_path: Path = DEFAULT_DATABASE_PATH,
) -> KeywordIndexStats:
    """Create and synchronize the FTS5 index used by BM25 retrieval."""
    if not database_path.exists():
        raise FileNotFoundError(f"database not found: {database_path}")

    with closing(open_connection(database_path)) as connection, connection:
        enable_wal_mode(connection)
        table_existed = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'papers_fts'"
        ).fetchone() is not None
        connection.executescript(FTS_SCHEMA_SQL)

        paper_count = int(
            connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        )
        indexed_count = int(
            connection.execute("SELECT COUNT(*) FROM papers_fts").fetchone()[0]
        )
        rebuilt = not table_existed or indexed_count != paper_count
        if rebuilt:
            connection.execute(
                "INSERT INTO papers_fts(papers_fts) VALUES ('rebuild')"
            )
            indexed_count = int(
                connection.execute("SELECT COUNT(*) FROM papers_fts").fetchone()[0]
            )
        connection.commit()

    return KeywordIndexStats(
        paper_count=paper_count,
        indexed_count=indexed_count,
        rebuilt=rebuilt,
    )


def search_keyword_papers(
    query: str,
    *,
    top_k: int,
    database_path: Path = DEFAULT_DATABASE_PATH,
) -> list[SearchResult]:
    """Return title and abstract matches ranked by SQLite BM25."""
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    match_query = build_match_query(clean_query)
    if not match_query:
        return []

    # Index maintenance is intentionally not part of this hot path: the API
    # runtime ensures the FTS index at startup and ingestion keeps it in sync
    # via triggers. The lazy retry below only covers a database that has never
    # been indexed at all.
    try:
        rows = _query_keyword_rows(match_query, top_k, database_path)
    except sqlite3.OperationalError as error:
        if "no such table" not in str(error).lower():
            raise
        ensure_keyword_index(database_path)
        rows = _query_keyword_rows(match_query, top_k, database_path)

    return [
        SearchResult(
            arxiv_id=str(row["arxiv_id"]),
            title=str(row["title"]),
            document=f"{row['title']}\n{row['abstract']}",
            entry_url=str(row["entry_url"]),
            primary_category=str(row["primary_category"]),
            similarity=None,
            keyword_score=-float(row["bm25_score"]),
        )
        for row in rows
    ]


def _query_keyword_rows(
    match_query: str,
    top_k: int,
    database_path: Path,
) -> list[sqlite3.Row]:
    with closing(open_connection(database_path)) as connection, connection:
        connection.row_factory = sqlite3.Row
        return connection.execute(
            """
            SELECT p.arxiv_id, p.title, p.abstract, p.entry_url,
                   p.primary_category,
                   bm25(papers_fts, ?, ?) AS bm25_score
            FROM papers_fts
            JOIN papers AS p ON p.rowid = papers_fts.rowid
            WHERE papers_fts MATCH ?
            ORDER BY bm25_score ASC, p.arxiv_id ASC
            LIMIT ?
            """,
            (TITLE_WEIGHT, ABSTRACT_WEIGHT, match_query, top_k),
        ).fetchall()


def build_match_query(query: str) -> str:
    """Build a safe recall-oriented FTS5 OR query from user text."""
    tokens = re.findall(r"[^\W_]+", query, flags=re.UNICODE)
    useful_tokens = [token for token in tokens if len(token) >= 2]
    quoted_tokens = [
        f'"{token.replace(chr(34), chr(34) * 2)}"'
        for token in useful_tokens
    ]
    return " OR ".join(quoted_tokens)
