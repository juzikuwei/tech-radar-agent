"""SQLite current-state storage for normalized paper records."""

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import sqlite3
from typing import Any


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id TEXT PRIMARY KEY,
    versioned_arxiv_id TEXT NOT NULL,
    raw_title TEXT NOT NULL,
    raw_abstract TEXT NOT NULL,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    authors_json TEXT NOT NULL,
    categories_json TEXT NOT NULL,
    primary_category TEXT NOT NULL,
    published_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    entry_url TEXT NOT NULL,
    pdf_url TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class ImportStats:
    """Observable result of importing one complete snapshot."""

    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    skipped_older: int = 0


class SnapshotImportError(ValueError):
    """Raised when a snapshot record violates the local data contract."""


def initialize_database(connection: sqlite3.Connection) -> None:
    """Create the current-state schema before a snapshot transaction starts."""
    connection.executescript(SCHEMA_SQL)
    connection.commit()


def import_jsonl_snapshot(snapshot_path: Path, database_path: Path) -> ImportStats:
    """Import one JSONL snapshot atomically into the SQLite current state."""
    database_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        initialize_database(connection)
        connection.execute("BEGIN IMMEDIATE")

        try:
            stats = _import_snapshot_lines(connection, snapshot_path)
        except Exception:
            connection.rollback()
            raise
        else:
            connection.commit()
            return stats


def load_papers_for_embedding(database_path: Path) -> list[dict[str, str]]:
    """Load the current normalized paper text required by an embedding index."""
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT arxiv_id, versioned_arxiv_id, title, abstract, content_hash,
                   primary_category, published_at, updated_at, entry_url
            FROM papers
            ORDER BY updated_at DESC, arxiv_id ASC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def _import_snapshot_lines(
    connection: sqlite3.Connection,
    snapshot_path: Path,
) -> ImportStats:
    inserted = 0
    updated = 0
    unchanged = 0
    skipped_older = 0

    with snapshot_path.open("r", encoding="utf-8") as snapshot_file:
        for line_number, line in enumerate(snapshot_file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SnapshotImportError(
                    f"invalid JSON at {snapshot_path}:{line_number}"
                ) from exc

            try:
                outcome = _upsert_record(connection, record)
            except (KeyError, TypeError, ValueError) as exc:
                raise SnapshotImportError(
                    f"invalid record at {snapshot_path}:{line_number}: {exc}"
                ) from exc

            if outcome == "inserted":
                inserted += 1
            elif outcome == "updated":
                updated += 1
            elif outcome == "unchanged":
                unchanged += 1
            elif outcome == "skipped_older":
                skipped_older += 1

    return ImportStats(
        inserted=inserted,
        updated=updated,
        unchanged=unchanged,
        skipped_older=skipped_older,
    )


def _upsert_record(connection: sqlite3.Connection, record: Any) -> str:
    if not isinstance(record, dict):
        raise TypeError("record must be a JSON object")

    normalized = _validate_record(record)
    existing = connection.execute(
        """
        SELECT updated_at, content_hash, last_seen_at
        FROM papers
        WHERE arxiv_id = ?
        """,
        (normalized["arxiv_id"],),
    ).fetchone()

    if existing is None:
        _insert_record(connection, normalized)
        return "inserted"

    incoming_updated_at = _parse_timestamp(
        normalized["updated_at"], field_name="updated_at"
    )
    existing_updated_at = _parse_timestamp(
        existing["updated_at"], field_name="stored updated_at"
    )
    newest_seen_at = max(
        _parse_timestamp(normalized["fetched_at"], field_name="fetched_at"),
        _parse_timestamp(existing["last_seen_at"], field_name="stored last_seen_at"),
    ).isoformat()

    if incoming_updated_at < existing_updated_at:
        connection.execute(
            "UPDATE papers SET last_seen_at = ? WHERE arxiv_id = ?",
            (newest_seen_at, normalized["arxiv_id"]),
        )
        return "skipped_older"

    if incoming_updated_at == existing_updated_at:
        if normalized["content_hash"] != existing["content_hash"]:
            raise ValueError(
                "same updated_at has a different content_hash; refusing ambiguous update"
            )
        connection.execute(
            "UPDATE papers SET last_seen_at = ? WHERE arxiv_id = ?",
            (newest_seen_at, normalized["arxiv_id"]),
        )
        return "unchanged"

    _update_record(connection, normalized, newest_seen_at)
    return "updated"


def _validate_record(record: dict[str, Any]) -> dict[str, Any]:
    string_fields = (
        "arxiv_id",
        "versioned_arxiv_id",
        "raw_title",
        "raw_abstract",
        "title",
        "abstract",
        "content_hash",
        "primary_category",
        "published_at",
        "updated_at",
        "entry_url",
        "pdf_url",
        "fetched_at",
    )
    normalized = {field: _require_string(record, field) for field in string_fields}
    normalized["authors"] = _require_string_list(record, "authors")
    normalized["categories"] = _require_string_list(record, "categories")

    _parse_timestamp(normalized["published_at"], field_name="published_at")
    _parse_timestamp(normalized["updated_at"], field_name="updated_at")
    _parse_timestamp(normalized["fetched_at"], field_name="fetched_at")
    return normalized


def _require_string(record: dict[str, Any], field: str) -> str:
    value = record[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_string_list(record: dict[str, Any], field: str) -> list[str]:
    value = record[field]
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{field} must be a list of non-empty strings")
    return value


def _parse_timestamp(value: str, *, field_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone offset")
    return parsed


def _insert_record(connection: sqlite3.Connection, record: dict[str, Any]) -> None:
    connection.execute(
        """
        INSERT INTO papers (
            arxiv_id, versioned_arxiv_id, raw_title, raw_abstract,
            title, abstract, content_hash, authors_json, categories_json,
            primary_category, published_at, updated_at, entry_url, pdf_url,
            first_seen_at, last_seen_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        _record_values(record, first_seen_at=record["fetched_at"], last_seen_at=record["fetched_at"]),
    )


def _update_record(
    connection: sqlite3.Connection,
    record: dict[str, Any],
    last_seen_at: str,
) -> None:
    connection.execute(
        """
        UPDATE papers SET
            versioned_arxiv_id = ?, raw_title = ?, raw_abstract = ?,
            title = ?, abstract = ?, content_hash = ?, authors_json = ?,
            categories_json = ?, primary_category = ?, published_at = ?,
            updated_at = ?, entry_url = ?, pdf_url = ?, last_seen_at = ?
        WHERE arxiv_id = ?
        """,
        (
            record["versioned_arxiv_id"],
            record["raw_title"],
            record["raw_abstract"],
            record["title"],
            record["abstract"],
            record["content_hash"],
            json.dumps(record["authors"], ensure_ascii=False),
            json.dumps(record["categories"], ensure_ascii=False),
            record["primary_category"],
            record["published_at"],
            record["updated_at"],
            record["entry_url"],
            record["pdf_url"],
            last_seen_at,
            record["arxiv_id"],
        ),
    )


def _record_values(
    record: dict[str, Any],
    *,
    first_seen_at: str,
    last_seen_at: str,
) -> tuple[object, ...]:
    return (
        record["arxiv_id"],
        record["versioned_arxiv_id"],
        record["raw_title"],
        record["raw_abstract"],
        record["title"],
        record["abstract"],
        record["content_hash"],
        json.dumps(record["authors"], ensure_ascii=False),
        json.dumps(record["categories"], ensure_ascii=False),
        record["primary_category"],
        record["published_at"],
        record["updated_at"],
        record["entry_url"],
        record["pdf_url"],
        first_seen_at,
        last_seen_at,
    )
