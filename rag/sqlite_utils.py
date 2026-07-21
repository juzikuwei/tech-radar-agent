"""Shared SQLite connection defaults for the single project database."""

from pathlib import Path
import sqlite3


BUSY_TIMEOUT_MS = 5_000


def open_connection(database_path: Path) -> sqlite3.Connection:
    """Open a connection with a bounded busy wait and enforced foreign keys."""
    connection = sqlite3.connect(
        database_path,
        timeout=BUSY_TIMEOUT_MS / 1_000,
    )
    connection.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def enable_wal_mode(connection: sqlite3.Connection) -> None:
    """Switch the database file to WAL so readers do not block the writer."""
    connection.execute("PRAGMA journal_mode = WAL")
