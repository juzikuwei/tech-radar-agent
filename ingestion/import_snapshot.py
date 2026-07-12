"""Import one completed JSONL snapshot into the SQLite current state."""

import argparse
from pathlib import Path
import sqlite3

from ingestion.repository import SnapshotImportError, import_jsonl_snapshot


DEFAULT_DATABASE_PATH = Path("data/tech_radar.db")


def build_parser() -> argparse.ArgumentParser:
    """Build arguments for the manual snapshot import command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    return parser


def main() -> int:
    """Import one snapshot and print observable outcome counts."""
    args = build_parser().parse_args()

    try:
        stats = import_jsonl_snapshot(args.snapshot, args.database)
    except (OSError, sqlite3.Error, SnapshotImportError) as exc:
        print(f"Import failed: {exc}")
        return 1

    print(f"Inserted: {stats.inserted}")
    print(f"Updated: {stats.updated}")
    print(f"Unchanged: {stats.unchanged}")
    print(f"Skipped older: {stats.skipped_older}")
    print(f"Database: {args.database}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
