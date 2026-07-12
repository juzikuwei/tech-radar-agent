"""Atomic JSONL snapshot persistence."""

from collections.abc import Iterable, Mapping
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def write_jsonl_snapshot(
    records: Iterable[Mapping[str, object]],
    output_path: Path,
) -> int:
    """Publish a JSONL file only after every record is written successfully."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    record_count = 0

    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f"{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            for record in records:
                json.dump(record, temporary_file, ensure_ascii=False)
                temporary_file.write("\n")
                record_count += 1

            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        temporary_path.replace(output_path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    return record_count
