import json
from pathlib import Path

import pytest

from ingestion.snapshot import write_jsonl_snapshot


def test_write_jsonl_snapshot_publishes_complete_batch(tmp_path: Path) -> None:
    output_path = tmp_path / "papers.jsonl"
    records = [{"arxiv_id": "1"}, {"arxiv_id": "2"}]

    count = write_jsonl_snapshot(records, output_path)

    saved_records = [json.loads(line) for line in output_path.read_text("utf-8").splitlines()]
    assert count == 2
    assert saved_records == records
    assert list(tmp_path.glob("*.tmp")) == []


def test_failed_snapshot_does_not_replace_previous_file(tmp_path: Path) -> None:
    output_path = tmp_path / "papers.jsonl"
    output_path.write_text('{"arxiv_id": "old"}\n', encoding="utf-8")
    invalid_records = [
        {"arxiv_id": "new"},
        {"not_json_serializable": object()},
    ]

    with pytest.raises(TypeError):
        write_jsonl_snapshot(invalid_records, output_path)

    assert output_path.read_text("utf-8") == '{"arxiv_id": "old"}\n'
    assert list(tmp_path.glob("*.tmp")) == []
