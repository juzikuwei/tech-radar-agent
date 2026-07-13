from pathlib import Path
import json

from fastapi.testclient import TestClient
import pytest

import api.services.chat as chat_service
from api.application import create_app
from config.model_settings import ModelSettings
from ingestion.repository import import_jsonl_snapshot
from ingestion.snapshot import write_jsonl_snapshot
from rag.application import RagResult
from rag.conversation import ConversationDecision
from rag.execution_trace import TraceEvent
from rag.research_agent import ResearchAgentError
from rag.runtime import RagRuntime


class FakeCollection:
    def count(self) -> int:
        return 1


def make_record(arxiv_id: str = "2607.00001") -> dict[str, object]:
    return {
        "arxiv_id": arxiv_id,
        "versioned_arxiv_id": f"{arxiv_id}v1",
        "raw_title": "Agent systems",
        "raw_abstract": "Raw abstract",
        "title": "Agent systems",
        "abstract": "Normalized abstract",
        "content_hash": "hash-v1",
        "authors": ["Ada Example"],
        "categories": ["cs.AI"],
        "primary_category": "cs.AI",
        "published_at": "2026-07-01T00:00:00+00:00",
        "updated_at": "2026-07-09T00:00:00+00:00",
        "entry_url": f"https://arxiv.org/abs/{arxiv_id}v1",
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}v1",
        "query": "all:agent",
        "fetched_at": "2026-07-10T00:00:00+00:00",
    }


def make_database(tmp_path: Path) -> Path:
    snapshot_path = tmp_path / "papers.jsonl"
    database_path = tmp_path / "papers.db"
    write_jsonl_snapshot([make_record()], snapshot_path)
    import_jsonl_snapshot(snapshot_path, database_path)
    return database_path


def make_runtime(database_path: Path) -> RagRuntime:
    return RagRuntime(
        collection=FakeCollection(),  # type: ignore[arg-type]
        embedder=object(),  # type: ignore[arg-type]
        reranker=object(),  # type: ignore[arg-type]
        settings=ModelSettings("key", "https://example.test", "model"),
        database_path=database_path,
    )


def test_health_and_knowledge_base_stats(tmp_path: Path) -> None:
    runtime = make_runtime(make_database(tmp_path))
    app = create_app(lambda: runtime)

    with TestClient(app) as client:
        health = client.get("/health")
        stats = client.get("/knowledge-base/stats")

    assert health.json() == {"status": "ok"}
    assert stats.json() == {"paper_count": 1, "vector_count": 1}


def test_allows_local_react_development_origin(tmp_path: Path) -> None:
    runtime = make_runtime(make_database(tmp_path))
    app = create_app(lambda: runtime)

    with TestClient(app) as client:
        response = client.options(
            "/chat",
            headers={
                "Origin": "http://127.0.0.1:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == (
        "http://127.0.0.1:5173"
    )


def test_chat_reloads_active_evidence_and_serializes_trace(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    runtime = make_runtime(make_database(tmp_path))
    observed: dict[str, object] = {}

    def fake_run_rag(question: str, **kwargs: object) -> RagResult:
        observed["question"] = question
        observed["history"] = kwargs["conversation_history"]
        observed["active_evidence"] = kwargs["active_evidence"]
        papers = kwargs["active_evidence"]
        return RagResult(
            question=question,
            papers=papers,  # type: ignore[arg-type]
            answer="grounded answer",
            generation_error=None,
            retrieval_attempts=0,
            standalone_question="resolved question",
            conversation_decision=ConversationDecision(
                coverage="sufficient",
                next_action="answer_from_existing",
                reason="existing evidence is sufficient",
                standalone_question="resolved question",
                reusable_arxiv_ids=("2607.00001",),
                missing_aspects=(),
                retrieval_query=None,
            ),
            trace=(
                TraceEvent(
                    stage="answer_generation",
                    label="Answer",
                    status="completed",
                    duration_ms=12.5,
                    details={"paper_count": 1},
                ),
            ),
        )

    monkeypatch.setattr(chat_service, "run_rag", fake_run_rag)
    app = create_app(lambda: runtime)
    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "question": "  follow-up question  ",
                "conversation_history": [
                    {
                        "user_message": "first question",
                        "assistant_message": "first answer",
                    }
                ],
                "active_evidence_ids": ["2607.00001"],
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "grounded answer"
    assert body["papers"][0]["arxiv_id"] == "2607.00001"
    assert body["trace"][0]["stage"] == "answer_generation"
    assert body["conversation_decision"]["next_action"] == "answer_from_existing"
    assert observed["question"] == "follow-up question"
    history = observed["history"]
    assert history[0].user_message == "first question"  # type: ignore[index,union-attr]
    evidence = observed["active_evidence"]
    assert evidence[0].document == "Agent systems\nNormalized abstract"  # type: ignore[index,union-attr]


def test_chat_stream_emits_trace_before_complete_result(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    runtime = make_runtime(make_database(tmp_path))

    def fake_run_rag(question: str, **kwargs: object) -> RagResult:
        event = TraceEvent(
            stage="dense_retrieval",
            label="E5 vector retrieval",
            status="completed",
            duration_ms=8.0,
            details={"result_count": 1},
        )
        on_trace = kwargs["on_trace"]
        on_trace(event)  # type: ignore[operator]
        return RagResult(
            question=question,
            papers=(),
            answer="complete answer",
            generation_error=None,
            trace=(event,),
        )

    monkeypatch.setattr(chat_service, "run_rag", fake_run_rag)
    app = create_app(lambda: runtime)

    with TestClient(app) as client:
        response = client.post(
            "/chat/stream",
            json={"question": "stream this trace"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in response.text.splitlines()]
    assert [event["type"] for event in events] == [
        "run_started",
        "trace",
        "result",
    ]
    assert events[1]["event"]["stage"] == "dense_retrieval"
    assert events[2]["result"]["answer"] == "complete answer"
    assert response.headers["x-accel-buffering"] == "no"


def test_react_mode_uses_research_agent(tmp_path: Path, monkeypatch: object) -> None:
    runtime = make_runtime(make_database(tmp_path))

    monkeypatch.setattr(
        chat_service,
        "run_research_agent",
        lambda question, **kwargs: RagResult(
            question=question,
            papers=(),
            answer="react answer",
            generation_error=None,
            trace=(),
        ),
    )
    monkeypatch.setattr(
        chat_service,
        "run_rag",
        lambda *args, **kwargs: pytest.fail("pipeline should not run"),
    )
    app = create_app(lambda: runtime)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"question": "comparison", "mode": "react"},
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "react answer"
    assert response.json()["mode"] == "react"
    assert response.json()["fallback_used"] is False


def test_react_failure_falls_back_and_preserves_trace(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    runtime = make_runtime(make_database(tmp_path))
    failed_event = TraceEvent(
        stage="agent_decision",
        label="Agent planning failed",
        status="failed",
        duration_ms=4.0,
        details={"error": "invalid plan"},
    )
    monkeypatch.setattr(
        chat_service,
        "run_research_agent",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ResearchAgentError("invalid plan", (failed_event,))
        ),
    )
    monkeypatch.setattr(
        chat_service,
        "run_rag",
        lambda question, **kwargs: RagResult(
            question=question,
            papers=(),
            answer="pipeline fallback",
            generation_error=None,
            trace=(),
        ),
    )
    app = create_app(lambda: runtime)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"question": "comparison", "mode": "react"},
        )

    body = response.json()
    assert body["answer"] == "pipeline fallback"
    assert body["fallback_used"] is True
    assert [event["stage"] for event in body["trace"]] == [
        "agent_decision",
        "react_fallback",
    ]


def test_chat_rejects_unknown_or_duplicate_active_evidence(tmp_path: Path) -> None:
    runtime = make_runtime(make_database(tmp_path))
    app = create_app(lambda: runtime)

    with TestClient(app) as client:
        unknown = client.post(
            "/chat",
            json={
                "question": "question",
                "active_evidence_ids": ["2607.99999"],
            },
        )
        duplicate = client.post(
            "/chat",
            json={
                "question": "question",
                "active_evidence_ids": ["2607.00001", "2607.00001"],
            },
        )

    assert unknown.status_code == 422
    assert unknown.json()["detail"] == {
        "unknown_active_evidence_ids": ["2607.99999"]
    }
    assert duplicate.status_code == 422
    assert "duplicates" in duplicate.json()["detail"]


def test_chat_rejects_more_than_six_conversation_turns(tmp_path: Path) -> None:
    runtime = make_runtime(make_database(tmp_path))
    app = create_app(lambda: runtime)
    turns = [
        {"user_message": f"question {index}", "assistant_message": "answer"}
        for index in range(7)
    ]

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"question": "follow-up", "conversation_history": turns},
        )

    assert response.status_code == 422
    assert "at most 6 turns" in response.json()["detail"]
