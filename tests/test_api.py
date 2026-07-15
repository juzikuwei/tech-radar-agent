from pathlib import Path
import json

from fastapi.testclient import TestClient
import pytest

import api.services.chat as chat_service
from api.application import create_app
from config.conversation_context_settings import ConversationContextSettings
from config.model_settings import ModelSettings
from ingestion.repository import import_jsonl_snapshot
from ingestion.snapshot import write_jsonl_snapshot
from rag.application import RagResult
from rag.conversation import ConversationDecision
from rag.conversation_store import (
    append_conversation_turn,
    create_conversation,
    initialize_conversation_store,
    load_conversation_state,
)
from rag.execution_trace import TraceEvent
from rag.research_agent import ResearchAgentError
from rag.runtime import RagRuntime


def parse_sse_events(body: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for block in body.replace("\r\n", "\n").split("\n\n"):
        data = "\n".join(
            line[5:].lstrip()
            for line in block.splitlines()
            if line.startswith("data:")
        )
        if data:
            events.append(json.loads(data))
    return events


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
    initialize_conversation_store(database_path)
    return database_path


def make_runtime(
    database_path: Path,
    *,
    context_settings: ConversationContextSettings | None = None,
) -> RagRuntime:
    return RagRuntime(
        collection=FakeCollection(),  # type: ignore[arg-type]
        embedder=object(),  # type: ignore[arg-type]
        reranker=object(),  # type: ignore[arg-type]
        settings=ModelSettings("key", "https://example.test", "model"),
        database_path=database_path,
        conversation_context_settings=(
            context_settings or ConversationContextSettings()
        ),
    )


def create_client_conversation(client: TestClient) -> str:
    response = client.post("/conversations")
    assert response.status_code == 201
    return str(response.json()["conversation_id"])


def test_health_and_knowledge_base_stats(tmp_path: Path) -> None:
    runtime = make_runtime(make_database(tmp_path))
    app = create_app(lambda: runtime)

    with TestClient(app) as client:
        health = client.get("/health")
        stats = client.get("/knowledge-base/stats")

    assert health.json() == {"status": "ok"}
    assert stats.json() == {"paper_count": 1, "vector_count": 1}


def test_allows_local_react_conversation_requests(tmp_path: Path) -> None:
    runtime = make_runtime(make_database(tmp_path))
    app = create_app(lambda: runtime)

    with TestClient(app) as client:
        response = client.options(
            "/conversations/example",
            headers={
                "Origin": "http://127.0.0.1:5173",
                "Access-Control-Request-Method": "DELETE",
                "Access-Control-Request-Headers": "content-type",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == (
        "http://127.0.0.1:5173"
    )


def test_conversation_crud_returns_complete_history(tmp_path: Path) -> None:
    runtime = make_runtime(make_database(tmp_path))
    app = create_app(lambda: runtime)

    with TestClient(app) as client:
        assert client.get("/conversations").json() == []
        conversation_id = create_client_conversation(client)
        listed = client.get("/conversations")
        loaded = client.get(f"/conversations/{conversation_id}")
        deleted = client.delete(f"/conversations/{conversation_id}")
        missing = client.get(f"/conversations/{conversation_id}")

    assert listed.json()[0]["title"] == "新对话"
    assert listed.json()[0]["turn_count"] == 0
    assert loaded.json()["turns"] == []
    assert deleted.status_code == 204
    assert missing.status_code == 404


def test_chat_loads_recent_state_and_persists_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = make_database(tmp_path)
    runtime = make_runtime(database_path)
    conversation = create_conversation(database_path)
    for index in range(7):
        append_conversation_turn(
            database_path,
            conversation.conversation_id,
            user_message=f"question {index}",
            assistant_message=f"answer {index}",
            paper_ids=("2607.00001",) if index == 6 else (),
        )
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
            f"/conversations/{conversation.conversation_id}/chat",
            json={"question": "  follow-up question  "},
        )
        stored = client.get(f"/conversations/{conversation.conversation_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "grounded answer"
    assert body["papers"][0]["arxiv_id"] == "2607.00001"
    assert body["trace"][0]["stage"] == "answer_generation"
    assert observed["question"] == "follow-up question"
    history = observed["history"]
    assert len(history) == 7  # type: ignore[arg-type]
    assert history[0].user_message == "question 0"  # type: ignore[index,union-attr]
    evidence = observed["active_evidence"]
    assert evidence[0].document == "Agent systems\nNormalized abstract"  # type: ignore[index,union-attr]
    stored_body = stored.json()
    assert stored_body["turn_count"] == 8
    assert stored_body["turns"][-1]["assistant_message"] == "grounded answer"
    assert stored_body["turns"][-1]["paper_ids"] == ["2607.00001"]
    assert stored_body["turns"][-1]["papers"][0]["rerank_score"] is None


def test_chat_compacts_context_before_running_rag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = make_database(tmp_path)
    runtime = make_runtime(
        database_path,
        context_settings=ConversationContextSettings(
            token_threshold=120,
            target_tokens=70,
        ),
    )
    conversation = create_conversation(database_path)
    for index in range(4):
        append_conversation_turn(
            database_path,
            conversation.conversation_id,
            user_message=f"original user {index} " + "目标" * 30,
            assistant_message=f"original assistant {index} " + "回答" * 30,
        )
    summary = json.dumps(
        {
            "user_goals": ["长期目标"],
            "confirmed_requirements": [],
            "decisions": [],
            "important_context": [],
            "open_questions": [],
        },
        ensure_ascii=False,
    )
    monkeypatch.setattr(
        "rag.context_compaction.generate_text",
        lambda *args, **kwargs: summary,
    )
    observed: dict[str, object] = {}

    def fake_run_rag(question: str, **kwargs: object) -> RagResult:
        observed.update(kwargs)
        return RagResult(
            question=question,
            papers=(),
            answer="answer",
            generation_error=None,
        )

    monkeypatch.setattr(chat_service, "run_rag", fake_run_rag)
    app = create_app(lambda: runtime)

    with TestClient(app) as client:
        response = client.post(
            f"/conversations/{conversation.conversation_id}/chat",
            json={"question": "continue"},
        )

    assert response.status_code == 200
    assert observed["context_summary"] is not None
    assert len(observed["conversation_history"]) < 4  # type: ignore[arg-type]
    assert response.json()["trace"][0]["stage"] == "conversation_compaction"


def test_chat_stream_emits_trace_then_persists_complete_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
        conversation_id = create_client_conversation(client)
        response = client.post(
            f"/conversations/{conversation_id}/chat/stream",
            json={"question": "stream this trace"},
        )
        stored = client.get(f"/conversations/{conversation_id}")

    events = parse_sse_events(response.text)
    assert [event["type"] for event in events] == [
        "run_started",
        "trace",
        "assistant_delta",
        "assistant_completed",
        "run_completed",
        "result",
    ]
    assert events[1]["event"]["stage"] == "dense_retrieval"
    assert events[2]["delta"] == "complete answer"
    assert events[-1]["result"]["answer"] == "complete answer"
    assert stored.json()["turn_count"] == 1
    assert response.headers["x-accel-buffering"] == "no"
    assert response.headers["content-type"].startswith("text/event-stream")


def test_react_mode_uses_research_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
        conversation_id = create_client_conversation(client)
        response = client.post(
            f"/conversations/{conversation_id}/chat",
            json={"question": "comparison", "mode": "react"},
        )

    assert response.json()["answer"] == "react answer"
    assert response.json()["mode"] == "react"
    assert response.json()["fallback_used"] is False


def test_react_initial_model_failure_falls_back_to_reliable_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
        conversation_id = create_client_conversation(client)
        response = client.post(
            f"/conversations/{conversation_id}/chat",
            json={"question": "comparison", "mode": "react"},
        )

    body = response.json()
    assert body["answer"] == "pipeline fallback"
    assert body["fallback_used"] is True
    assert [event["stage"] for event in body["trace"]] == [
        "agent_decision",
        "react_fallback",
    ]


def test_react_tool_failure_still_falls_back_to_reliable_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = make_runtime(make_database(tmp_path))
    failed_event = TraceEvent(
        stage="agent_tool_observation",
        label="tool failed",
        status="failed",
        duration_ms=4.0,
        details={"error": "search failed"},
    )
    monkeypatch.setattr(
        chat_service,
        "run_research_agent",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ResearchAgentError("search failed", (failed_event,), tool_calls=1)
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
        conversation_id = create_client_conversation(client)
        response = client.post(
            f"/conversations/{conversation_id}/chat",
            json={"question": "comparison", "mode": "react"},
        )

    body = response.json()
    assert body["answer"] == "pipeline fallback"
    assert [event["stage"] for event in body["trace"]] == [
        "agent_tool_observation",
        "react_fallback",
    ]


def test_old_chat_contract_is_removed_and_client_state_is_forbidden(
    tmp_path: Path,
) -> None:
    runtime = make_runtime(make_database(tmp_path))
    app = create_app(lambda: runtime)

    with TestClient(app) as client:
        old_route = client.post("/chat", json={"question": "question"})
        conversation_id = create_client_conversation(client)
        old_fields = client.post(
            f"/conversations/{conversation_id}/chat",
            json={
                "question": "question",
                "conversation_history": [],
                "active_evidence_ids": [],
            },
        )

    assert old_route.status_code == 404
    assert old_fields.status_code == 422
    error_fields = {
        tuple(error["loc"])[-1] for error in old_fields.json()["detail"]
    }
    assert error_fields == {"conversation_history", "active_evidence_ids"}


def test_chat_rejects_missing_but_allows_long_conversation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = make_database(tmp_path)
    runtime = make_runtime(database_path)
    conversation = create_conversation(database_path)
    for index in range(105):
        append_conversation_turn(
            database_path,
            conversation.conversation_id,
            user_message=f"question {index}",
            assistant_message="answer",
        )
    monkeypatch.setattr(
        chat_service,
        "run_rag",
        lambda question, **kwargs: RagResult(
            question=question,
            papers=(),
            answer="answer",
            generation_error=None,
        ),
    )
    app = create_app(lambda: runtime)

    with TestClient(app) as client:
        missing = client.post(
            "/conversations/missing/chat",
            json={"question": "question"},
        )
        long_chat = client.post(
            f"/conversations/{conversation.conversation_id}/chat",
            json={"question": "one too many"},
        )
        missing_stream = client.post(
            "/conversations/missing/chat/stream",
            json={"question": "question"},
        )
        long_stream = client.post(
            f"/conversations/{conversation.conversation_id}/chat/stream",
            json={"question": "one too many"},
        )

    assert missing.status_code == 404
    assert long_chat.status_code == 200
    assert missing_stream.status_code == 404
    assert long_stream.status_code == 200


def test_generation_failure_does_not_enter_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = make_runtime(make_database(tmp_path))
    monkeypatch.setattr(
        chat_service,
        "run_rag",
        lambda question, **kwargs: RagResult(
            question=question,
            papers=(),
            answer=None,
            generation_error="provider unavailable",
            trace=(),
        ),
    )
    app = create_app(lambda: runtime)

    with TestClient(app) as client:
        conversation_id = create_client_conversation(client)
        response = client.post(
            f"/conversations/{conversation_id}/chat",
            json={"question": "question"},
        )
        stored = client.get(f"/conversations/{conversation_id}")

    assert response.status_code == 200
    assert response.json()["answer"] is None
    assert stored.json()["turn_count"] == 0


def test_direct_response_is_stored_without_clearing_active_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = make_database(tmp_path)
    runtime = make_runtime(database_path)
    conversation = create_conversation(database_path)
    append_conversation_turn(
        database_path,
        conversation.conversation_id,
        user_message="research question",
        assistant_message="research answer",
        paper_ids=("2607.00001",),
    )
    monkeypatch.setattr(
        chat_service,
        "run_rag",
        lambda question, **kwargs: RagResult(
            question=question,
            papers=(),
            answer="不客气。",
            generation_error=None,
            trace=(),
            response_kind="conversation",
        ),
    )
    app = create_app(lambda: runtime)

    with TestClient(app) as client:
        response = client.post(
            f"/conversations/{conversation.conversation_id}/chat",
            json={"question": "谢谢"},
        )
        stored = client.get(f"/conversations/{conversation.conversation_id}")

    state = load_conversation_state(database_path, conversation.conversation_id)
    assert response.json()["response_kind"] == "conversation"
    assert stored.json()["turns"][-1]["response_kind"] == "conversation"
    assert stored.json()["turns"][-1]["paper_ids"] == []
    assert state.active_evidence_ids == ("2607.00001",)
