from rag.execution_trace import TraceRecorder


def test_recorder_preserves_event_order_and_details() -> None:
    recorder = TraceRecorder()

    recorder.record(
        stage="dense_retrieval",
        label="Dense",
        details={"round": 1, "result_count": 5},
    )
    recorder.record(
        stage="retrieval_judgment",
        label="Judge",
        status="failed",
        details={"error": "invalid JSON"},
    )

    assert [event.stage for event in recorder.events] == [
        "dense_retrieval",
        "retrieval_judgment",
    ]
    assert recorder.events[0].details["result_count"] == 5
    assert recorder.events[1].status == "failed"


def test_events_returns_an_immutable_snapshot() -> None:
    recorder = TraceRecorder()
    first_snapshot = recorder.events

    recorder.record(stage="answer_generation", label="Answer")

    assert first_snapshot == ()
    assert len(recorder.events) == 1


def test_recorder_notifies_after_storing_each_event() -> None:
    observed = []
    recorder = TraceRecorder(on_event=observed.append)

    recorder.record(stage="dense_retrieval", label="Dense")
    recorder.record(stage="answer_generation", label="Answer")

    assert observed == list(recorder.events)
