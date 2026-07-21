from types import SimpleNamespace
from datetime import datetime
from pathlib import Path
import sqlite3

from eval.memory_adapter import _copy_sqlite_database
from eval.run_eval import default_output_path
from eval.runner import run_agent_cases, run_answer_cases, run_memory_cases, run_retrieval_cases
from eval.schemas import (
    AgentCase,
    AgentExpectation,
    AgentTurn,
    AnswerCase,
    MemoryCase,
    RetrievalCase,
)


def test_retrieval_runner_records_order_and_latency() -> None:
    case = RetrievalCase("q", "query", True, ("a",), {"a": 3})

    report = run_retrieval_cases(
        (case,),
        search=lambda query, top_k: (
            SimpleNamespace(arxiv_id="a", similarity=0.9, rerank_score=5.0),
        ),
        top_k=5,
    )

    assert report["suite"] == "retrieval"
    assert report["summary"]["answerable"]["mean_hit_at_k"] == 1.0
    assert report["cases"][0]["duration_ms"] >= 0


def test_agent_runner_updates_history_and_active_evidence() -> None:
    case = AgentCase(
        "a",
        (
            AgentTurn("first"),
            AgentTurn(
                "follow-up",
                AgentExpectation(("answer_from_existing",), 0, 0, "required"),
            ),
        ),
    )
    seen = []

    def run_turn(question, history, evidence):
        seen.append((question, len(history), len(evidence)))
        return SimpleNamespace(
            question=question,
            answer="answer",
            papers=(SimpleNamespace(arxiv_id="a"),),
            retrieval_attempts=0,
            conversation_decision=SimpleNamespace(
                next_action="answer_from_existing",
                reusable_arxiv_ids=("a",),
            ),
            trace=(),
        )

    report = run_agent_cases((case,), run_turn=run_turn)

    assert report["summary"]["passed"] == 1
    assert seen == [("first", 0, 0), ("follow-up", 1, 1)]


def test_answer_and_memory_runners_return_pass_rates() -> None:
    answer_case = AnswerCase(
        "a", "question", True, expected_citations=("0000.00001",)
    )
    answer_result = SimpleNamespace(
        answer="answer [0000.00001]",
        papers=(SimpleNamespace(arxiv_id="0000.00001"),),
    )
    answer_report = run_answer_cases(
        (answer_case,),
        run_turn=lambda question, history, evidence: answer_result,
    )
    memory_case = MemoryCase("m", ("first",), ("constraint",), min_compactions=1)
    memory_report = run_memory_cases(
        (memory_case,),
        run_memory=lambda case: ("constraint", 1),
    )

    assert answer_report["summary"]["pass_rate"] == 1.0
    assert memory_report["summary"]["pass_rate"] == 1.0


def test_default_output_path_separates_suites_by_name_and_timestamp() -> None:
    moment = datetime(2026, 7, 21, 15, 30, 0)

    retrieval = default_output_path("retrieval", now=moment)
    memory = default_output_path("memory", now=moment)

    assert retrieval == Path("eval/results/retrieval_20260721_153000.json")
    assert memory == Path("eval/results/memory_20260721_153000.json")
    assert retrieval != memory


def test_memory_adapter_copies_sqlite_without_mutating_source(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE values_table (value TEXT NOT NULL)")
        connection.execute("INSERT INTO values_table VALUES ('source')")
        connection.commit()

    _copy_sqlite_database(source, target)
    with sqlite3.connect(target) as connection:
        connection.execute("INSERT INTO values_table VALUES ('target')")
        connection.commit()
    with sqlite3.connect(source) as connection:
        values = connection.execute("SELECT value FROM values_table").fetchall()

    assert values == [("source",)]
