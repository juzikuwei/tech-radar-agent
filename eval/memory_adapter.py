"""Production adapter for long-conversation compaction evaluations."""

import atexit
from dataclasses import replace
import gc
from pathlib import Path
import shutil
import sqlite3
from tempfile import mkdtemp
from time import sleep
from typing import Literal

from api.schemas import ChatRequest
from api.services.chat import execute_chat
from config.conversation_context_settings import ConversationContextSettings
from eval.runner import MemoryFn
from eval.schemas import MemoryCase
from rag.conversation_store import (
    create_conversation,
    initialize_conversation_store,
    load_conversation_state,
)
from rag.runtime import RagRuntime


def _copy_sqlite_database(source: Path, target: Path) -> None:
    """Create a transactionally consistent SQLite copy for an eval case."""
    source_connection = sqlite3.connect(source)
    target_connection = sqlite3.connect(target)
    try:
        source_connection.backup(target_connection)
    finally:
        target_connection.close()
        source_connection.close()


def _remove_temporary_directory(directory: Path) -> None:
    """Best-effort cleanup for SQLite handles released late on Windows."""
    gc.collect()
    for delay_seconds in (0.05, 0.2, 0.5):
        try:
            shutil.rmtree(directory)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            sleep(delay_seconds)
            gc.collect()
    atexit.register(shutil.rmtree, directory, ignore_errors=True)


def build_memory_runner(
    runtime: RagRuntime,
    *,
    mode: Literal["pipeline", "react"],
    top_k: int,
    token_threshold: int = 500,
    target_tokens: int = 220,
) -> MemoryFn:
    """Run each memory case in an isolated copy of the production database."""
    context_settings = ConversationContextSettings(
        token_threshold=token_threshold,
        target_tokens=target_tokens,
    )

    def run_memory(case: MemoryCase) -> tuple[str | None, int]:
        directory = Path(mkdtemp(prefix="tech-radar-memory-eval-"))
        try:
            database_path = directory / "eval.db"
            _copy_sqlite_database(runtime.database_path, database_path)
            initialize_conversation_store(database_path)
            conversation = create_conversation(database_path)
            eval_runtime = replace(
                runtime,
                database_path=database_path,
                web_search_client=None,
                conversation_context_settings=context_settings,
            )
            compaction_count = 0

            def observe_trace(event: object) -> None:
                nonlocal compaction_count
                if (
                    getattr(event, "stage", None) == "conversation_compaction"
                    and getattr(event, "status", None) == "completed"
                ):
                    compaction_count += 1

            for question in case.turns:
                execute_chat(
                    conversation.conversation_id,
                    ChatRequest(question=question, top_k=top_k, mode=mode),
                    eval_runtime,
                    on_trace=observe_trace,
                )
            state = load_conversation_state(
                database_path,
                conversation.conversation_id,
            )
            return state.context_summary, compaction_count
        finally:
            _remove_temporary_directory(directory)

    return run_memory
