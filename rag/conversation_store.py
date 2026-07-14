"""SQLite persistence for conversations and completed turns."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Literal
from uuid import uuid4

from rag.conversation import (
    ConversationTurn,
    MAX_ACTIVE_EVIDENCE,
    MAX_CONVERSATION_TURNS,
    MAX_STORED_TURNS,
)


DEFAULT_CONVERSATION_TITLE = "新对话"
TITLE_CHAR_LIMIT = 50

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    active_evidence_ids_json TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS conversation_turns (
    turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    user_message TEXT NOT NULL,
    assistant_message TEXT NOT NULL,
    paper_ids_json TEXT NOT NULL,
    response_kind TEXT NOT NULL DEFAULT 'research',
    created_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
);

CREATE INDEX IF NOT EXISTS idx_conversation_turns_conversation
ON conversation_turns(conversation_id, turn_id);
"""


class ConversationNotFoundError(LookupError):
    """Raised when a requested conversation does not exist."""


class ConversationTurnLimitError(ValueError):
    """Raised when a conversation already contains the maximum turns."""


@dataclass(frozen=True)
class ConversationSummary:
    """List-ready metadata for one persistent conversation."""

    conversation_id: str
    title: str
    created_at: str
    updated_at: str
    turn_count: int


@dataclass(frozen=True)
class StoredConversationTurn:
    """One completed exchange stored without request-local execution trace."""

    turn_id: int
    user_message: str
    assistant_message: str
    paper_ids: tuple[str, ...]
    response_kind: Literal["research", "conversation"]
    created_at: str


@dataclass(frozen=True)
class StoredConversation:
    """One conversation and its complete persisted history."""

    summary: ConversationSummary
    turns: tuple[StoredConversationTurn, ...]


@dataclass(frozen=True)
class ConversationState:
    """Bounded state required to execute the next chat turn."""

    summary: ConversationSummary
    recent_turns: tuple[ConversationTurn, ...]
    active_evidence_ids: tuple[str, ...]


def initialize_conversation_store(database_path: Path) -> None:
    """Create conversation tables in the existing SQLite database."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        connection.executescript(SCHEMA_SQL)
        _migrate_conversation_schema(connection)
        connection.commit()


def create_conversation(database_path: Path) -> ConversationSummary:
    """Create an empty conversation with a stable UUID."""
    conversation_id = str(uuid4())
    created_at = _utc_now()
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO conversations (
                conversation_id, title, created_at, updated_at,
                active_evidence_ids_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                DEFAULT_CONVERSATION_TITLE,
                created_at,
                created_at,
                "[]",
            ),
        )
        connection.commit()
    return ConversationSummary(
        conversation_id=conversation_id,
        title=DEFAULT_CONVERSATION_TITLE,
        created_at=created_at,
        updated_at=created_at,
        turn_count=0,
    )


def list_conversations(database_path: Path) -> list[ConversationSummary]:
    """List conversations from most recently updated to least recent."""
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT conversations.conversation_id, conversations.title,
                   conversations.created_at, conversations.updated_at,
                   COUNT(conversation_turns.turn_id) AS turn_count
            FROM conversations
            LEFT JOIN conversation_turns
              ON conversation_turns.conversation_id = conversations.conversation_id
            GROUP BY conversations.conversation_id
            ORDER BY conversations.updated_at DESC,
                     conversations.conversation_id ASC
            """
        ).fetchall()
    return [_summary_from_row(row) for row in rows]


def get_conversation(
    database_path: Path,
    conversation_id: str,
) -> StoredConversation:
    """Load one conversation and every persisted turn in display order."""
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        summary_row = connection.execute(
            """
            SELECT conversations.conversation_id, conversations.title,
                   conversations.created_at, conversations.updated_at,
                   COUNT(conversation_turns.turn_id) AS turn_count
            FROM conversations
            LEFT JOIN conversation_turns
              ON conversation_turns.conversation_id = conversations.conversation_id
            WHERE conversations.conversation_id = ?
            GROUP BY conversations.conversation_id
            """,
            (conversation_id,),
        ).fetchone()
        if summary_row is None:
            raise ConversationNotFoundError(conversation_id)
        turn_rows = connection.execute(
            """
            SELECT turn_id, user_message, assistant_message,
                   paper_ids_json, response_kind, created_at
            FROM conversation_turns
            WHERE conversation_id = ?
            ORDER BY turn_id ASC
            """,
            (conversation_id,),
        ).fetchall()
    return StoredConversation(
        summary=_summary_from_row(summary_row),
        turns=tuple(_turn_from_row(row) for row in turn_rows),
    )


def load_conversation_state(
    database_path: Path,
    conversation_id: str,
) -> ConversationState:
    """Load the recent model window and latest evidence for one chat request."""
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN")
        summary_row = connection.execute(
            """
            SELECT conversations.conversation_id, conversations.title,
                   conversations.created_at, conversations.updated_at,
                   conversations.active_evidence_ids_json,
                   COUNT(conversation_turns.turn_id) AS turn_count
            FROM conversations
            LEFT JOIN conversation_turns
              ON conversation_turns.conversation_id = conversations.conversation_id
            WHERE conversations.conversation_id = ?
            GROUP BY conversations.conversation_id
            """,
            (conversation_id,),
        ).fetchone()
        if summary_row is None:
            raise ConversationNotFoundError(conversation_id)
        turn_rows = connection.execute(
            """
            SELECT user_message, assistant_message, paper_ids_json
            FROM conversation_turns
            WHERE conversation_id = ?
            ORDER BY turn_id DESC
            LIMIT ?
            """,
            (conversation_id, MAX_CONVERSATION_TURNS),
        ).fetchall()

    summary = _summary_from_row(summary_row)
    active_evidence_ids = _paper_ids_from_json(
        str(summary_row["active_evidence_ids_json"])
    )
    recent_turns = tuple(
        ConversationTurn(
            user_message=str(row["user_message"]),
            assistant_message=str(row["assistant_message"]),
            evidence_ids=_paper_ids_from_json(str(row["paper_ids_json"])),
        )
        for row in reversed(turn_rows)
    )
    return ConversationState(
        summary=summary,
        recent_turns=recent_turns,
        active_evidence_ids=active_evidence_ids[:MAX_ACTIVE_EVIDENCE],
    )


def append_conversation_turn(
    database_path: Path,
    conversation_id: str,
    *,
    user_message: str,
    assistant_message: str,
    paper_ids: tuple[str, ...] = (),
    response_kind: Literal["research", "conversation"] = "research",
    active_evidence_ids: tuple[str, ...] | None = None,
) -> StoredConversationTurn:
    """Atomically append one completed turn and update conversation metadata."""
    clean_user_message = user_message.strip()
    clean_assistant_message = assistant_message.strip()
    clean_paper_ids = tuple(dict.fromkeys(paper_id.strip() for paper_id in paper_ids))
    if active_evidence_ids is None:
        clean_active_evidence_ids = (
            clean_paper_ids[:MAX_ACTIVE_EVIDENCE]
            if response_kind == "research"
            else None
        )
    else:
        clean_active_evidence_ids = tuple(
            dict.fromkeys(paper_id.strip() for paper_id in active_evidence_ids)
        )[:MAX_ACTIVE_EVIDENCE]
    if not clean_user_message or not clean_assistant_message:
        raise ValueError("conversation messages must not be blank")
    if any(not paper_id for paper_id in clean_paper_ids):
        raise ValueError("paper_ids must contain only non-empty strings")
    if clean_active_evidence_ids is not None and any(
        not paper_id for paper_id in clean_active_evidence_ids
    ):
        raise ValueError(
            "active_evidence_ids must contain only non-empty strings"
        )
    if response_kind not in {"research", "conversation"}:
        raise ValueError("response_kind is invalid")

    created_at = _utc_now()
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN IMMEDIATE")
        conversation = connection.execute(
            """
            SELECT title, active_evidence_ids_json
            FROM conversations
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()
        if conversation is None:
            connection.rollback()
            raise ConversationNotFoundError(conversation_id)
        turn_count = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM conversation_turns
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()[0]
        )
        if turn_count >= MAX_STORED_TURNS:
            connection.rollback()
            raise ConversationTurnLimitError(conversation_id)

        cursor = connection.execute(
            """
            INSERT INTO conversation_turns (
                conversation_id, user_message, assistant_message,
                paper_ids_json, response_kind, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                clean_user_message,
                clean_assistant_message,
                json.dumps(clean_paper_ids, ensure_ascii=False),
                response_kind,
                created_at,
            ),
        )
        title = (
            _title_from_question(clean_user_message)
            if turn_count == 0
            else str(conversation["title"])
        )
        next_active_evidence_json = (
            str(conversation["active_evidence_ids_json"])
            if clean_active_evidence_ids is None
            else json.dumps(clean_active_evidence_ids, ensure_ascii=False)
        )
        connection.execute(
            """
            UPDATE conversations
            SET title = ?, updated_at = ?, active_evidence_ids_json = ?
            WHERE conversation_id = ?
            """,
            (title, created_at, next_active_evidence_json, conversation_id),
        )
        connection.commit()

    return StoredConversationTurn(
        turn_id=int(cursor.lastrowid),
        user_message=clean_user_message,
        assistant_message=clean_assistant_message,
        paper_ids=clean_paper_ids,
        response_kind=response_kind,
        created_at=created_at,
    )


def delete_conversation(database_path: Path, conversation_id: str) -> None:
    """Delete a conversation and its turns in one explicit transaction."""
    with sqlite3.connect(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        exists = connection.execute(
            "SELECT 1 FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        if exists is None:
            connection.rollback()
            raise ConversationNotFoundError(conversation_id)
        connection.execute(
            "DELETE FROM conversation_turns WHERE conversation_id = ?",
            (conversation_id,),
        )
        connection.execute(
            "DELETE FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        )
        connection.commit()


def _summary_from_row(row: sqlite3.Row) -> ConversationSummary:
    return ConversationSummary(
        conversation_id=str(row["conversation_id"]),
        title=str(row["title"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        turn_count=int(row["turn_count"]),
    )


def _turn_from_row(row: sqlite3.Row) -> StoredConversationTurn:
    response_kind = str(row["response_kind"])
    if response_kind not in {"research", "conversation"}:
        raise ValueError("stored response_kind is invalid")
    return StoredConversationTurn(
        turn_id=int(row["turn_id"]),
        user_message=str(row["user_message"]),
        assistant_message=str(row["assistant_message"]),
        paper_ids=_paper_ids_from_json(str(row["paper_ids_json"])),
        response_kind=response_kind,
        created_at=str(row["created_at"]),
    )


def _migrate_conversation_schema(connection: sqlite3.Connection) -> None:
    conversation_columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(conversations)")
    }
    if "active_evidence_ids_json" not in conversation_columns:
        connection.execute(
            """
            ALTER TABLE conversations
            ADD COLUMN active_evidence_ids_json TEXT NOT NULL DEFAULT '[]'
            """
        )
        conversation_ids = connection.execute(
            "SELECT conversation_id FROM conversations"
        ).fetchall()
        for row in conversation_ids:
            latest = connection.execute(
                """
                SELECT paper_ids_json
                FROM conversation_turns
                WHERE conversation_id = ?
                ORDER BY turn_id DESC
                LIMIT 1
                """,
                (row[0],),
            ).fetchone()
            if latest is not None:
                connection.execute(
                    """
                    UPDATE conversations
                    SET active_evidence_ids_json = ?
                    WHERE conversation_id = ?
                    """,
                    (latest[0], row[0]),
                )

    turn_columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(conversation_turns)")
    }
    if "response_kind" not in turn_columns:
        connection.execute(
            """
            ALTER TABLE conversation_turns
            ADD COLUMN response_kind TEXT NOT NULL DEFAULT 'research'
            """
        )


def _paper_ids_from_json(value: str) -> tuple[str, ...]:
    paper_ids = json.loads(value)
    if not isinstance(paper_ids, list) or not all(
        isinstance(paper_id, str) and paper_id for paper_id in paper_ids
    ):
        raise ValueError("stored paper_ids_json is invalid")
    return tuple(paper_ids)


def _title_from_question(question: str) -> str:
    return " ".join(question.split())[:TITLE_CHAR_LIMIT]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
