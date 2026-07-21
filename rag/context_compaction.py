"""Token-triggered compaction for persistent conversation context."""

from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any

from config.conversation_context_settings import ConversationContextSettings
from config.model_settings import ModelSettings
from rag.conversation import ConversationTurn
from rag.conversation_store import (
    ConversationState,
    load_conversation_state,
    save_conversation_compaction,
)
from rag.llm_client import StatusCallback, generate_text


COMPACTION_MAX_OUTPUT_TOKENS = 1_200
COMPACTION_BATCH_INPUT_TOKEN_LIMIT = 8_000
_CJK_PATTERN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]"
)
_SUMMARY_FIELDS = (
    "user_goals",
    "confirmed_requirements",
    "decisions",
    "important_context",
    "open_questions",
)

SYSTEM_PROMPT = """你负责压缩一个持续对话的工作上下文，而不是回答用户问题。

输入包含旧的结构化摘要，以及一批按原始顺序排列的完整对话轮次。请把仍会影响后续对话的信息合并成新的摘要。

规则：
1. 只保留用户目标、明确要求、已经确认的决定、重要上下文和仍未解决的问题。
2. 用户原话和完整问答仍由数据库永久保存；摘要只是可重建的工作记忆，不得声称自己是原文。
3. 不要把助手以前说过的技术结论当成事实，不要总结或改写论文内容。
4. evidence_ids 只表示历史轮次关联过哪些论文；不要推断论文结论，也不要编造 ID。
5. 新信息与旧摘要冲突时，以时间更晚、用户明确确认的信息为准。
6. 删除已经完成、被撤销或不再相关的临时细节，保留尚未解决的问题。
7. 每个字段最多 20 项，每项应简短、独立、可直接用于理解后续消息。

只输出 JSON：
{
  "user_goals": ["..."],
  "confirmed_requirements": ["..."],
  "decisions": ["..."],
  "important_context": ["..."],
  "open_questions": ["..."]
}
"""


class ConversationCompactionError(RuntimeError):
    """Raised when context must be compacted but no valid summary is produced."""


class CurrentMessageTooLargeError(ConversationCompactionError):
    """The new user message alone exceeds the context budget; no amount of
    history compaction can help, so the caller should reject the request."""


@dataclass(frozen=True)
class CompactionResult:
    """Prepared context and optional metadata for one completed compaction."""

    state: ConversationState
    compacted_turn_count: int = 0
    estimated_tokens_before: int = 0
    estimated_tokens_after: int = 0

    @property
    def compacted(self) -> bool:
        return self.compacted_turn_count > 0


def prepare_conversation_context(
    database_path: Path,
    conversation_id: str,
    *,
    current_message: str = "",
    settings: ConversationContextSettings,
    model_settings: ModelSettings,
    client: Any | None = None,
    on_retry: StatusCallback | None = None,
) -> CompactionResult:
    """Compact the oldest pending turns when the history token budget is exceeded."""
    state = load_conversation_state(database_path, conversation_id)
    estimated_before = estimate_context_tokens(
        state.context_summary,
        state.uncompacted_turns,
    ) + estimate_text_tokens(current_message)
    if estimated_before <= settings.token_threshold:
        return CompactionResult(
            state=state,
            estimated_tokens_before=estimated_before,
            estimated_tokens_after=estimated_before,
        )

    current_message_tokens = estimate_text_tokens(current_message)
    if current_message_tokens > settings.token_threshold:
        raise CurrentMessageTooLargeError(
            "The current message alone exceeds the conversation context budget"
        )
    target_history_tokens = max(
        0,
        settings.target_tokens - current_message_tokens,
    )
    working_summary = state.context_summary
    remaining = state.uncompacted_turns
    compacted_turn_count = 0
    compacted_through_turn_id: int | None = None
    estimated_working = estimated_before

    while (
        estimated_working > settings.token_threshold
        or (remaining and estimated_working > settings.target_tokens)
    ):
        batch, remaining_after_batch = _select_compaction_batch(
            working_summary,
            remaining,
            target_tokens=target_history_tokens,
            batch_token_limit=COMPACTION_BATCH_INPUT_TOKEN_LIMIT,
        )
        if not batch or batch[-1].turn_id is None:
            raise ConversationCompactionError(
                "Conversation context exceeded its token budget but no complete "
                "turn was available for compaction"
            )

        content = generate_text(
            build_compaction_messages(working_summary, batch),
            settings=model_settings,
            client=client,
            on_retry=on_retry,
            response_format={"type": "json_object"},
            max_tokens=COMPACTION_MAX_OUTPUT_TOKENS,
            temperature=0.0,
        )
        next_summary = parse_compaction_summary(content)
        estimated_next = (
            estimate_context_tokens(next_summary, remaining_after_batch)
            + current_message_tokens
        )
        if estimated_next >= estimated_working:
            raise ConversationCompactionError(
                "Conversation compaction did not reduce the estimated context size"
            )
        working_summary = next_summary
        remaining = remaining_after_batch
        compacted_turn_count += len(batch)
        compacted_through_turn_id = batch[-1].turn_id
        estimated_working = estimated_next

    if compacted_through_turn_id is None or working_summary is None:
        raise ConversationCompactionError(
            "Conversation compaction did not produce a persisted boundary"
        )
    try:
        save_conversation_compaction(
            database_path,
            conversation_id,
            context_summary=working_summary,
            compacted_through_turn_id=compacted_through_turn_id,
            expected_previous_turn_id=state.compacted_through_turn_id,
        )
    except RuntimeError as error:
        raise ConversationCompactionError(
            "Conversation context changed while compaction was running"
        ) from error
    updated_state = load_conversation_state(database_path, conversation_id)
    estimated_after = estimate_context_tokens(
        updated_state.context_summary,
        updated_state.uncompacted_turns,
    ) + estimate_text_tokens(current_message)
    return CompactionResult(
        state=updated_state,
        compacted_turn_count=compacted_turn_count,
        estimated_tokens_before=estimated_before,
        estimated_tokens_after=estimated_after,
    )


def estimate_context_tokens(
    context_summary: str | None,
    turns: Sequence[ConversationTurn],
) -> int:
    """Conservatively estimate tokens without depending on a provider tokenizer."""
    total = estimate_text_tokens(context_summary or "")
    for turn in turns:
        total += 12
        total += estimate_text_tokens(turn.user_message)
        total += estimate_text_tokens(turn.assistant_message)
        total += estimate_text_tokens(json.dumps(turn.evidence_ids))
    return total


def estimate_text_tokens(text: str) -> int:
    """Estimate CJK characters individually and other UTF-8 text in four-char units."""
    if not text:
        return 0
    cjk_count = len(_CJK_PATTERN.findall(text))
    non_cjk_count = len(_CJK_PATTERN.sub("", text).encode("utf-8"))
    return cjk_count + math.ceil(non_cjk_count / 4)


def build_compaction_messages(
    context_summary: str | None,
    turns: Sequence[ConversationTurn],
) -> list[dict[str, str]]:
    """Build a summary request without loading or rewriting paper source text."""
    payload = {
        "previous_summary": (
            json.loads(context_summary) if context_summary else _empty_summary()
        ),
        "turns_to_compact": [
            {
                "user_message": turn.user_message,
                "assistant_message": turn.assistant_message,
                "evidence_ids": list(turn.evidence_ids),
            }
            for turn in turns
        ],
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, indent=2),
        },
    ]


def parse_compaction_summary(content: str) -> str:
    """Validate and normalize a structured rolling summary."""
    try:
        value = json.loads(content)
    except json.JSONDecodeError as error:
        raise ConversationCompactionError(
            "Conversation summary is not valid JSON"
        ) from error
    if not isinstance(value, dict) or set(value) != set(_SUMMARY_FIELDS):
        raise ConversationCompactionError(
            "Conversation summary has an invalid schema"
        )
    normalized: dict[str, list[str]] = {}
    for field in _SUMMARY_FIELDS:
        items = value[field]
        if not isinstance(items, list) or len(items) > 20:
            raise ConversationCompactionError(
                f"Conversation summary field {field} must be a list of at most 20 items"
            )
        clean_items: list[str] = []
        for item in items:
            if not isinstance(item, str) or not item.strip():
                raise ConversationCompactionError(
                    f"Conversation summary field {field} contains an invalid item"
                )
            clean_items.append(item.strip())
        normalized[field] = list(dict.fromkeys(clean_items))
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))


def _select_compaction_batch(
    context_summary: str | None,
    turns: Sequence[ConversationTurn],
    *,
    target_tokens: int,
    batch_token_limit: int,
) -> tuple[tuple[ConversationTurn, ...], tuple[ConversationTurn, ...]]:
    pending = list(turns)
    batch: list[ConversationTurn] = []
    while pending and estimate_context_tokens(context_summary, pending) > target_tokens:
        candidate = pending[0]
        candidate_tokens = estimate_context_tokens(
            context_summary,
            [*batch, candidate],
        )
        if candidate_tokens > batch_token_limit:
            if not batch:
                raise ConversationCompactionError(
                    "One complete historical turn is too large for a safe "
                    "compaction batch"
                )
            break
        batch.append(pending.pop(0))
    return tuple(batch), tuple(pending)


def _empty_summary() -> dict[str, list[str]]:
    return {field: [] for field in _SUMMARY_FIELDS}
