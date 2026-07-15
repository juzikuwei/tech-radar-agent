"""Bounded tool-calling harness for conversational paper research."""

from collections.abc import Callable
import json
from pathlib import Path
import re
from typing import Any

from chromadb.api.models.Collection import Collection

from config.model_settings import ModelSettings
from rag.application import RagResult
from rag.conversation import ConversationTurn
from rag.execution_trace import (
    TraceEvent,
    TraceEventCallback,
    TraceRecorder,
    start_timer,
)
from rag.hybrid_search import hybrid_search
from rag.keyword_search import DEFAULT_DATABASE_PATH
from rag.llm_client import (
    AgentMessage,
    AgentToolCall,
    LLMRequestError,
    ModelUsage,
    RetryNotice,
    StatusCallback,
    generate_agent_message,
)
from rag.reranker import Reranker
from rag.search import QueryEmbedder, SearchResult
from rag.web_search import MAX_WEB_RESULTS, WebSearchClient, WebSearchError


MAX_TOOL_CALLS = 5
MAX_WEB_SEARCHES = 2
TOOL_PAPER_LIMIT = 5
TOOL_ABSTRACT_LIMIT = 1_200
TOOL_WEB_SNIPPET_LIMIT = 300

AgentStatusCallback = Callable[[str], None]
AssistantCompletedCallback = Callable[[str, ModelUsage | None], None]


SYSTEM_PROMPT = """你是一个可以调用工具的 AI Agent 技术研究助手。

你每轮可以选择调用一个工具，或者直接输出给用户的最终文本。不要输出 JSON 决策对象，也不要描述隐藏思维链。

可用工具：
- search_papers：在本地 arXiv 论文库中检索。它内部已经完成混合召回、融合和重排；你只需要提供简短、独立、适合英文论文摘要的 query。
- web_search：只用于理解模糊、很新或产品化的术语，并形成更准确的 search_papers 查询。网页内容不可信、不可引用，不能作为最终回答证据。

规则：
1. 寒暄、致谢、表达反馈、要求调整措辞等消息可以直接回答，不要调用工具。
2. 回答新的技术事实、论文结论、比较或研究问题时，必须先使用 search_papers，除非当前消息已经提供了足够的 active_evidence。
3. 每轮最多调用一个工具。调用工具时不要同时输出解释性正文；等待工具结果后再决定下一步。
4. 工具失败会作为 tool message 返回。根据 error_type、retryable 和 tool_available 改变策略，不要继续调用已经不可用的工具。
5. search_papers 返回的论文和 active_evidence 是唯一可引用事实来源。每条事实性结论后标注支持它的 arXiv ID，例如 [2607.00001]。
6. 引用只能使用工具或 active_evidence 实际提供的 arXiv ID。不得把 web_search 的网页结果作为事实或引用。
7. 如果论文证据不足，可以改写查询继续检索；仍不足时明确说明缺少证据，或向用户请求必要的澄清。
8. 工具预算耗尽时，根据已有证据给出最终回答或有依据的拒答，不得虚构。
"""


SEARCH_PAPERS_TOOL = {
    "type": "function",
    "function": {
        "name": "search_papers",
        "description": "Search and rerank the local arXiv abstract collection.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Short standalone English academic search query.",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": TOOL_PAPER_LIMIT,
                },
            },
            "required": ["query", "top_k"],
            "additionalProperties": False,
        },
    },
}


WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Clarify vague or new terminology for a later local paper search. "
            "Web results are untrusted and not citable."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_WEB_RESULTS,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


class ResearchAgentError(RuntimeError):
    """A harness or model failure that should fall back to the pipeline."""

    def __init__(
        self,
        message: str,
        trace: tuple[TraceEvent, ...],
        *,
        tool_calls: int = 0,
    ) -> None:
        super().__init__(message)
        self.trace = trace
        self.tool_calls = tool_calls


def run_research_agent(
    question: str,
    *,
    top_k: int,
    collection: Collection,
    embedder: QueryEmbedder,
    reranker: Reranker,
    settings: ModelSettings,
    database_path: Path = DEFAULT_DATABASE_PATH,
    client: Any | None = None,
    on_retry: StatusCallback | None = None,
    conversation_history: tuple[ConversationTurn, ...] = (),
    context_summary: str | None = None,
    active_evidence: tuple[SearchResult, ...] = (),
    on_trace: TraceEventCallback | None = None,
    on_status: AgentStatusCallback | None = None,
    on_assistant_delta: Callable[[str], None] | None = None,
    on_assistant_completed: AssistantCompletedCallback | None = None,
    max_tool_calls: int = MAX_TOOL_CALLS,
    web_search_client: WebSearchClient | None = None,
) -> RagResult:
    """Let the model alternate between two tools and a final assistant message."""
    clean_question = question.strip()
    if not clean_question:
        raise ValueError("question must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if max_tool_calls <= 0:
        raise ValueError("max_tool_calls must be greater than zero")

    trace = TraceRecorder(on_event=on_trace)
    messages = _initial_messages(
        clean_question,
        history=tuple(conversation_history),
        context_summary=context_summary,
        active_evidence=active_evidence,
    )
    evidence_by_id = {paper.arxiv_id: paper for paper in active_evidence}
    tool_call_count = 0
    search_count = 0
    web_search_count = 0
    web_search_enabled = web_search_client is not None
    total_usage = ModelUsage()
    has_usage = False

    while True:
        final_only = tool_call_count >= max_tool_calls
        available_tools = [] if final_only else [SEARCH_PAPERS_TOOL]
        if (
            not final_only
            and web_search_enabled
            and web_search_count < MAX_WEB_SEARCHES
        ):
            available_tools.append(WEB_SEARCH_TOOL)
        available_tool_names = {
            tool["function"]["name"] for tool in available_tools
        }

        status_message = (
            "工具调用预算已用完，正在整理最终回答…"
            if final_only
            else "模型正在判断是否需要调用工具…"
        )
        _emit_status(on_status, status_message)
        model_started_at = start_timer()
        trace.record(
            stage="model",
            label="模型开始生成下一步",
            status="started",
            details={
                "tool_call_count": tool_call_count,
                "available_tools": [
                    tool["function"]["name"] for tool in available_tools
                ],
            },
        )

        def handle_retry(notice: RetryNotice) -> None:
            trace.record(
                stage="model",
                label="模型请求重试中",
                status="retrying",
                details={
                    "next_attempt": notice.next_attempt,
                    "max_attempts": notice.max_attempts,
                    "wait_seconds": notice.wait_seconds,
                    "reason": notice.reason,
                },
            )
            _emit_status(
                on_status,
                f"模型请求失败，{notice.wait_seconds:g} 秒后重试…",
            )
            if on_retry is not None:
                on_retry(notice)

        try:
            assistant = generate_agent_message(
                messages,
                settings=settings,
                tools=available_tools,
                client=client,
                on_retry=handle_retry,
                on_delta=on_assistant_delta,
            )
        except LLMRequestError as error:
            trace.record(
                stage="model",
                label="模型生成失败",
                status="failed",
                started_at=model_started_at,
                details={"error": str(error)},
            )
            raise ResearchAgentError(
                str(error),
                trace.events,
                tool_calls=tool_call_count,
            ) from error

        if assistant.usage is not None:
            total_usage = total_usage + assistant.usage
            has_usage = True
        trace.record(
            stage="model",
            label="模型完成本轮输出",
            status="completed",
            started_at=model_started_at,
            details={
                "finish_reason": assistant.finish_reason,
                "tool_calls": [call.name for call in assistant.tool_calls],
                "answer_char_count": len(assistant.content),
                "usage": _usage_payload(assistant.usage),
            },
        )
        messages.append(_assistant_message(assistant))

        if not assistant.tool_calls:
            answer = assistant.content.strip()
            if not answer:
                raise ResearchAgentError(
                    "Model returned no final answer",
                    trace.events,
                    tool_calls=tool_call_count,
                )
            final_papers = _final_papers(
                answer,
                evidence=tuple(evidence_by_id.values()),
                top_k=top_k,
            )
            final_usage = total_usage if has_usage else None
            if on_assistant_completed is not None:
                on_assistant_completed(answer, assistant.usage)
            _emit_status(on_status, "回答生成完成。")
            return RagResult(
                question=clean_question,
                papers=final_papers,
                answer=answer,
                generation_error=None,
                retrieval_attempts=search_count,
                standalone_question=clean_question,
                trace=trace.events,
                response_kind=(
                    "research"
                    if search_count > 0 or final_papers
                    else "conversation"
                ),
                usage=final_usage,
            )

        if final_only:
            raise ResearchAgentError(
                "Model requested a tool after the tool budget was exhausted",
                trace.events,
                tool_calls=tool_call_count,
            )

        for call_index, call in enumerate(assistant.tool_calls):
            if call_index > 0:
                payload = {
                    "ok": False,
                    "error": "only one tool call is allowed per assistant turn",
                    "error_type": "parallel_tool_call_rejected",
                    "retryable": True,
                    "tool_available": True,
                }
                trace.record(
                    stage="tool",
                    label="额外工具调用已拒绝",
                    status="failed",
                    details={
                        "tool": call.name,
                        "arguments": call.arguments,
                        "output": payload,
                    },
                )
                messages.append(_tool_message(call, payload))
                continue
            if tool_call_count >= max_tool_calls:
                messages.append(
                    _tool_message(
                        call,
                        {
                            "ok": False,
                            "error": "tool call budget exhausted",
                            "error_type": "budget",
                            "retryable": False,
                            "tool_available": False,
                        },
                    )
                )
                continue
            tool_call_count += 1
            if call.name not in available_tool_names:
                payload = {
                    "ok": False,
                    "error": f"tool is not available in this turn: {call.name}",
                    "error_type": "tool_unavailable",
                    "retryable": False,
                    "tool_available": False,
                }
                trace.record(
                    stage="tool",
                    label="不可用工具调用已拒绝",
                    status="failed",
                    details={
                        "tool": call.name,
                        "arguments": call.arguments,
                        "output": payload,
                    },
                )
                messages.append(_tool_message(call, payload))
                continue
            if call.name == "search_papers":
                papers = _execute_paper_search(
                    call,
                    collection=collection,
                    embedder=embedder,
                    reranker=reranker,
                    database_path=database_path,
                    default_top_k=top_k,
                    trace=trace,
                    on_status=on_status,
                )
                search_count += 1
                if isinstance(papers, tuple):
                    for paper in papers:
                        evidence_by_id.setdefault(paper.arxiv_id, paper)
                    messages.append(
                        _tool_message(call, _paper_tool_payload(papers))
                    )
                else:
                    messages.append(_tool_message(call, papers))
            elif call.name == "web_search":
                web_search_count += 1
                payload, keep_available = _execute_web_search(
                    call,
                    web_search_client=web_search_client,
                    retry_available=(
                        web_search_count < MAX_WEB_SEARCHES
                        and tool_call_count < max_tool_calls
                    ),
                    trace=trace,
                    on_status=on_status,
                )
                web_search_enabled = keep_available
                messages.append(_tool_message(call, payload))
            else:
                payload = {
                    "ok": False,
                    "error": f"unknown tool: {call.name}",
                    "error_type": "unknown_tool",
                    "retryable": False,
                    "tool_available": False,
                }
                trace.record(
                    stage="tool",
                    label="未知工具调用失败",
                    status="failed",
                    details={"tool": call.name, "output": payload},
                )
                messages.append(_tool_message(call, payload))


def _initial_messages(
    question: str,
    *,
    history: tuple[ConversationTurn, ...],
    context_summary: str | None,
    active_evidence: tuple[SearchResult, ...],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    for turn in history:
        messages.extend(
            [
                {"role": "user", "content": turn.user_message},
                {"role": "assistant", "content": turn.assistant_message},
            ]
        )
    active_payload = [
        {
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "abstract": paper.document[:TOOL_ABSTRACT_LIMIT],
        }
        for paper in active_evidence[:TOOL_PAPER_LIMIT]
    ]
    messages.append(
        {
            "role": "user",
            "content": (
                (
                    "<conversation_summary>\n"
                    f"{context_summary}\n"
                    "</conversation_summary>\n\n"
                    if context_summary
                    else ""
                )
                + f"<current_question>\n{question}\n</current_question>\n\n"
                "<active_evidence>\n"
                f"{json.dumps(active_payload, ensure_ascii=False)}\n"
                "</active_evidence>"
            ),
        }
    )
    return messages


def _assistant_message(message: AgentMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": "assistant",
        "content": message.content or None,
    }
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.call_id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.raw_arguments,
                },
            }
            for call in message.tool_calls
        ]
    return payload


def _tool_message(
    call: AgentToolCall,
    payload: dict[str, object],
) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call.call_id,
        "name": call.name,
        "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    }


def _execute_paper_search(
    call: AgentToolCall,
    *,
    collection: Collection,
    embedder: QueryEmbedder,
    reranker: Reranker,
    database_path: Path,
    default_top_k: int,
    trace: TraceRecorder,
    on_status: AgentStatusCallback | None,
) -> tuple[SearchResult, ...] | dict[str, object]:
    query = _string_argument(call, "query")
    requested_top_k = _integer_argument(call, "top_k")
    if not query:
        return _invalid_tool_arguments(call, "query must be a non-empty string", trace)
    if requested_top_k is None:
        requested_top_k = min(default_top_k, TOOL_PAPER_LIMIT)
    if not 1 <= requested_top_k <= TOOL_PAPER_LIMIT:
        return _invalid_tool_arguments(call, "top_k must be between 1 and 5", trace)

    arguments = {"query": query, "top_k": requested_top_k}
    _emit_status(on_status, f"正在检索论文：{query}")
    started_at = start_timer()
    trace.record(
        stage="tool",
        label="即将执行论文检索工具",
        status="started",
        details={"tool": "search_papers", "arguments": arguments},
    )
    try:
        papers = tuple(
            hybrid_search(
                query,
                top_k=requested_top_k,
                collection=collection,
                embedder=embedder,
                reranker=reranker,
                database_path=database_path,
                trace=TraceRecorder(),
                retrieval_round=1,
            )
        )
    except Exception as error:
        payload = {
            "ok": False,
            "error": str(error),
            "error_type": "tool_execution",
            "retryable": False,
            "tool_available": True,
        }
        trace.record(
            stage="tool",
            label="论文检索工具执行失败",
            status="failed",
            started_at=started_at,
            details={
                "tool": "search_papers",
                "arguments": arguments,
                "output": payload,
            },
        )
        _emit_status(on_status, "论文检索失败，模型正在调整策略…")
        return payload

    output = {
        "result_count": len(papers),
        "papers": [
            {"arxiv_id": paper.arxiv_id, "title": paper.title}
            for paper in papers
        ],
    }
    trace.record(
        stage="tool",
        label="论文检索工具执行完成",
        status="completed",
        started_at=started_at,
        details={
            "tool": "search_papers",
            "arguments": arguments,
            "output": output,
        },
    )
    return papers


def _execute_web_search(
    call: AgentToolCall,
    *,
    web_search_client: WebSearchClient | None,
    retry_available: bool,
    trace: TraceRecorder,
    on_status: AgentStatusCallback | None,
) -> tuple[dict[str, object], bool]:
    query = _string_argument(call, "query")
    requested_count = _integer_argument(call, "max_results") or MAX_WEB_RESULTS
    if not query:
        payload = _invalid_tool_arguments(
            call,
            "query must be a non-empty string",
            trace,
        )
        return payload, False
    if not 1 <= requested_count <= MAX_WEB_RESULTS:
        payload = _invalid_tool_arguments(
            call,
            f"max_results must be between 1 and {MAX_WEB_RESULTS}",
            trace,
        )
        return payload, False

    arguments = {"query": query, "max_results": requested_count}
    _emit_status(on_status, f"正在查询网页术语：{query}")
    started_at = start_timer()
    trace.record(
        stage="tool",
        label="即将执行网页搜索工具",
        status="started",
        details={"tool": "web_search", "arguments": arguments},
    )
    if web_search_client is None:
        error: Exception = WebSearchError(
            "web search tool is not configured",
            error_type="configuration",
        )
    else:
        try:
            results = web_search_client.search(query, max_results=requested_count)
        except Exception as caught_error:
            error = caught_error
        else:
            payload = {
                "ok": True,
                "result_count": len(results),
                "results": [
                    {
                        "title": item.title,
                        "snippet": item.snippet[:TOOL_WEB_SNIPPET_LIMIT],
                    }
                    for item in results
                ],
                "untrusted": True,
                "citable": False,
            }
            trace.record(
                stage="tool",
                label="网页搜索工具执行完成",
                status="completed",
                started_at=started_at,
                details={
                    "tool": "web_search",
                    "arguments": arguments,
                    "output": {
                        "result_count": len(results),
                        "titles": [item.title for item in results],
                    },
                },
            )
            return payload, True

    if isinstance(error, WebSearchError):
        error_type = error.error_type
        retryable = error.retryable
    elif isinstance(error, ValueError):
        error_type = "invalid_request"
        retryable = False
    else:
        error_type = "tool_execution"
        retryable = False
    keep_available = retryable and retry_available
    payload = {
        "ok": False,
        "error": str(error),
        "error_type": error_type,
        "retryable": retryable,
        "tool_available": keep_available,
    }
    trace.record(
        stage="tool",
        label="网页搜索工具执行失败",
        status="failed",
        started_at=started_at,
        details={
            "tool": "web_search",
            "arguments": arguments,
            "output": payload,
        },
    )
    _emit_status(
        on_status,
        "网页搜索失败，正在重新选择工具…",
    )
    return payload, keep_available


def _invalid_tool_arguments(
    call: AgentToolCall,
    message: str,
    trace: TraceRecorder,
) -> dict[str, object]:
    payload = {
        "ok": False,
        "error": message,
        "error_type": "invalid_arguments",
        "retryable": False,
        "tool_available": True,
    }
    trace.record(
        stage="tool",
        label="工具参数校验失败",
        status="failed",
        details={
            "tool": call.name,
            "arguments": call.arguments,
            "output": payload,
        },
    )
    return payload


def _paper_tool_payload(papers: tuple[SearchResult, ...]) -> dict[str, object]:
    return {
        "ok": True,
        "result_count": len(papers),
        "papers": [
            {
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "abstract": paper.document[:TOOL_ABSTRACT_LIMIT],
            }
            for paper in papers[:TOOL_PAPER_LIMIT]
        ],
    }


def _string_argument(call: AgentToolCall, key: str) -> str | None:
    if call.arguments is None:
        return None
    value = call.arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _integer_argument(call: AgentToolCall, key: str) -> int | None:
    if call.arguments is None:
        return None
    value = call.arguments.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    return value


def _final_papers(
    answer: str,
    *,
    evidence: tuple[SearchResult, ...],
    top_k: int,
) -> tuple[SearchResult, ...]:
    by_id = {paper.arxiv_id: paper for paper in evidence}
    cited_ids = [
        candidate
        for group in re.findall(r"\[([^\]]+)\]", answer)
        for candidate in re.split(r"[,，\s]+", group.strip())
        if candidate in by_id
    ]
    selected: list[SearchResult] = []
    selected_ids: set[str] = set()
    for paper_id in cited_ids:
        if paper_id not in selected_ids:
            selected.append(by_id[paper_id])
            selected_ids.add(paper_id)
    if selected:
        return tuple(selected)
    ranked = sorted(
        evidence,
        key=lambda paper: (-float(paper.rerank_score or 0.0), paper.arxiv_id),
    )
    return tuple(ranked[:top_k])


def _usage_payload(usage: ModelUsage | None) -> dict[str, int] | None:
    if usage is None:
        return None
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }


def _emit_status(
    callback: AgentStatusCallback | None,
    message: str,
) -> None:
    if callback is not None:
        callback(message)
