import type {
  ChatRequest,
  ChatResponse,
  ChatStreamEvent,
  Conversation,
  ConversationSummary,
  KnowledgeBaseStats,
  TraceEvent,
} from "./types";

const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";

export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ?? DEFAULT_API_BASE_URL
).replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number | null = null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export async function getKnowledgeBaseStats(): Promise<KnowledgeBaseStats> {
  return requestJson<KnowledgeBaseStats>("/knowledge-base/stats");
}

export async function listConversations(): Promise<ConversationSummary[]> {
  return requestJson<ConversationSummary[]>("/conversations");
}

export async function createConversation(): Promise<ConversationSummary> {
  return requestJson<ConversationSummary>("/conversations", {
    method: "POST",
  });
}

export async function getConversation(
  conversationId: string,
): Promise<Conversation> {
  return requestJson<Conversation>(
    `/conversations/${encodeURIComponent(conversationId)}`,
  );
}

export async function deleteConversation(conversationId: string): Promise<void> {
  await requestVoid(
    `/conversations/${encodeURIComponent(conversationId)}`,
    { method: "DELETE" },
  );
}

export async function sendChat(
  conversationId: string,
  payload: ChatRequest,
): Promise<ChatResponse> {
  return requestJson<ChatResponse>(
    `/conversations/${encodeURIComponent(conversationId)}/chat`,
    {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    },
  );
}

export async function sendChatStream(
  conversationId: string,
  payload: ChatRequest,
  onTrace: (event: TraceEvent) => void,
  signal?: AbortSignal,
): Promise<ChatResponse> {
  let response: Response;
  try {
    response = await fetch(
      `${API_BASE_URL}/conversations/${encodeURIComponent(conversationId)}/chat/stream`,
      {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal,
      },
    );
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw error;
    }
    throw new ApiError(
      `无法连接 FastAPI：请确认后端已启动（${API_BASE_URL}）`,
    );
  }

  if (!response.ok) {
    throw new ApiError(await readErrorMessage(response), response.status);
  }
  if (!response.body) {
    throw new ApiError("浏览器没有收到可读取的响应流。", response.status);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result: ChatResponse | null = null;

  while (true) {
    const chunk = await reader.read();
    buffer += decoder.decode(chunk.value, { stream: !chunk.done });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      result = consumeStreamLine(line, onTrace, result);
    }
    if (chunk.done) {
      break;
    }
  }

  if (buffer.trim()) {
    result = consumeStreamLine(buffer, onTrace, result);
  }
  if (!result) {
    throw new ApiError("响应流提前结束，没有收到最终回答。", response.status);
  }
  return result;
}

function consumeStreamLine(
  line: string,
  onTrace: (event: TraceEvent) => void,
  currentResult: ChatResponse | null,
): ChatResponse | null {
  if (!line.trim()) {
    return currentResult;
  }

  let event: ChatStreamEvent;
  try {
    event = JSON.parse(line) as ChatStreamEvent;
  } catch {
    throw new ApiError("后端返回了无法解析的 Trace 数据。");
  }

  if (event.type === "trace") {
    onTrace(event.event);
    return currentResult;
  }
  if (event.type === "result") {
    return event.result;
  }
  if (event.type === "error") {
    throw new ApiError(event.message);
  }
  return currentResult;
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init);
  } catch {
    throw new ApiError(
      `无法连接 FastAPI：请确认后端已启动（${API_BASE_URL}）`,
    );
  }

  if (!response.ok) {
    throw new ApiError(await readErrorMessage(response), response.status);
  }
  return (await response.json()) as T;
}

async function requestVoid(path: string, init?: RequestInit): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init);
  } catch {
    throw new ApiError(
      `无法连接 FastAPI：请确认后端已启动（${API_BASE_URL}）`,
    );
  }
  if (!response.ok) {
    throw new ApiError(await readErrorMessage(response), response.status);
  }
}

async function readErrorMessage(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") {
      return body.detail;
    }
    if (body.detail !== undefined) {
      return JSON.stringify(body.detail);
    }
  } catch {
    // Fall through to a stable status-based message.
  }
  return `请求失败（HTTP ${response.status}）`;
}
