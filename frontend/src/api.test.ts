import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, sendChatStream } from "./api";
import type { ChatRequest, ChatResponse } from "./types";


const payload: ChatRequest = { question: "q", top_k: 5, mode: "react" };

const chatResult: ChatResponse = {
  question: "q",
  answer: "最终回答",
  generation_error: null,
  papers: [],
  trace: [],
  retrieval_attempts: 0,
  standalone_question: null,
  conversation_decision: null,
  response_kind: "research",
  mode: "react",
  fallback_used: false,
  usage: null,
};

const resultEvent = `data: ${JSON.stringify({ type: "result", result: chatResult })}\n\n`;


function makeHandlers() {
  return {
    onTrace: vi.fn(),
    onStatus: vi.fn(),
    onAssistantDelta: vi.fn(),
    onAssistantCompleted: vi.fn(),
  };
}

function sseResponse(chunks: string[]): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk));
      }
      controller.close();
    },
  });
  return new Response(stream, { status: 200 });
}

function stubFetchWith(chunks: string[]) {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(sseResponse(chunks)));
}

interface FakeReader {
  read: () => Promise<ReadableStreamReadResult<Uint8Array>>;
  cancel: ReturnType<typeof vi.fn>;
}

function stubFetchWithReader(reader: FakeReader) {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      body: { getReader: () => reader },
    }),
  );
}


afterEach(() => {
  vi.unstubAllGlobals();
});


describe("sendChatStream SSE parsing", () => {
  it("handles multiple events inside a single chunk", async () => {
    const handlers = makeHandlers();
    stubFetchWith([
      'data: {"type":"status","message":"启动中"}\n\n'
      + 'data: {"type":"assistant_delta","delta":"你好"}\n\n'
      + resultEvent,
    ]);

    const result = await sendChatStream("c1", payload, handlers);

    expect(handlers.onStatus).toHaveBeenCalledWith("启动中");
    expect(handlers.onAssistantDelta).toHaveBeenCalledWith("你好");
    expect(result).toEqual(chatResult);
  });

  it("reassembles an event split across chunk boundaries", async () => {
    const handlers = makeHandlers();
    stubFetchWith([
      'data: {"type":"sta',
      'tus","message":"分块状态"}\n',
      "\n",
      resultEvent,
    ]);

    const result = await sendChatStream("c1", payload, handlers);

    expect(handlers.onStatus).toHaveBeenCalledWith("分块状态");
    expect(result).toEqual(chatResult);
  });

  it("accepts CRLF line endings", async () => {
    const handlers = makeHandlers();
    stubFetchWith([
      'data: {"type":"assistant_delta","delta":"甲"}\r\n\r\n'
      + `data: ${JSON.stringify({ type: "result", result: chatResult })}\r\n\r\n`,
    ]);

    const result = await sendChatStream("c1", payload, handlers);

    expect(handlers.onAssistantDelta).toHaveBeenCalledWith("甲");
    expect(result).toEqual(chatResult);
  });

  it("ignores SSE comment lines such as heartbeat pings", async () => {
    const handlers = makeHandlers();
    stubFetchWith([
      ": ping\n\n",
      ': ping\ndata: {"type":"status","message":"心跳后仍解析"}\n\n',
      ": 注释与数据同块\n" + resultEvent,
    ]);

    const result = await sendChatStream("c1", payload, handlers);

    expect(handlers.onStatus).toHaveBeenCalledWith("心跳后仍解析");
    expect(result).toEqual(chatResult);
  });

  it("throws the backend message when an error event arrives mid-stream", async () => {
    const handlers = makeHandlers();
    stubFetchWith([
      'data: {"type":"assistant_delta","delta":"部分"}\n\n'
      + 'data: {"type":"error","message":"后端执行失败"}\n\n',
    ]);

    await expect(sendChatStream("c1", payload, handlers)).rejects.toThrow(
      "后端执行失败",
    );
    expect(handlers.onAssistantDelta).toHaveBeenCalledWith("部分");
  });

  it("fails with a Chinese error when the stream ends without a result", async () => {
    stubFetchWith(['data: {"type":"status","message":"进行中"}\n\n']);

    await expect(sendChatStream("c1", payload, makeHandlers())).rejects.toThrow(
      "响应流提前结束，没有收到最终回答。",
    );
  });

  it("re-throws aborts untouched and cancels the reader", async () => {
    const controller = new AbortController();
    const cancel = vi.fn().mockResolvedValue(undefined);
    stubFetchWithReader({
      read: () =>
        new Promise((_, reject) => {
          const fail = () =>
            reject(new DOMException("aborted", "AbortError"));
          if (controller.signal.aborted) {
            fail();
          } else {
            controller.signal.addEventListener("abort", fail);
          }
        }),
      cancel,
    });

    const pending = sendChatStream(
      "c1",
      payload,
      makeHandlers(),
      controller.signal,
    );
    const assertion = expect(pending).rejects.toMatchObject({
      name: "AbortError",
    });
    controller.abort();
    await assertion;
    expect(cancel).toHaveBeenCalled();
  });

  it("wraps low-level read failures into a Chinese ApiError", async () => {
    const cancel = vi.fn().mockResolvedValue(undefined);
    stubFetchWithReader({
      read: () => Promise.reject(new TypeError("network error")),
      cancel,
    });

    const pending = sendChatStream("c1", payload, makeHandlers());
    await expect(pending).rejects.toBeInstanceOf(ApiError);
    await expect(pending).rejects.toThrow(
      "读取响应流时连接中断，请检查网络或后端服务后重试。",
    );
    expect(cancel).toHaveBeenCalled();
  });
});
