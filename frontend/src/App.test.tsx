import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import {
  createConversation,
  deleteConversation,
  getConversation,
  getKnowledgeBaseStats,
  listConversations,
  sendChatStream,
} from "./api";
import type {
  ChatResponse,
  Conversation,
  ConversationSummary,
  TraceEvent,
} from "./types";

vi.mock("./api", () => ({
  createConversation: vi.fn(),
  deleteConversation: vi.fn(),
  getConversation: vi.fn(),
  getKnowledgeBaseStats: vi.fn(),
  listConversations: vi.fn(),
  sendChatStream: vi.fn(),
}));

const firstSummary: ConversationSummary = {
  conversation_id: "conversation-1",
  title: "第一个会话",
  created_at: "2026-07-13T08:00:00+00:00",
  updated_at: "2026-07-13T08:00:00+00:00",
  turn_count: 0,
};

const secondSummary: ConversationSummary = {
  conversation_id: "conversation-2",
  title: "第二个会话",
  created_at: "2026-07-13T07:00:00+00:00",
  updated_at: "2026-07-13T07:00:00+00:00",
  turn_count: 1,
};

const newSummary: ConversationSummary = {
  conversation_id: "conversation-new",
  title: "新对话",
  created_at: "2026-07-13T09:00:00+00:00",
  updated_at: "2026-07-13T09:00:00+00:00",
  turn_count: 0,
};

const response: ChatResponse = {
  question: "Agentic RAG 是什么？",
  answer: "这是一个基于证据的回答 [2501.09136]。",
  generation_error: null,
  retrieval_attempts: 1,
  standalone_question: "Agentic RAG 是什么？",
  conversation_decision: null,
  response_kind: "research",
  mode: "react",
  fallback_used: false,
  usage: null,
  papers: [
    {
      arxiv_id: "2501.09136",
      title: "Agentic Retrieval-Augmented Generation",
      document: "Title\nAbstract",
      entry_url: "https://arxiv.org/abs/2501.09136",
      primary_category: "cs.AI",
      similarity: 0.8,
      keyword_score: null,
      fusion_score: 0.7,
      rerank_score: 4.1,
    },
  ],
  trace: [
    {
      stage: "answer_generation",
      label: "DeepSeek 最终回答生成",
      status: "completed",
      duration_ms: 1200,
      details: { paper_count: 1 },
    },
  ],
};

function conversation(
  summary: ConversationSummary,
  answer?: string,
): Conversation {
  return {
    ...summary,
    turns: answer ? [
      {
        turn_id: 1,
        user_message: "已保存的问题",
        assistant_message: answer,
        paper_ids: ["2501.09136"],
        papers: response.papers.map((paper) => ({
          ...paper,
          similarity: null,
          fusion_score: null,
          rerank_score: null,
        })),
        response_kind: "research",
        created_at: "2026-07-13T08:01:00+00:00",
      },
    ] : [],
  };
}

describe("App", () => {
  beforeEach(() => {
    vi.mocked(getKnowledgeBaseStats).mockResolvedValue({
      paper_count: 10530,
      vector_count: 10530,
    });
    vi.mocked(listConversations).mockResolvedValue([firstSummary]);
    vi.mocked(getConversation).mockResolvedValue(conversation(firstSummary));
    vi.mocked(createConversation).mockResolvedValue(newSummary);
    vi.mocked(deleteConversation).mockResolvedValue();
    vi.mocked(sendChatStream).mockResolvedValue(response);
  });

  it("reloads persisted text and citations without inventing an old trace", async () => {
    vi.mocked(getConversation).mockResolvedValue(
      conversation({ ...firstSummary, turn_count: 1 }, "刷新后仍然存在的回答。"),
    );

    render(<App />);

    expect(await screen.findByText("刷新后仍然存在的回答。")).toBeInTheDocument();
    expect(screen.getByText(/Trace 不回放/)).toBeInTheDocument();
    expect(screen.getByText("Agentic Retrieval-Augmented Generation")).toBeInTheDocument();
    expect(screen.queryByText("DeepSeek 最终回答生成")).not.toBeInTheDocument();
  });

  it("submits only conversation id, question, mode, and top-k", async () => {
    const user = userEvent.setup();
    render(<App />);

    await waitFor(() => expect(screen.getByLabelText(INPUT_LABEL)).toBeEnabled());
    await user.type(screen.getByLabelText(INPUT_LABEL), "Agentic RAG 是什么？");
    await user.click(screen.getByRole("button", { name: "发送" }));

    expect(await screen.findByText(/这是一个基于证据的回答/)).toBeInTheDocument();
    expect(sendChatStream).toHaveBeenCalledWith(
      "conversation-1",
      {
        question: "Agentic RAG 是什么？",
        top_k: 5,
        mode: "react",
      },
      expect.objectContaining({
        onTrace: expect.any(Function),
        onStatus: expect.any(Function),
        onAssistantDelta: expect.any(Function),
        onAssistantCompleted: expect.any(Function),
      }),
      expect.any(AbortSignal),
    );
  });

  it("creates a separate empty conversation instead of clearing local memory", async () => {
    const user = userEvent.setup();
    vi.mocked(getConversation).mockResolvedValue(
      conversation({ ...firstSummary, turn_count: 1 }, "旧会话回答。"),
    );
    render(<App />);

    expect(await screen.findByText("旧会话回答。")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "新对话" }));

    expect(createConversation).toHaveBeenCalledTimes(1);
    expect(screen.queryByText("旧会话回答。")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /新对话0 轮/ })).toBeInTheDocument();
  });

  it("switches between server-side conversations", async () => {
    const user = userEvent.setup();
    vi.mocked(listConversations).mockResolvedValue([firstSummary, secondSummary]);
    vi.mocked(getConversation).mockImplementation(async (conversationId) => (
      conversationId === secondSummary.conversation_id
        ? conversation(secondSummary, "第二个会话的历史。")
        : conversation(firstSummary)
    ));
    render(<App />);

    await waitFor(() => expect(getConversation).toHaveBeenCalledWith("conversation-1"));
    await user.click(screen.getByRole("button", { name: /第二个会话1 轮/ }));

    expect(await screen.findByText("第二个会话的历史。")).toBeInTheDocument();
    expect(getConversation).toHaveBeenLastCalledWith("conversation-2");
  });

  it("deletes the active conversation and selects the next one", async () => {
    const user = userEvent.setup();
    vi.mocked(listConversations).mockResolvedValue([firstSummary, secondSummary]);
    vi.mocked(getConversation).mockImplementation(async (conversationId) => (
      conversationId === secondSummary.conversation_id
        ? conversation(secondSummary, "删除后选中的历史。")
        : conversation(firstSummary)
    ));
    render(<App />);

    await waitFor(() => expect(getConversation).toHaveBeenCalledWith("conversation-1"));
    await user.click(screen.getByRole("button", { name: "删除会话 第一个会话" }));

    expect(deleteConversation).toHaveBeenCalledWith("conversation-1");
    expect(await screen.findByText("删除后选中的历史。")).toBeInTheDocument();
  });

  it("renders trace stages before the final answer arrives", async () => {
    const user = userEvent.setup();
    let finishRequest: ((value: ChatResponse) => void) | undefined;
    const pendingResult = new Promise<ChatResponse>((resolve) => {
      finishRequest = resolve;
    });
    vi.mocked(sendChatStream).mockImplementation(
      async (_conversationId, _payload, handlers) => {
        handlers.onStatus("模型正在生成回答…");
        handlers.onTrace(response.trace[0] as TraceEvent);
        handlers.onAssistantDelta("实时生成的回答");
        handlers.onAssistantCompleted("实时生成的回答", {
          prompt_tokens: 20,
          completion_tokens: 5,
          total_tokens: 25,
        });
        return pendingResult;
      },
    );
    render(<App />);

    await waitFor(() => expect(screen.getByLabelText(INPUT_LABEL)).toBeEnabled());
    await user.type(screen.getByLabelText(INPUT_LABEL), "显示执行过程");
    await user.click(screen.getByRole("button", { name: "发送" }));

    expect(await screen.findByText("生成最终回答")).toBeInTheDocument();
    expect(screen.getAllByText("模型正在生成回答…")).toHaveLength(2);
    expect(screen.getByText("实时生成的回答")).toBeInTheDocument();
    expect(screen.getByText("本条消息 25 tokens")).toBeInTheDocument();
    expect(screen.queryByText(/这是一个基于证据的回答/)).not.toBeInTheDocument();

    finishRequest?.(response);
    expect(await screen.findByText(/这是一个基于证据的回答/)).toBeInTheDocument();
  });

  it("allows switching back to the reliable pipeline", async () => {
    const user = userEvent.setup();
    render(<App />);

    await waitFor(() => expect(screen.getByLabelText(INPUT_LABEL)).toBeEnabled());
    await user.click(screen.getByRole("button", { name: /可靠管线/ }));
    await user.type(screen.getByLabelText(INPUT_LABEL), "使用固定流程");
    await user.click(screen.getByRole("button", { name: "发送" }));

    await screen.findByText(/这是一个基于证据的回答/);
    expect(sendChatStream).toHaveBeenCalledWith(
      "conversation-1",
      expect.objectContaining({ mode: "pipeline" }),
      expect.objectContaining({
        onTrace: expect.any(Function),
      }),
      expect.any(AbortSignal),
    );
  });
});

const INPUT_LABEL = "继续追问或开始一个新技术话题";
