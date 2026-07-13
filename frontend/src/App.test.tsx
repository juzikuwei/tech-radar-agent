import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import { getKnowledgeBaseStats, sendChatStream } from "./api";
import type { ChatResponse, TraceEvent } from "./types";

vi.mock("./api", () => ({
  getKnowledgeBaseStats: vi.fn(),
  sendChatStream: vi.fn(),
}));

const response: ChatResponse = {
  question: "Agentic RAG 是什么？",
  answer: "这是一个基于证据的回答 [2501.09136]。",
  generation_error: null,
  retrieval_attempts: 1,
  standalone_question: "Agentic RAG 是什么？",
  conversation_decision: null,
  mode: "react",
  fallback_used: false,
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

describe("App", () => {
  beforeEach(() => {
    vi.mocked(getKnowledgeBaseStats).mockResolvedValue({
      paper_count: 10530,
      vector_count: 10530,
    });
    vi.mocked(sendChatStream).mockResolvedValue(response);
  });

  it("submits a question and renders answer evidence and trace", async () => {
    const user = userEvent.setup();
    render(<App />);

    await waitFor(() => expect(screen.getAllByText("10,530")).toHaveLength(2));
    await user.type(screen.getByLabelText("继续追问或开始一个新技术话题"), "Agentic RAG 是什么？");
    await user.click(screen.getByRole("button", { name: "发送" }));

    expect(await screen.findByText(/这是一个基于证据的回答/)).toBeInTheDocument();
    expect(screen.getByText("Agentic Retrieval-Augmented Generation")).toBeInTheDocument();
    expect(screen.getByText("DeepSeek 最终回答生成")).toBeInTheDocument();
    expect(sendChatStream).toHaveBeenCalledWith({
      question: "Agentic RAG 是什么？",
      conversation_history: [],
      active_evidence_ids: [],
      top_k: 5,
      mode: "react",
    }, expect.any(Function), expect.any(AbortSignal));
  });

  it("clears completed conversation state", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: SUGGESTION_TEXT }));
    expect(await screen.findByText(/这是一个基于证据的回答/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "新对话" }));

    expect(screen.queryByText(/这是一个基于证据的回答/)).not.toBeInTheDocument();
    expect(screen.getByText("有什么可以帮你研究的？")).toBeInTheDocument();
  });

  it("renders trace stages before the final answer arrives", async () => {
    const user = userEvent.setup();
    let finishRequest: ((value: ChatResponse) => void) | undefined;
    const pendingResult = new Promise<ChatResponse>((resolve) => {
      finishRequest = resolve;
    });
    vi.mocked(sendChatStream).mockImplementation(async (_payload, onTrace) => {
      onTrace(response.trace[0] as TraceEvent);
      return pendingResult;
    });
    render(<App />);

    await user.type(
      screen.getByLabelText("继续追问或开始一个新技术话题"),
      "显示执行过程",
    );
    await user.click(screen.getByRole("button", { name: "发送" }));

    expect(await screen.findByText("DeepSeek 最终回答生成")).toBeInTheDocument();
    expect(screen.queryByText(/这是一个基于证据的回答/)).not.toBeInTheDocument();

    finishRequest?.(response);
    expect(await screen.findByText(/这是一个基于证据的回答/)).toBeInTheDocument();
  });

  it("allows switching back to the reliable pipeline", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: /可靠管线/ }));
    await user.type(
      screen.getByLabelText("继续追问或开始一个新技术话题"),
      "使用固定流程",
    );
    await user.click(screen.getByRole("button", { name: "发送" }));

    await screen.findByText(/这是一个基于证据的回答/);
    expect(sendChatStream).toHaveBeenCalledWith(
      expect.objectContaining({ mode: "pipeline" }),
      expect.any(Function),
      expect.any(AbortSignal),
    );
  });
});

const SUGGESTION_TEXT = "Agentic RAG 和普通 RAG 有什么区别？";
