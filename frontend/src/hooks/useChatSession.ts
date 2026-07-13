import { useMemo, useRef, useState } from "react";

import { sendChatStream } from "../api";
import { MAX_HISTORY_TURNS } from "../constants";
import type { ChatMode, ChatResponse, CompletedTurn, TraceEvent } from "../types";


export function useChatSession() {
  const [mode, setMode] = useState<ChatMode>("react");
  const [turns, setTurns] = useState<CompletedTurn[]>([]);
  const [activeEvidenceIds, setActiveEvidenceIds] = useState<string[]>([]);
  const [draft, setDraft] = useState("");
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);
  const [liveTrace, setLiveTrace] = useState<TraceEvent[]>([]);
  const [requestError, setRequestError] = useState<string | null>(null);
  const [failedResult, setFailedResult] = useState<ChatResponse | null>(null);
  const activeRequest = useRef<AbortController | null>(null);

  const historyForApi = useMemo(
    () =>
      turns.slice(-MAX_HISTORY_TURNS).map((turn) => ({
        user_message: turn.question,
        assistant_message: turn.answer,
      })),
    [turns],
  );

  async function submitQuestion(questionInput: string) {
    const question = questionInput.trim();
    if (!question || pendingQuestion) {
      return;
    }

    setDraft("");
    setPendingQuestion(question);
    setLiveTrace([]);
    setRequestError(null);
    setFailedResult(null);
    const controller = new AbortController();
    activeRequest.current = controller;
    try {
      const result = await sendChatStream(
        {
          question,
          conversation_history: historyForApi,
          active_evidence_ids: activeEvidenceIds,
          top_k: 5,
          mode,
        },
        (event) => setLiveTrace((current) => [...current, event]),
        controller.signal,
      );
      if (!result.answer) {
        setFailedResult(result);
        setRequestError(
          result.generation_error ?? "回答生成失败，但检索证据仍可查看。",
        );
        return;
      }

      const answer = result.answer;
      setTurns((current) => [
        ...current,
        {
          id: `${Date.now()}-${current.length}`,
          question,
          answer,
          result,
        },
      ]);
      setActiveEvidenceIds(result.papers.map((paper) => paper.arxiv_id));
    } catch (error) {
      if (controller.signal.aborted) {
        return;
      }
      setRequestError(
        error instanceof Error ? error.message : "请求失败，请稍后重试。",
      );
    } finally {
      if (activeRequest.current === controller) {
        activeRequest.current = null;
        setPendingQuestion(null);
        setLiveTrace([]);
      }
    }
  }

  function resetConversation() {
    activeRequest.current?.abort();
    activeRequest.current = null;
    setTurns([]);
    setActiveEvidenceIds([]);
    setPendingQuestion(null);
    setLiveTrace([]);
    setRequestError(null);
    setFailedResult(null);
    setDraft("");
  }

  return {
    turns,
    mode,
    setMode,
    draft,
    setDraft,
    pendingQuestion,
    liveTrace,
    requestError,
    failedResult,
    submitQuestion,
    resetConversation,
    canReset: Boolean(turns.length || requestError || pendingQuestion),
  };
}
