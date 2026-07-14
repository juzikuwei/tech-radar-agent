import { useCallback, useEffect, useRef, useState } from "react";

import {
  createConversation,
  deleteConversation,
  getConversation,
  listConversations,
  sendChatStream,
} from "../api";
import type {
  ChatMode,
  ChatResponse,
  CompletedTurn,
  Conversation,
  ConversationSummary,
  ModelUsage,
  TraceEvent,
} from "../types";


export function useChatSession() {
  const [mode, setMode] = useState<ChatMode>("react");
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [turns, setTurns] = useState<CompletedTurn[]>([]);
  const [draft, setDraft] = useState("");
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);
  const [liveTrace, setLiveTrace] = useState<TraceEvent[]>([]);
  const [liveAnswer, setLiveAnswer] = useState("");
  const [liveStatus, setLiveStatus] = useState("请求已接收，正在启动…");
  const [liveUsage, setLiveUsage] = useState<ModelUsage | null>(null);
  const [requestError, setRequestError] = useState<string | null>(null);
  const [failedResult, setFailedResult] = useState<ChatResponse | null>(null);
  const [loadingConversation, setLoadingConversation] = useState(true);
  const [managingConversations, setManagingConversations] = useState(false);
  const activeRequest = useRef<AbortController | null>(null);
  const activeConversationIdRef = useRef<string | null>(null);
  const loadSequence = useRef(0);
  const initialization = useRef<Promise<void> | null>(null);

  const showConversation = useCallback((conversation: Conversation) => {
    activeConversationIdRef.current = conversation.conversation_id;
    setActiveConversationId(conversation.conversation_id);
    setTurns(conversation.turns.map((turn) => ({
      id: String(turn.turn_id),
      question: turn.user_message,
      answer: turn.assistant_message,
      papers: turn.papers,
      responseKind: turn.response_kind,
      result: null,
    })));
    setDraft("");
    setPendingQuestion(null);
    setLiveTrace([]);
    setLiveAnswer("");
    setLiveStatus("请求已接收，正在启动…");
    setLiveUsage(null);
    setRequestError(null);
    setFailedResult(null);
  }, []);

  const loadConversation = useCallback(async (conversationId: string) => {
    const sequence = ++loadSequence.current;
    activeRequest.current?.abort();
    activeRequest.current = null;
    activeConversationIdRef.current = conversationId;
    setActiveConversationId(conversationId);
    setLoadingConversation(true);
    setRequestError(null);
    setFailedResult(null);
    setPendingQuestion(null);
    setLiveTrace([]);
    setLiveAnswer("");
    setLiveStatus("请求已接收，正在启动…");
    setLiveUsage(null);
    try {
      const conversation = await getConversation(conversationId);
      if (loadSequence.current === sequence) {
        showConversation(conversation);
      }
    } catch (error) {
      if (loadSequence.current === sequence) {
        setTurns([]);
        setRequestError(
          error instanceof Error ? error.message : "加载会话失败，请稍后重试。",
        );
      }
    } finally {
      if (loadSequence.current === sequence) {
        setLoadingConversation(false);
      }
    }
  }, [showConversation]);

  useEffect(() => {
    if (initialization.current) {
      return;
    }
    initialization.current = (async () => {
      try {
        let available = await listConversations();
        if (!available.length) {
          available = [await createConversation()];
        }
        setConversations(available);
        await loadConversation(available[0].conversation_id);
      } catch (error) {
        setLoadingConversation(false);
        setRequestError(
          error instanceof Error ? error.message : "初始化会话失败，请稍后重试。",
        );
      }
    })();
  }, [loadConversation]);

  async function startNewConversation() {
    setManagingConversations(true);
    activeRequest.current?.abort();
    activeRequest.current = null;
    try {
      const created = await createConversation();
      loadSequence.current += 1;
      activeConversationIdRef.current = created.conversation_id;
      setActiveConversationId(created.conversation_id);
      setConversations((current) => [
        created,
        ...current.filter(
          (conversation) => conversation.conversation_id !== created.conversation_id,
        ),
      ]);
      setTurns([]);
      setDraft("");
      setPendingQuestion(null);
      setLiveTrace([]);
      setLiveAnswer("");
      setLiveStatus("请求已接收，正在启动…");
      setLiveUsage(null);
      setRequestError(null);
      setFailedResult(null);
      setLoadingConversation(false);
    } catch (error) {
      setRequestError(
        error instanceof Error ? error.message : "新建会话失败，请稍后重试。",
      );
    } finally {
      setManagingConversations(false);
    }
  }

  async function removeConversation(conversationId: string) {
    setManagingConversations(true);
    if (activeConversationIdRef.current === conversationId) {
      activeRequest.current?.abort();
      activeRequest.current = null;
    }
    try {
      await deleteConversation(conversationId);
      const remaining = conversations.filter(
        (conversation) => conversation.conversation_id !== conversationId,
      );
      setConversations(remaining);
      if (activeConversationIdRef.current === conversationId) {
        if (remaining.length) {
          await loadConversation(remaining[0].conversation_id);
        } else {
          await startNewConversation();
        }
      }
    } catch (error) {
      setRequestError(
        error instanceof Error ? error.message : "删除会话失败，请稍后重试。",
      );
    } finally {
      setManagingConversations(false);
    }
  }

  async function submitQuestion(questionInput: string) {
    const question = questionInput.trim();
    const conversationId = activeConversationIdRef.current;
    if (!question || pendingQuestion || !conversationId || loadingConversation) {
      return;
    }

    setDraft("");
    setPendingQuestion(question);
    setLiveTrace([]);
    setLiveAnswer("");
    setLiveStatus("请求已接收，正在启动…");
    setLiveUsage(null);
    setRequestError(null);
    setFailedResult(null);
    const controller = new AbortController();
    activeRequest.current = controller;
    try {
      const result = await sendChatStream(
        conversationId,
        { question, top_k: 5, mode },
        {
          onTrace: (event) => {
            if (activeConversationIdRef.current === conversationId) {
              if (event.stage === "tool" && event.status === "started") {
                setLiveAnswer("");
                setLiveUsage(null);
              }
              setLiveTrace((current) => [...current, event]);
            }
          },
          onStatus: (message) => {
            if (activeConversationIdRef.current === conversationId) {
              setLiveStatus(message);
            }
          },
          onAssistantDelta: (delta) => {
            if (activeConversationIdRef.current === conversationId) {
              setLiveAnswer((current) => current + delta);
            }
          },
          onAssistantCompleted: (content, usage) => {
            if (activeConversationIdRef.current === conversationId) {
              setLiveAnswer(content);
              setLiveUsage(usage);
            }
          },
        },
        controller.signal,
      );
      if (activeConversationIdRef.current !== conversationId) {
        return;
      }
      if (!result.answer) {
        setFailedResult(result);
        setRequestError(
          result.generation_error ?? "回答生成失败，但检索证据仍可查看。",
        );
        return;
      }

      setTurns((current) => [
        ...current,
        {
          id: `${Date.now()}-${current.length}`,
          question,
          answer: result.answer as string,
          papers: result.papers,
          responseKind: result.response_kind,
          result,
        },
      ]);
      try {
        setConversations(await listConversations());
      } catch {
        // The answer is already persisted; a refresh can recover list metadata.
      }
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
        setLiveAnswer("");
        setLiveStatus("请求已接收，正在启动…");
        setLiveUsage(null);
      }
    }
  }

  const activeConversation = conversations.find(
    (conversation) => conversation.conversation_id === activeConversationId,
  ) ?? null;

  return {
    conversations,
    activeConversation,
    activeConversationId,
    turns,
    mode,
    setMode,
    draft,
    setDraft,
    pendingQuestion,
    liveTrace,
    liveAnswer,
    liveStatus,
    liveUsage,
    requestError,
    failedResult,
    loadingConversation,
    managingConversations,
    submitQuestion,
    startNewConversation,
    selectConversation: loadConversation,
    removeConversation,
    isReady: Boolean(activeConversationId) && !loadingConversation,
  };
}
