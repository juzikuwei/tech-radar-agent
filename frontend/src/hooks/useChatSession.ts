import { useCallback, useEffect, useReducer, useRef } from "react";

import {
  createConversation,
  deleteConversation,
  getConversation,
  listConversations,
  sendChatStream,
} from "../api";
import { DEFAULT_TOP_K } from "../constants";
import type {
  ChatMode,
  ChatResponse,
  CompletedTurn,
  Conversation,
  ConversationSummary,
  ModelUsage,
  TraceEvent,
} from "../types";

const INITIAL_LIVE_STATUS = "请求已接收，正在启动…";
const STREAM_CONTINUES_NOTICE = "上一条回答将在后台继续生成，稍后可回到原会话查看。";

type StreamPhase = "idle" | "active" | "failed";

interface ChatSessionState {
  mode: ChatMode;
  conversations: ConversationSummary[];
  activeConversationId: string | null;
  turns: CompletedTurn[];
  draft: string;
  streamPhase: StreamPhase;
  pendingQuestion: string | null;
  liveTrace: TraceEvent[];
  liveAnswer: string;
  liveStatus: string;
  liveUsage: ModelUsage | null;
  requestError: string | null;
  failedResult: ChatResponse | null;
  loadingConversation: boolean;
  managingConversations: boolean;
  initializationFailed: boolean;
  backgroundNotice: string | null;
}

type ChatSessionAction =
  | { type: "SET_MODE"; mode: ChatMode }
  | { type: "SET_DRAFT"; draft: string }
  | { type: "SET_CONVERSATIONS"; conversations: ConversationSummary[] }
  | { type: "CANCEL_LIVE"; notice: string | null }
  | { type: "LOAD_START"; conversationId: string }
  | { type: "LOAD_SUCCESS"; conversation: Conversation }
  | { type: "LOAD_FAILURE"; message: string }
  | { type: "LOAD_SETTLED" }
  | { type: "MANAGE_START" }
  | { type: "MANAGE_FAILURE"; message: string }
  | { type: "MANAGE_SETTLED" }
  | { type: "CONVERSATION_CREATED"; conversation: ConversationSummary }
  | { type: "INIT_FAILURE"; message: string }
  | { type: "INIT_RETRY" }
  | { type: "STREAM_START"; question: string }
  | { type: "STREAM_TRACE"; event: TraceEvent }
  | { type: "STREAM_STATUS"; message: string }
  | { type: "STREAM_DELTA"; delta: string }
  | { type: "STREAM_ANSWER"; content: string; usage: ModelUsage | null }
  | { type: "STREAM_SUCCESS"; turn: CompletedTurn }
  | { type: "STREAM_SOFT_FAILURE"; result: ChatResponse; message: string }
  | { type: "STREAM_FAILURE"; message: string };

const LIVE_RESET = {
  streamPhase: "idle" as StreamPhase,
  pendingQuestion: null,
  liveTrace: [] as TraceEvent[],
  liveAnswer: "",
  liveStatus: INITIAL_LIVE_STATUS,
  liveUsage: null,
} satisfies Partial<ChatSessionState>;

const INITIAL_STATE: ChatSessionState = {
  mode: "react",
  conversations: [],
  activeConversationId: null,
  turns: [],
  draft: "",
  ...LIVE_RESET,
  requestError: null,
  failedResult: null,
  loadingConversation: true,
  managingConversations: false,
  initializationFailed: false,
  backgroundNotice: null,
};

function toCompletedTurns(conversation: Conversation): CompletedTurn[] {
  return conversation.turns.map((turn) => ({
    id: String(turn.turn_id),
    question: turn.user_message,
    answer: turn.assistant_message,
    papers: turn.papers,
    responseKind: turn.response_kind,
    result: null,
  }));
}

function reducer(
  state: ChatSessionState,
  action: ChatSessionAction,
): ChatSessionState {
  switch (action.type) {
    case "SET_MODE":
      return { ...state, mode: action.mode };
    case "SET_DRAFT":
      return { ...state, draft: action.draft };
    case "SET_CONVERSATIONS":
      return { ...state, conversations: action.conversations };
    case "CANCEL_LIVE":
      // 统一的“断流即清理”：无论后续 API 成败，输入框都不会被卡死。
      return { ...state, ...LIVE_RESET, backgroundNotice: action.notice };
    case "LOAD_START":
      return {
        ...state,
        ...LIVE_RESET,
        activeConversationId: action.conversationId,
        loadingConversation: true,
        requestError: null,
        failedResult: null,
      };
    case "LOAD_SUCCESS":
      return {
        ...state,
        ...LIVE_RESET,
        activeConversationId: action.conversation.conversation_id,
        turns: toCompletedTurns(action.conversation),
        draft: "",
        requestError: null,
        failedResult: null,
      };
    case "LOAD_FAILURE":
      return { ...state, turns: [], requestError: action.message };
    case "LOAD_SETTLED":
      return { ...state, loadingConversation: false };
    case "MANAGE_START":
      return { ...state, managingConversations: true };
    case "MANAGE_FAILURE":
      return { ...state, requestError: action.message };
    case "MANAGE_SETTLED":
      return { ...state, managingConversations: false };
    case "CONVERSATION_CREATED":
      return {
        ...state,
        ...LIVE_RESET,
        activeConversationId: action.conversation.conversation_id,
        conversations: [
          action.conversation,
          ...state.conversations.filter(
            (conversation) =>
              conversation.conversation_id !== action.conversation.conversation_id,
          ),
        ],
        turns: [],
        draft: "",
        requestError: null,
        failedResult: null,
        loadingConversation: false,
      };
    case "INIT_FAILURE":
      return {
        ...state,
        loadingConversation: false,
        requestError: action.message,
        initializationFailed: true,
      };
    case "INIT_RETRY":
      return {
        ...state,
        loadingConversation: true,
        requestError: null,
        initializationFailed: false,
      };
    case "STREAM_START":
      return {
        ...state,
        ...LIVE_RESET,
        streamPhase: "active",
        pendingQuestion: action.question,
        draft: "",
        requestError: null,
        failedResult: null,
        backgroundNotice: null,
      };
    case "STREAM_TRACE": {
      const isToolStart =
        action.event.stage === "tool" && action.event.status === "started";
      return {
        ...state,
        liveTrace: [...state.liveTrace, action.event],
        liveAnswer: isToolStart ? "" : state.liveAnswer,
        liveUsage: isToolStart ? null : state.liveUsage,
      };
    }
    case "STREAM_STATUS":
      return { ...state, liveStatus: action.message };
    case "STREAM_DELTA":
      return { ...state, liveAnswer: state.liveAnswer + action.delta };
    case "STREAM_ANSWER":
      return { ...state, liveAnswer: action.content, liveUsage: action.usage };
    case "STREAM_SUCCESS":
      return {
        ...state,
        ...LIVE_RESET,
        turns: [...state.turns, action.turn],
      };
    case "STREAM_SOFT_FAILURE":
      // 回答未通过生成/校验：检索证据保留在 failedResult 中展示。
      return {
        ...state,
        ...LIVE_RESET,
        failedResult: action.result,
        requestError: action.message,
      };
    case "STREAM_FAILURE":
      // 硬失败：保留已流出的部分回答与 Trace，与错误卡并存展示；
      // 下一次提问或切换会话时再清理。
      return {
        ...state,
        streamPhase: "failed",
        requestError: action.message,
      };
    default:
      return state;
  }
}

export function useChatSession() {
  const [state, dispatch] = useReducer(reducer, INITIAL_STATE);
  const activeRequest = useRef<AbortController | null>(null);
  const activeConversationIdRef = useRef<string | null>(null);
  const loadSequence = useRef(0);
  const initialization = useRef<Promise<void> | null>(null);
  const turnCounter = useRef(0);

  /** abort 当前 SSE 并同步清理 pending/live 状态；notice 仅在确有活跃流时展示。 */
  const cancelActiveStream = useCallback((notice: string | null = null) => {
    const controller = activeRequest.current;
    activeRequest.current = null;
    controller?.abort();
    dispatch({ type: "CANCEL_LIVE", notice: controller ? notice : null });
  }, []);

  const loadConversation = useCallback(async (conversationId: string) => {
    if (conversationId === activeConversationIdRef.current) {
      // 点击当前已激活的会话：不打断进行中的流，也不清空状态。
      return;
    }
    const sequence = ++loadSequence.current;
    cancelActiveStream(STREAM_CONTINUES_NOTICE);
    activeConversationIdRef.current = conversationId;
    dispatch({ type: "LOAD_START", conversationId });
    try {
      const conversation = await getConversation(conversationId);
      if (loadSequence.current === sequence) {
        dispatch({ type: "LOAD_SUCCESS", conversation });
      }
    } catch (error) {
      if (loadSequence.current === sequence) {
        dispatch({
          type: "LOAD_FAILURE",
          message:
            error instanceof Error ? error.message : "加载会话失败，请稍后重试。",
        });
      }
    } finally {
      if (loadSequence.current === sequence) {
        dispatch({ type: "LOAD_SETTLED" });
      }
    }
  }, [cancelActiveStream]);

  const initialize = useCallback(async () => {
    try {
      let available = await listConversations();
      if (!available.length) {
        available = [await createConversation()];
      }
      dispatch({ type: "SET_CONVERSATIONS", conversations: available });
      await loadConversation(available[0].conversation_id);
    } catch (error) {
      initialization.current = null;
      dispatch({
        type: "INIT_FAILURE",
        message:
          error instanceof Error ? error.message : "初始化会话失败，请稍后重试。",
      });
    }
  }, [loadConversation]);

  useEffect(() => {
    if (initialization.current) {
      return;
    }
    initialization.current = initialize();
  }, [initialize]);

  const retryInitialization = useCallback(() => {
    if (initialization.current) {
      return;
    }
    dispatch({ type: "INIT_RETRY" });
    initialization.current = initialize();
  }, [initialize]);

  async function startNewConversation() {
    dispatch({ type: "MANAGE_START" });
    cancelActiveStream(STREAM_CONTINUES_NOTICE);
    try {
      const created = await createConversation();
      loadSequence.current += 1;
      activeConversationIdRef.current = created.conversation_id;
      dispatch({ type: "CONVERSATION_CREATED", conversation: created });
    } catch (error) {
      dispatch({
        type: "MANAGE_FAILURE",
        message:
          error instanceof Error ? error.message : "新建会话失败，请稍后重试。",
      });
    } finally {
      dispatch({ type: "MANAGE_SETTLED" });
    }
  }

  async function removeConversation(conversationId: string) {
    dispatch({ type: "MANAGE_START" });
    if (activeConversationIdRef.current === conversationId) {
      cancelActiveStream(null);
    }
    try {
      await deleteConversation(conversationId);
      const remaining = state.conversations.filter(
        (conversation) => conversation.conversation_id !== conversationId,
      );
      dispatch({ type: "SET_CONVERSATIONS", conversations: remaining });
      if (activeConversationIdRef.current === conversationId) {
        if (remaining.length) {
          await loadConversation(remaining[0].conversation_id);
        } else {
          await startNewConversation();
        }
      }
    } catch (error) {
      dispatch({
        type: "MANAGE_FAILURE",
        message:
          error instanceof Error ? error.message : "删除会话失败，请稍后重试。",
      });
    } finally {
      dispatch({ type: "MANAGE_SETTLED" });
    }
  }

  async function submitQuestion(questionInput: string) {
    const question = questionInput.trim();
    const conversationId = activeConversationIdRef.current;
    if (
      !question
      || state.streamPhase === "active"
      || !conversationId
      || state.loadingConversation
    ) {
      return;
    }

    dispatch({ type: "STREAM_START", question });
    const controller = new AbortController();
    activeRequest.current = controller;
    try {
      const result = await sendChatStream(
        conversationId,
        { question, top_k: DEFAULT_TOP_K, mode: state.mode },
        {
          onTrace: (event) => {
            if (activeConversationIdRef.current === conversationId) {
              dispatch({ type: "STREAM_TRACE", event });
            }
          },
          onStatus: (message) => {
            if (activeConversationIdRef.current === conversationId) {
              dispatch({ type: "STREAM_STATUS", message });
            }
          },
          onAssistantDelta: (delta) => {
            if (activeConversationIdRef.current === conversationId) {
              dispatch({ type: "STREAM_DELTA", delta });
            }
          },
          onAssistantCompleted: (content, usage) => {
            if (activeConversationIdRef.current === conversationId) {
              dispatch({ type: "STREAM_ANSWER", content, usage });
            }
          },
        },
        controller.signal,
      );
      if (activeConversationIdRef.current !== conversationId) {
        return;
      }
      if (!result.answer) {
        dispatch({
          type: "STREAM_SOFT_FAILURE",
          result,
          message: result.generation_error ?? "回答生成失败，但检索证据仍可查看。",
        });
        return;
      }

      dispatch({
        type: "STREAM_SUCCESS",
        turn: {
          id: `local-${++turnCounter.current}`,
          question,
          answer: result.answer,
          papers: result.papers,
          responseKind: result.response_kind,
          result,
        },
      });
      try {
        dispatch({
          type: "SET_CONVERSATIONS",
          conversations: await listConversations(),
        });
      } catch {
        // The answer is already persisted; a refresh can recover list metadata.
      }
    } catch (error) {
      if (controller.signal.aborted) {
        return;
      }
      if (activeConversationIdRef.current !== conversationId) {
        return;
      }
      dispatch({
        type: "STREAM_FAILURE",
        message:
          error instanceof Error ? error.message : "请求失败，请稍后重试。",
      });
    } finally {
      if (activeRequest.current === controller) {
        activeRequest.current = null;
      }
    }
  }

  const activeConversation = state.conversations.find(
    (conversation) => conversation.conversation_id === state.activeConversationId,
  ) ?? null;

  return {
    conversations: state.conversations,
    activeConversation,
    activeConversationId: state.activeConversationId,
    turns: state.turns,
    mode: state.mode,
    setMode: (mode: ChatMode) => dispatch({ type: "SET_MODE", mode }),
    draft: state.draft,
    setDraft: (draft: string) => dispatch({ type: "SET_DRAFT", draft }),
    pendingQuestion: state.pendingQuestion,
    streaming: state.streamPhase === "active",
    liveTrace: state.liveTrace,
    liveAnswer: state.liveAnswer,
    liveStatus: state.liveStatus,
    liveUsage: state.liveUsage,
    requestError: state.requestError,
    failedResult: state.failedResult,
    loadingConversation: state.loadingConversation,
    managingConversations: state.managingConversations,
    initializationFailed: state.initializationFailed,
    backgroundNotice: state.backgroundNotice,
    retryInitialization,
    submitQuestion,
    startNewConversation,
    selectConversation: loadConversation,
    removeConversation,
    isReady: Boolean(state.activeConversationId) && !state.loadingConversation,
  };
}
