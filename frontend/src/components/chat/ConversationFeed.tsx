import type { RefObject } from "react";
import type {
  ChatResponse,
  CompletedTurn,
  ModelUsage,
  TraceEvent,
} from "../../types";
import { ResultDetails } from "../results/ResultDetails";
import { LiveTracePanel } from "../trace/LiveTracePanel";
import { ConversationTurn } from "./ConversationTurn";
import { CitationMarkdown } from "./CitationMarkdown";
import { EmptyState } from "./EmptyState";
import { UserBubble } from "./UserBubble";


interface ConversationFeedProps {
  turns: CompletedTurn[];
  pendingQuestion: string | null;
  streaming: boolean;
  liveTrace: TraceEvent[];
  liveAnswer: string;
  liveStatus: string;
  liveUsage: ModelUsage | null;
  requestError: string | null;
  failedResult: ChatResponse | null;
  loadingConversation: boolean;
  backgroundNotice: string | null;
  onRetryInitialization: (() => void) | null;
  conversationEndRef: RefObject<HTMLDivElement | null>;
  onSuggestion: (value: string) => void;
}


export function ConversationFeed({
  turns,
  pendingQuestion,
  streaming,
  liveTrace,
  liveAnswer,
  liveStatus,
  liveUsage,
  requestError,
  failedResult,
  loadingConversation,
  backgroundNotice,
  onRetryInitialization,
  conversationEndRef,
  onSuggestion,
}: ConversationFeedProps) {
  return (
    <section className="conversation">
      {loadingConversation ? (
        <div className="session-loading">正在加载会话…</div>
      ) : null}

      {backgroundNotice ? (
        <div className="background-notice" role="status">{backgroundNotice}</div>
      ) : null}

      {!loadingConversation && !turns.length && !pendingQuestion && !requestError ? (
        <EmptyState onSuggestion={onSuggestion} />
      ) : null}

      {turns.map((turn, index) => (
        <ConversationTurn key={turn.id} turn={turn} index={index + 1} />
      ))}

      {pendingQuestion ? (
        <PendingAnswer
          question={pendingQuestion}
          streaming={streaming}
          trace={liveTrace}
          answer={liveAnswer}
          status={liveStatus}
          usage={liveUsage}
        />
      ) : null}

      {requestError ? (
        <div className="error-card" role="alert">
          <strong>当前操作未完成</strong>
          <p>{requestError}</p>
          {onRetryInitialization ? (
            <button
              type="button"
              className="retry-button"
              onClick={onRetryInitialization}
            >
              重试
            </button>
          ) : null}
          {failedResult ? <ResultDetails result={failedResult} /> : null}
        </div>
      ) : null}
      <div ref={conversationEndRef} />
    </section>
  );
}


function PendingAnswer({
  question,
  streaming,
  trace,
  answer,
  status,
  usage,
}: {
  question: string;
  streaming: boolean;
  trace: TraceEvent[];
  answer: string;
  status: string;
  usage: ModelUsage | null;
}) {
  return (
    <>
      <UserBubble question={question} />
      <div
        className={streaming ? "assistant-row loading-row" : "assistant-row"}
        aria-busy={streaming}
      >
        <div className="assistant-avatar" role="img" aria-label="研究助手">
          <span aria-hidden="true">✦</span>
        </div>
        <div className="assistant-content">
          <LiveTracePanel events={trace} status={status} active={streaming} />
          {answer ? (
            <div className="markdown-answer live-answer">
              <CitationMarkdown content={answer} papers={[]} />
            </div>
          ) : null}
          {usage ? (
            <div className="live-usage">
              本条消息 {usage.total_tokens} tokens
            </div>
          ) : null}
        </div>
      </div>
    </>
  );
}
