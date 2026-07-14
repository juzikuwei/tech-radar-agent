import type { RefObject } from "react";

import type { ChatResponse, CompletedTurn, TraceEvent } from "../../types";
import { ResultDetails } from "../results/ResultDetails";
import { LiveTracePanel } from "../trace/LiveTracePanel";
import { ConversationTurn } from "./ConversationTurn";
import { EmptyState } from "./EmptyState";
import { UserBubble } from "./UserBubble";


interface ConversationFeedProps {
  turns: CompletedTurn[];
  pendingQuestion: string | null;
  liveTrace: TraceEvent[];
  requestError: string | null;
  failedResult: ChatResponse | null;
  loadingConversation: boolean;
  conversationEndRef: RefObject<HTMLDivElement | null>;
  onSuggestion: (value: string) => void;
}


export function ConversationFeed({
  turns,
  pendingQuestion,
  liveTrace,
  requestError,
  failedResult,
  loadingConversation,
  conversationEndRef,
  onSuggestion,
}: ConversationFeedProps) {
  return (
    <section className="conversation" aria-live="polite">
      {loadingConversation ? (
        <div className="session-loading">正在加载会话…</div>
      ) : null}

      {!loadingConversation && !turns.length && !pendingQuestion && !requestError ? (
        <EmptyState onSuggestion={onSuggestion} />
      ) : null}

      {turns.map((turn, index) => (
        <ConversationTurn key={turn.id} turn={turn} index={index + 1} />
      ))}

      {pendingQuestion ? (
        <PendingAnswer question={pendingQuestion} trace={liveTrace} />
      ) : null}

      {requestError ? (
        <div className="error-card" role="alert">
          <strong>当前操作未完成</strong>
          <p>{requestError}</p>
          {failedResult ? <ResultDetails result={failedResult} /> : null}
        </div>
      ) : null}
      <div ref={conversationEndRef} />
    </section>
  );
}


function PendingAnswer({
  question,
  trace,
}: {
  question: string;
  trace: TraceEvent[];
}) {
  return (
    <>
      <UserBubble question={question} />
      <div className="assistant-row loading-row">
        <div className="assistant-avatar" aria-label="研究助手">✦</div>
        <LiveTracePanel events={trace} />
      </div>
    </>
  );
}
