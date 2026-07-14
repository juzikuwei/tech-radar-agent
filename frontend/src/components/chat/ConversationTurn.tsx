import ReactMarkdown from "react-markdown";

import { ACTION_LABELS } from "../../constants";
import type { CompletedTurn } from "../../types";
import { PaperList } from "../results/PaperList";
import { TracePanel } from "../trace/TracePanel";
import { UserBubble } from "./UserBubble";


export function ConversationTurn({
  turn,
  index,
}: {
  turn: CompletedTurn;
  index: number;
}) {
  return (
    <article className="exchange">
      <UserBubble question={turn.question} />
      <div className="assistant-row">
        <div className="assistant-avatar" aria-label="研究助手">✦</div>
        <div className="assistant-content">
          <div className="answer-meta">
            <span>研究助手 · {String(index).padStart(2, "0")}</span>
            {turn.result ? (
              <>
                <span>
                  {turn.result.response_kind === "conversation"
                    ? "直接对话回应"
                    : `${turn.result.retrieval_attempts} 次新检索`}
                </span>
                <span>
                  {turn.result.mode === "react" ? "研究 Agent" : "可靠管线"}
                  {turn.result.fallback_used ? "（已降级）" : ""}
                </span>
                {turn.result.conversation_decision ? (
                  <span>
                    {ACTION_LABELS[turn.result.conversation_decision.next_action] ??
                      turn.result.conversation_decision.next_action}
                  </span>
                ) : null}
                {turn.result.usage ? (
                  <span>{turn.result.usage.total_tokens} tokens</span>
                ) : null}
              </>
            ) : (
              <span>
                {turn.responseKind === "conversation" ? "对话回应" : "研究回答"}
                · Trace 不回放
              </span>
            )}
          </div>
          {turn.result ? (
            <div className="completed-trace">
              <TracePanel
                events={turn.result.trace}
                title={`已完成 ${turn.result.trace.length} 个执行步骤`}
              />
            </div>
          ) : null}
          <div className="markdown-answer">
            <ReactMarkdown>{turn.answer}</ReactMarkdown>
          </div>
          {turn.papers.length ? (
            <div className="result-details">
              <PaperList papers={turn.papers} />
            </div>
          ) : null}
        </div>
      </div>
    </article>
  );
}
