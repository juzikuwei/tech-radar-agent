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
            <span>{turn.result.retrieval_attempts} 次新检索</span>
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
          </div>
          <div className="completed-trace">
            <TracePanel
              events={turn.result.trace}
              title={`已完成 ${turn.result.trace.length} 个执行步骤`}
            />
          </div>
          <div className="markdown-answer">
            <ReactMarkdown>{turn.answer}</ReactMarkdown>
          </div>
          <div className="result-details">
            <PaperList papers={turn.result.papers} />
          </div>
        </div>
      </div>
    </article>
  );
}
