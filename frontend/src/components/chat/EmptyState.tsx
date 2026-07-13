import { SUGGESTIONS } from "../../constants";


export function EmptyState({
  onSuggestion,
}: {
  onSuggestion: (value: string) => void;
}) {
  return (
    <div className="empty-state">
      <div className="assistant-emblem" aria-hidden="true">
        <span>✦</span>
      </div>
      <h3>有什么可以帮你研究的？</h3>
      <p>向我询问 AI 与 Agent 技术问题，我会先检索论文证据，再组织回答。</p>
      <div className="suggestion-grid">
        {SUGGESTIONS.map((suggestion) => (
          <button
            key={suggestion}
            type="button"
            onClick={() => void onSuggestion(suggestion)}
          >
            <span className="suggestion-icon" aria-hidden="true">⌁</span>
            <span>{suggestion}</span>
            <span className="suggestion-arrow" aria-hidden="true">→</span>
          </button>
        ))}
      </div>
    </div>
  );
}
