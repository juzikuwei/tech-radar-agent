import type { ChatMode } from "../types";


export function MainHeader({ mode }: { mode: ChatMode }) {
  return (
    <header className="main-header">
      <div className="header-title">
        <span className="mobile-brand-mark" aria-hidden="true">P</span>
        <div>
          <strong>论文研究助手</strong>
          <span>基于本地 arXiv 知识库</span>
        </div>
      </div>
      <div className="header-badge">
        <span className="header-status-dot" />
        {mode === "react" ? "研究 Agent · 证据可追溯" : "可靠管线 · 证据可追溯"}
      </div>
    </header>
  );
}
