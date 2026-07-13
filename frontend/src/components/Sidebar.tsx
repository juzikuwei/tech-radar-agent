import { MAX_HISTORY_TURNS } from "../constants";
import type { ChatMode, KnowledgeBaseStats } from "../types";


interface SidebarProps {
  stats: KnowledgeBaseStats | null;
  statsError: string | null;
  turnCount: number;
  canReset: boolean;
  onReset: () => void;
  mode: ChatMode;
  onModeChange: (mode: ChatMode) => void;
}


export function Sidebar({
  stats,
  statsError,
  turnCount,
  canReset,
  onReset,
  mode,
  onModeChange,
}: SidebarProps) {
  return (
    <aside className="sidebar">
      <div className="sidebar-main">
        <div className="sidebar-brand">
          <div className="brand-mark" aria-hidden="true">P</div>
          <div>
            <strong>Paper Radar</strong>
            <span>Research Copilot</span>
          </div>
        </div>

        <button
          className="new-chat-button"
          type="button"
          onClick={onReset}
          disabled={!canReset}
        >
          <span aria-hidden="true">＋</span>
          新对话
        </button>

        <div className="mode-picker" role="group" aria-label="回答模式">
          <button
            type="button"
            className={mode === "react" ? "active" : ""}
            aria-pressed={mode === "react"}
            onClick={() => onModeChange("react")}
          >
            <strong>研究 Agent</strong>
            <span>先规划，再按证据缺口检索</span>
          </button>
          <button
            type="button"
            className={mode === "pipeline" ? "active" : ""}
            aria-pressed={mode === "pipeline"}
            onClick={() => onModeChange("pipeline")}
          >
            <strong>可靠管线</strong>
            <span>固定判断，最多两次检索</span>
          </button>
        </div>

        <div className="sidebar-note">
          <p>当前能力</p>
          <span>检索本地论文、生成引用回答，并展示完整 Agent 执行过程。</span>
        </div>
      </div>

      <div className="sidebar-footer">
        <p className="sidebar-section-label">知识库状态</p>
        <div className="status-stack" aria-label="系统状态">
          <StatusCard
            label="论文"
            value={stats ? stats.paper_count.toLocaleString("zh-CN") : "—"}
          />
          <StatusCard
            label="向量"
            value={stats ? stats.vector_count.toLocaleString("zh-CN") : "—"}
          />
          <StatusCard
            label="上下文"
            value={`${Math.min(turnCount, MAX_HISTORY_TURNS)}/${MAX_HISTORY_TURNS}`}
          />
        </div>
        <p className={statsError ? "service-state error" : "service-state"}>
          <span className="status-dot" />
          {statsError ?? "本地服务运行正常"}
        </p>
      </div>
    </aside>
  );
}


function StatusCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="status-card">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
