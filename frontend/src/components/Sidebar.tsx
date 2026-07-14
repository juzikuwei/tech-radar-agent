import { MAX_HISTORY_TURNS, MAX_STORED_TURNS } from "../constants";
import type { ChatMode, ConversationSummary, KnowledgeBaseStats } from "../types";


interface SidebarProps {
  stats: KnowledgeBaseStats | null;
  statsError: string | null;
  conversations: ConversationSummary[];
  activeConversationId: string | null;
  activeTurnCount: number;
  managingConversations: boolean;
  onNewConversation: () => void;
  onSelectConversation: (conversationId: string) => void;
  onDeleteConversation: (conversationId: string) => void;
  mode: ChatMode;
  onModeChange: (mode: ChatMode) => void;
}


export function Sidebar({
  stats,
  statsError,
  conversations,
  activeConversationId,
  activeTurnCount,
  managingConversations,
  onNewConversation,
  onSelectConversation,
  onDeleteConversation,
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
          onClick={onNewConversation}
          disabled={managingConversations}
        >
          <span aria-hidden="true">＋</span>
          新对话
        </button>

        <section className="conversation-sidebar-section" aria-label="会话列表">
          <p className="sidebar-section-label">会话</p>
          <div className="conversation-list">
            {conversations.map((conversation) => (
              <div
                className={
                  conversation.conversation_id === activeConversationId
                    ? "conversation-list-item active"
                    : "conversation-list-item"
                }
                key={conversation.conversation_id}
              >
                <button
                  type="button"
                  className="conversation-select"
                  onClick={() => onSelectConversation(conversation.conversation_id)}
                  disabled={managingConversations}
                >
                  <strong>{conversation.title}</strong>
                  <span>{conversation.turn_count} 轮</span>
                </button>
                <button
                  type="button"
                  className="conversation-delete"
                  aria-label={`删除会话 ${conversation.title}`}
                  onClick={() => onDeleteConversation(conversation.conversation_id)}
                  disabled={managingConversations}
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        </section>

        <div className="mode-picker" role="group" aria-label="回答模式">
          <button
            type="button"
            className={mode === "react" ? "active" : ""}
            aria-pressed={mode === "react"}
            onClick={() => onModeChange("react")}
          >
            <strong>研究 Agent</strong>
            <span>自主选择工具，最多调用 5 次</span>
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
      </div>

      <div className="sidebar-footer">
        <p className="sidebar-section-label">知识库与会话状态</p>
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
            label="已存"
            value={`${activeTurnCount}/${MAX_STORED_TURNS}`}
          />
          <StatusCard
            label="模型窗口"
            value={`${Math.min(activeTurnCount, MAX_HISTORY_TURNS)}/${MAX_HISTORY_TURNS}`}
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
