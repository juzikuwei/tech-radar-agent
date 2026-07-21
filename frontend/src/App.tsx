import { useEffect, useRef } from "react";

import { MainHeader } from "./components/MainHeader";
import { Sidebar } from "./components/Sidebar";
import { ChatComposer } from "./components/chat/ChatComposer";
import { ConversationFeed } from "./components/chat/ConversationFeed";
import { useChatSession } from "./hooks/useChatSession";
import { useKnowledgeBaseStats } from "./hooks/useKnowledgeBaseStats";

const STICK_TO_BOTTOM_THRESHOLD_PX = 120;


function App() {
  const { stats, error: statsError } = useKnowledgeBaseStats();
  const chat = useChatSession();
  const conversationEndRef = useRef<HTMLDivElement>(null);
  const stickToBottom = useRef(true);

  useEffect(() => {
    const handleScroll = () => {
      const root = document.documentElement;
      const distanceFromBottom =
        root.scrollHeight - window.innerHeight - window.scrollY;
      stickToBottom.current = distanceFromBottom <= STICK_TO_BOTTOM_THRESHOLD_PX;
    };
    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  useEffect(() => {
    if (!stickToBottom.current) {
      return;
    }
    const conversationEnd = conversationEndRef.current;
    if (typeof conversationEnd?.scrollIntoView !== "function") {
      return;
    }
    const prefersReducedMotion =
      typeof window.matchMedia === "function"
      && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    conversationEnd.scrollIntoView({
      behavior: prefersReducedMotion ? "auto" : "smooth",
    });
  }, [
    chat.turns,
    chat.pendingQuestion,
    chat.requestError,
    chat.liveAnswer,
    chat.liveTrace,
  ]);

  return (
    <div className="app-shell">
      <Sidebar
        stats={stats}
        statsError={statsError}
        conversations={chat.conversations}
        activeConversationId={chat.activeConversationId}
        activeTurnCount={chat.activeConversation?.turn_count ?? chat.turns.length}
        managingConversations={chat.managingConversations}
        onNewConversation={chat.startNewConversation}
        onSelectConversation={chat.selectConversation}
        onDeleteConversation={chat.removeConversation}
        mode={chat.mode}
        onModeChange={chat.setMode}
      />

      <main className="main-panel">
        <MainHeader mode={chat.mode} />
        <ConversationFeed
          turns={chat.turns}
          pendingQuestion={chat.pendingQuestion}
          streaming={chat.streaming}
          liveTrace={chat.liveTrace}
          liveAnswer={chat.liveAnswer}
          liveStatus={chat.liveStatus}
          liveUsage={chat.liveUsage}
          requestError={chat.requestError}
          failedResult={chat.failedResult}
          loadingConversation={chat.loadingConversation}
          backgroundNotice={chat.backgroundNotice}
          onRetryInitialization={
            chat.initializationFailed ? chat.retryInitialization : null
          }
          conversationEndRef={conversationEndRef}
          onSuggestion={chat.submitQuestion}
        />
        <ChatComposer
          draft={chat.draft}
          pending={chat.streaming || !chat.isReady}
          onDraftChange={chat.setDraft}
          onSubmit={chat.submitQuestion}
        />
      </main>
    </div>
  );
}


export default App;
