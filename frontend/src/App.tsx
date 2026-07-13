import { useEffect, useRef } from "react";

import { MainHeader } from "./components/MainHeader";
import { Sidebar } from "./components/Sidebar";
import { ChatComposer } from "./components/chat/ChatComposer";
import { ConversationFeed } from "./components/chat/ConversationFeed";
import { useChatSession } from "./hooks/useChatSession";
import { useKnowledgeBaseStats } from "./hooks/useKnowledgeBaseStats";


function App() {
  const { stats, error: statsError } = useKnowledgeBaseStats();
  const chat = useChatSession();
  const conversationEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const conversationEnd = conversationEndRef.current;
    if (typeof conversationEnd?.scrollIntoView === "function") {
      conversationEnd.scrollIntoView({ behavior: "smooth" });
    }
  }, [chat.turns, chat.pendingQuestion, chat.requestError]);

  return (
    <div className="app-shell">
      <Sidebar
        stats={stats}
        statsError={statsError}
        turnCount={chat.turns.length}
        canReset={chat.canReset}
        onReset={chat.resetConversation}
        mode={chat.mode}
        onModeChange={chat.setMode}
      />

      <main className="main-panel">
        <MainHeader mode={chat.mode} />
        <ConversationFeed
          turns={chat.turns}
          pendingQuestion={chat.pendingQuestion}
          liveTrace={chat.liveTrace}
          requestError={chat.requestError}
          failedResult={chat.failedResult}
          conversationEndRef={conversationEndRef}
          onSuggestion={chat.submitQuestion}
        />
        <ChatComposer
          draft={chat.draft}
          pending={Boolean(chat.pendingQuestion)}
          onDraftChange={chat.setDraft}
          onSubmit={chat.submitQuestion}
        />
      </main>
    </div>
  );
}


export default App;
