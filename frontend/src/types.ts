export type ChatMode = "pipeline" | "react";

export interface ChatRequest {
  question: string;
  top_k: number;
  mode: ChatMode;
}

export interface Paper {
  arxiv_id: string;
  title: string;
  document: string;
  entry_url: string;
  primary_category: string;
  similarity: number | null;
  keyword_score: number | null;
  fusion_score: number | null;
  rerank_score: number | null;
}

export interface TraceEvent {
  stage: string;
  label: string;
  status:
    | "started"
    | "retrying"
    | "streaming"
    | "completed"
    | "failed"
    | "skipped";
  duration_ms: number;
  details: Record<string, unknown>;
}

export interface ModelUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface ConversationDecision {
  coverage: string;
  next_action: string;
  reason: string;
  standalone_question: string;
  reusable_arxiv_ids: string[];
  missing_aspects: string[];
  retrieval_query: string | null;
}

export interface ChatResponse {
  question: string;
  answer: string | null;
  generation_error: string | null;
  papers: Paper[];
  trace: TraceEvent[];
  retrieval_attempts: number;
  standalone_question: string | null;
  conversation_decision: ConversationDecision | null;
  response_kind: "research" | "conversation";
  mode: ChatMode;
  fallback_used: boolean;
  usage: ModelUsage | null;
}

export type ChatStreamEvent =
  | { type: "run_started"; question: string; mode: ChatMode }
  | { type: "trace"; event: TraceEvent }
  | { type: "status"; message: string }
  | { type: "assistant_delta"; delta: string }
  | {
      type: "assistant_completed";
      message: { content: string; usage: ModelUsage | null };
    }
  | { type: "run_completed"; usage: ModelUsage | null }
  | { type: "run_failed"; message: string }
  | { type: "result"; result: ChatResponse }
  | { type: "error"; message: string };

export interface KnowledgeBaseStats {
  paper_count: number;
  vector_count: number;
}

export interface ConversationSummary {
  conversation_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  turn_count: number;
}

export interface PersistedConversationTurn {
  turn_id: number;
  user_message: string;
  assistant_message: string;
  paper_ids: string[];
  papers: Paper[];
  response_kind: "research" | "conversation";
  created_at: string;
}

export interface Conversation extends ConversationSummary {
  turns: PersistedConversationTurn[];
}

export interface CompletedTurn {
  id: string;
  question: string;
  answer: string;
  papers: Paper[];
  responseKind: "research" | "conversation";
  result: ChatResponse | null;
}
