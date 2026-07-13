export interface ConversationTurnInput {
  user_message: string;
  assistant_message: string;
}

export type ChatMode = "pipeline" | "react";

export interface ChatRequest {
  question: string;
  conversation_history: ConversationTurnInput[];
  active_evidence_ids: string[];
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
  status: "completed" | "failed" | "skipped";
  duration_ms: number;
  details: Record<string, unknown>;
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
  mode: ChatMode;
  fallback_used: boolean;
}

export type ChatStreamEvent =
  | { type: "run_started"; question: string; mode: ChatMode }
  | { type: "trace"; event: TraceEvent }
  | { type: "result"; result: ChatResponse }
  | { type: "error"; message: string };

export interface KnowledgeBaseStats {
  paper_count: number;
  vector_count: number;
}

export interface CompletedTurn {
  id: string;
  question: string;
  answer: string;
  result: ChatResponse;
}
