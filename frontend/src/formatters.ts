import type { Paper } from "./types";

export function scoreLabel(paper: Paper): string {
  if (paper.rerank_score !== null) {
    return `重排 ${paper.rerank_score.toFixed(3)}`;
  }
  if (paper.similarity !== null) {
    return `相似度 ${paper.similarity.toFixed(3)}`;
  }
  return "已选入证据";
}

export function formatDuration(durationMs: number): string {
  if (durationMs < 1) return "<1 ms";
  if (durationMs < 1_000) return `${Math.round(durationMs)} ms`;
  return `${(durationMs / 1_000).toFixed(2)} s`;
}

export function formatDetail(value: unknown): string {
  if (Array.isArray(value)) return value.join("、") || "—";
  if (typeof value === "boolean") return value ? "是" : "否";
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

export function detailLabel(key: string): string {
  const labels: Record<string, string> = {
    round: "轮次",
    query: "查询",
    sufficient: "证据充分",
    reason: "理由",
    rewritten_query: "改写查询",
    next_action: "动作",
    tool: "工具",
    tool_calls: "工具调用",
    tool_call_count: "已调用工具",
    available_tools: "可用工具",
    arguments: "参数",
    output: "输出",
    usage: "Token usage",
    finish_reason: "结束原因",
    top_k: "返回数量",
    selected_arxiv_ids: "最终证据",
    coverage: "覆盖",
    result_count: "结果数",
    paper_count: "论文数",
    top_arxiv_ids: "Top IDs",
    reusable_arxiv_ids: "复用 IDs",
    error: "错误",
    cited_arxiv_ids: "已验证引用",
    unknown_citation_ids: "未知引用",
    answer_char_count: "回答字符数",
  };
  return labels[key] ?? key.replaceAll("_", " ");
}
