export const MAX_HISTORY_TURNS = 6;

export const SUGGESTIONS = [
  "Agentic RAG 和普通 RAG 有什么区别？",
  "多 Agent 系统执行失败后，如何定位最早出错步骤？",
  "Cross-encoder 在混合检索中有什么作用？",
];

export const ACTION_LABELS: Record<string, string> = {
  answer_from_existing: "直接复用已有证据",
  retrieve_missing: "检索缺失信息",
  fresh_retrieval: "开始新话题检索",
};
