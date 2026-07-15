import type { TraceEvent } from "../../types";


export interface PresentedTraceStep {
  key: string;
  label: string;
  status: TraceEvent["status"];
  duration_ms: number;
  rawEvents: TraceEvent[];
}


export function presentTraceEvents(events: TraceEvent[]): PresentedTraceStep[] {
  const steps: PresentedTraceStep[] = [];
  let pendingModelIndex: number | null = null;
  const pendingTools = new Map<string, number>();

  events.forEach((event, eventIndex) => {
    if (event.stage === "model") {
      if (event.status === "started") {
        pendingModelIndex = steps.length;
        steps.push(toStep(event, eventIndex, "模型正在分析问题"));
        return;
      }
      if (
        pendingModelIndex !== null
        && ["completed", "failed"].includes(event.status)
      ) {
        steps[pendingModelIndex] = mergeStep(
          steps[pendingModelIndex],
          event,
          modelLabel(event),
        );
        pendingModelIndex = null;
        return;
      }
    }

    if (event.stage === "tool") {
      const tool = readString(event.details.tool) ?? "unknown";
      if (event.status === "started") {
        pendingTools.set(tool, steps.length);
        steps.push(toStep(event, eventIndex, pendingToolLabel(tool)));
        return;
      }
      const pendingIndex = pendingTools.get(tool);
      if (
        pendingIndex !== undefined
        && ["completed", "failed"].includes(event.status)
      ) {
        steps[pendingIndex] = mergeStep(
          steps[pendingIndex],
          event,
          toolLabel(tool, event),
        );
        pendingTools.delete(tool);
        return;
      }
    }

    steps.push(toStep(event, eventIndex, standaloneLabel(event)));
  });

  return steps;
}


function toStep(
  event: TraceEvent,
  eventIndex: number,
  label: string,
): PresentedTraceStep {
  return {
    key: `${event.stage}-${eventIndex}`,
    label,
    status: event.status,
    duration_ms: event.duration_ms,
    rawEvents: [event],
  };
}


function mergeStep(
  started: PresentedTraceStep,
  terminal: TraceEvent,
  label: string,
): PresentedTraceStep {
  return {
    ...started,
    label,
    status: terminal.status,
    duration_ms: terminal.duration_ms,
    rawEvents: [...started.rawEvents, terminal],
  };
}


function modelLabel(event: TraceEvent): string {
  if (event.status === "failed") return "模型生成失败";
  const toolCalls = Array.isArray(event.details.tool_calls)
    ? event.details.tool_calls.filter((value): value is string => typeof value === "string")
    : [];
  if (toolCalls.includes("search_papers")) return "模型选择论文检索";
  if (toolCalls.includes("web_search")) return "模型选择网页查询";
  if (toolCalls.length) return "模型选择工具";
  return "模型生成回答";
}


function pendingToolLabel(tool: string): string {
  if (tool === "search_papers") return "正在检索本地论文";
  if (tool === "web_search") return "正在查询网页资料";
  return "正在执行工具";
}


function toolLabel(tool: string, event: TraceEvent): string {
  if (event.status === "failed") {
    if (tool === "search_papers") return "本地论文检索失败";
    if (tool === "web_search") return "网页资料查询失败";
    return "工具执行失败";
  }
  const resultCount = readResultCount(event.details.output);
  if (tool === "search_papers") {
    return resultCount === null ? "检索本地论文" : `检索本地论文 · 找到 ${resultCount} 篇`;
  }
  if (tool === "web_search") {
    return resultCount === null ? "查询网页资料" : `查询网页资料 · 找到 ${resultCount} 条`;
  }
  return "执行外部工具";
}


function standaloneLabel(event: TraceEvent): string {
  const labels: Record<string, string> = {
    conversation_compaction: "压缩较早的会话上下文",
    conversation_evidence_decision: "判断是否复用历史证据",
    dense_retrieval: "向量召回论文",
    keyword_retrieval: "关键词召回论文",
    rank_fusion: "融合检索候选",
    candidate_rerank: "重排论文证据",
    final_union_rerank: "重排合并后的论文证据",
    active_evidence_rerank: "重排已有论文证据",
    retrieval_judgment: "判断论文证据是否充分",
    answer_generation: "生成最终回答",
    conversation_response: "生成直接对话回应",
    react_fallback: "降级到可靠管线",
  };
  if (event.stage === "answer_validation") {
    if (event.status !== "failed") return "引用校验通过";
    return event.details.reason === "no_evidence"
      ? "本地证据不足，已安全拒答"
      : "引用校验未通过，已安全拒答";
  }
  if (event.stage === "model" && event.status === "retrying") {
    return "模型请求重试";
  }
  return labels[event.stage] ?? event.label;
}


function readResultCount(value: unknown): number | null {
  if (!value || typeof value !== "object") return null;
  const resultCount = (value as Record<string, unknown>).result_count;
  return typeof resultCount === "number" ? resultCount : null;
}


function readString(value: unknown): string | null {
  return typeof value === "string" && value ? value : null;
}
