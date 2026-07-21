import { describe, expect, it } from "vitest";

import type { TraceEvent } from "../../types";
import { presentTraceEvents } from "./tracePresentation";


function event(overrides: Partial<TraceEvent>): TraceEvent {
  return {
    stage: "model",
    label: "raw event",
    status: "completed",
    duration_ms: 10,
    details: {},
    ...overrides,
  };
}


describe("presentTraceEvents", () => {
  it("merges model and tool lifecycle pairs into product steps", () => {
    const steps = presentTraceEvents([
      event({ status: "started", details: { available_tools: ["search_papers"] } }),
      event({ details: { tool_calls: ["search_papers"] }, duration_ms: 100 }),
      event({ stage: "tool", status: "started", details: { tool: "search_papers" } }),
      event({
        stage: "tool",
        details: { tool: "search_papers", output: { result_count: 5 } },
        duration_ms: 250,
      }),
    ]);

    expect(steps).toHaveLength(2);
    expect(steps[0].label).toBe("模型选择论文检索");
    expect(steps[0].rawEvents).toHaveLength(2);
    expect(steps[1].label).toBe("检索本地论文 · 找到 5 篇");
    expect(steps[1].duration_ms).toBe(250);
  });

  it("uses a safe-refusal label for no-evidence validation", () => {
    const [step] = presentTraceEvents([
      event({
        stage: "answer_validation",
        status: "failed",
        details: { reason: "no_evidence" },
      }),
    ]);

    expect(step.label).toBe("本地证据不足，已安全拒答");
    expect(step.status).toBe("failed");
  });

  it("folds retrying events into the pending model step", () => {
    const steps = presentTraceEvents([
      event({ status: "started", details: {} }),
      event({ status: "retrying", duration_ms: 40 }),
    ]);

    expect(steps).toHaveLength(1);
    expect(steps[0].label).toBe("模型请求重试");
    expect(steps[0].status).toBe("retrying");
    expect(steps[0].rawEvents).toHaveLength(2);
  });

  it("closes a retried model step with its terminal event", () => {
    const steps = presentTraceEvents([
      event({ status: "started", details: {} }),
      event({ status: "retrying", duration_ms: 40 }),
      event({ status: "completed", details: {}, duration_ms: 900 }),
    ]);

    expect(steps).toHaveLength(1);
    expect(steps[0].label).toBe("模型生成回答");
    expect(steps[0].status).toBe("completed");
    expect(steps[0].duration_ms).toBe(900);
    expect(steps[0].rawEvents).toHaveLength(3);
  });
});
