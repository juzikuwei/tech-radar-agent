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
});
