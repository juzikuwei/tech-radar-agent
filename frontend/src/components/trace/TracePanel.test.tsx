import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { TracePanel } from "./TracePanel";


describe("TracePanel", () => {
  it("merges lifecycle events and keeps raw details collapsed", async () => {
    const user = userEvent.setup();
    render(
      <TracePanel
        events={[
          {
            stage: "model",
            label: "模型开始生成下一步",
            status: "started",
            duration_ms: 0,
            details: { available_tools: ["search_papers"] },
          },
          {
            stage: "model",
            label: "模型完成本轮输出",
            status: "completed",
            duration_ms: 1250,
            details: {
              tool_calls: ["search_papers"],
              usage: { total_tokens: 120 },
            },
          },
        ]}
      />,
    );

    await user.click(screen.getByText("已完成 1 个执行步骤"));

    expect(screen.getByText("模型选择论文检索")).toBeInTheDocument();
    expect(screen.getByText("1.25 s")).toBeInTheDocument();
    await user.click(screen.getByText("查看技术详情"));
    expect(screen.getByText("model · started")).toBeInTheDocument();
    expect(screen.getByText("model · completed")).toBeInTheDocument();
    expect(screen.getByText(/Token usage/)).toBeInTheDocument();
  });
});
