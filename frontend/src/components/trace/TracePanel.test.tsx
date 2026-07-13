import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { TracePanel } from "./TracePanel";


describe("TracePanel", () => {
  it("renders stage status timing and structured details", async () => {
    const user = userEvent.setup();
    render(
      <TracePanel
        events={[
          {
            stage: "retrieval_judgment",
            label: "DeepSeek 检索充分性判断",
            status: "completed",
            duration_ms: 1250,
            details: {
              sufficient: false,
              rewritten_query: "agentic rag comparison",
            },
          },
        ]}
      />,
    );

    await user.click(screen.getByText("Agent 执行过程"));

    expect(screen.getByText("DeepSeek 检索充分性判断")).toBeInTheDocument();
    expect(screen.getByText("retrieval_judgment")).toBeInTheDocument();
    expect(screen.getByText("1.25 s")).toBeInTheDocument();
    expect(screen.getByText(/证据充分/)).toBeInTheDocument();
    expect(screen.getByText(/agentic rag comparison/)).toBeInTheDocument();
  });
});
