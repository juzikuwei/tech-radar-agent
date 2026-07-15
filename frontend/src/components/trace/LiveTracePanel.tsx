import { formatDuration } from "../../formatters";
import type { TraceEvent } from "../../types";
import { presentTraceEvents } from "./tracePresentation";


export function LiveTracePanel({
  events,
  status,
}: {
  events: TraceEvent[];
  status: string;
}) {
  const steps = presentTraceEvents(events);
  return (
    <div className="live-trace" aria-label="Agent 正在执行">
      <div className="live-trace-header">
        <span className="live-trace-spinner" aria-hidden="true" />
        <strong>{status}</strong>
      </div>
      <div className="live-trace-list">
        {steps.map((step) => (
          <div className="live-trace-event" key={step.key}>
            <div className={`live-trace-item ${step.status}`}>
              <span className="live-trace-check" aria-hidden="true">
                {step.status === "failed"
                  ? "!"
                  : step.status === "started" || step.status === "retrying"
                    ? "·"
                    : "✓"}
              </span>
              <span>{step.label}</span>
              <time>{formatDuration(step.duration_ms)}</time>
            </div>
          </div>
        ))}
        <div className="live-trace-item active">
          <span className="active-step-dot" aria-hidden="true" />
          <span>{status}</span>
        </div>
      </div>
    </div>
  );
}
