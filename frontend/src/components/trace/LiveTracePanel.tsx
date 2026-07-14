import { formatDuration } from "../../formatters";
import type { TraceEvent } from "../../types";
import { TraceEventDetails } from "./TraceEventDetails";


export function LiveTracePanel({
  events,
  status,
}: {
  events: TraceEvent[];
  status: string;
}) {
  return (
    <div className="live-trace" aria-label="Agent 正在执行">
      <div className="live-trace-header">
        <span className="live-trace-spinner" aria-hidden="true" />
        <strong>{status}</strong>
      </div>
      <div className="live-trace-list">
        {events.map((event, index) => (
          <div className="live-trace-event" key={`${event.stage}-${index}`}>
            <div className={`live-trace-item ${event.status}`}>
              <span className="live-trace-check" aria-hidden="true">
                {event.status === "failed"
                  ? "!"
                  : event.status === "started" || event.status === "retrying"
                    ? "·"
                    : "✓"}
              </span>
              <span>{event.label}</span>
              <time>{formatDuration(event.duration_ms)}</time>
            </div>
            {event.stage.startsWith("agent_") ? (
              <TraceEventDetails details={event.details} compact />
            ) : null}
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
