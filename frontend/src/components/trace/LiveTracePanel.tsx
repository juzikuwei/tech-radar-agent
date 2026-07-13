import { formatDuration } from "../../formatters";
import type { TraceEvent } from "../../types";
import { TraceEventDetails } from "./TraceEventDetails";


export function LiveTracePanel({ events }: { events: TraceEvent[] }) {
  return (
    <div className="live-trace" aria-label="Agent 正在执行">
      <div className="live-trace-header">
        <span className="live-trace-spinner" aria-hidden="true" />
        <strong>正在分析和检索</strong>
      </div>
      <div className="live-trace-list">
        {events.map((event, index) => (
          <div className="live-trace-event" key={`${event.stage}-${index}`}>
            <div className={`live-trace-item ${event.status}`}>
              <span className="live-trace-check" aria-hidden="true">
                {event.status === "failed" ? "!" : "✓"}
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
          <span>{events.length ? "正在执行下一步…" : "请求已接收，正在启动…"}</span>
        </div>
      </div>
    </div>
  );
}
