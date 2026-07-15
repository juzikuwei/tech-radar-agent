import { formatDuration } from "../../formatters";
import type { TraceEvent } from "../../types";
import { TraceEventDetails } from "./TraceEventDetails";
import { presentTraceEvents } from "./tracePresentation";


export function TracePanel({
  events,
  title,
}: {
  events: TraceEvent[];
  title?: string;
}) {
  const steps = presentTraceEvents(events);
  const displayTitle = title ?? `已完成 ${steps.length} 个执行步骤`;
  return (
    <details className="detail-section trace-section">
      <summary>
        <span>{displayTitle}</span>
        <span className="summary-count">{steps.length}</span>
      </summary>
      <div className="trace-list">
        {steps.map((step, index) => (
          <div
            className={`trace-item ${step.status}`}
            key={step.key}
          >
            <div className="trace-index">{index + 1}</div>
            <div className="trace-body">
              <div className="trace-heading">
                <strong>{step.label}</strong>
                <span>{formatDuration(step.duration_ms)}</span>
              </div>
              <details className="trace-technical-details">
                <summary>查看技术详情</summary>
                {step.rawEvents.map((event, rawIndex) => (
                  <div className="trace-raw-event" key={`${event.stage}-${rawIndex}`}>
                    <code>{event.stage} · {event.status}</code>
                    <strong>{event.label}</strong>
                    <TraceEventDetails details={event.details} />
                  </div>
                ))}
              </details>
            </div>
          </div>
        ))}
      </div>
    </details>
  );
}
