import { formatDuration } from "../../formatters";
import type { TraceEvent } from "../../types";
import { TraceEventDetails } from "./TraceEventDetails";


export function TracePanel({
  events,
  title = "Agent 执行过程",
}: {
  events: TraceEvent[];
  title?: string;
}) {
  return (
    <details className="detail-section trace-section">
      <summary>
        <span>{title}</span>
        <span className="summary-count">{events.length}</span>
      </summary>
      <div className="trace-list">
        {events.map((event, index) => (
          <div
            className={`trace-item ${event.status}`}
            key={`${event.stage}-${index}`}
          >
            <div className="trace-index">{index + 1}</div>
            <div className="trace-body">
              <div className="trace-heading">
                <strong>{event.label}</strong>
                <span>{formatDuration(event.duration_ms)}</span>
              </div>
              <p>{event.stage}</p>
              <TraceEventDetails details={event.details} />
            </div>
          </div>
        ))}
      </div>
    </details>
  );
}
