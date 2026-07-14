import { detailLabel, formatDetail } from "../../formatters";


interface PlanItem {
  id: string;
  question: string;
  status: string;
}


export function TraceEventDetails({
  details,
  compact = false,
}: {
  details: Record<string, unknown>;
  compact?: boolean;
}) {
  const plan = readPlan(details.plan);
  const reason = typeof details.reason_summary === "string"
    ? details.reason_summary
    : null;
  const compactKeys = [
    "tool",
    "arguments",
    "output",
    "usage",
    "finish_reason",
  ];
  const hiddenKeys = new Set(["plan", "reason_summary"]);
  const entries = Object.entries(details).filter(([key]) => {
    if (hiddenKeys.has(key)) return false;
    return !compact || compactKeys.includes(key);
  });

  if (!plan.length && !reason && !entries.length) {
    return null;
  }

  return (
    <div className={compact ? "trace-event-details compact" : "trace-event-details"}>
      {reason ? <p className="trace-reason">{reason}</p> : null}
      {plan.length ? (
        <ol className="research-plan-list">
          {plan.map((item) => (
            <li key={item.id} className={item.status}>
              <span className="plan-status" aria-hidden="true">
                {item.status === "covered" ? "✓" : item.status === "unresolved" ? "!" : "·"}
              </span>
              <span>{item.question}</span>
            </li>
          ))}
        </ol>
      ) : null}
      {entries.length ? (
        <div className="trace-details">
          {entries.map(([key, value]) => (
            <span key={key}>
              <b>{detailLabel(key)}</b> {formatDetail(value)}
            </span>
          ))}
        </div>
      ) : null}
    </div>
  );
}


function readPlan(value: unknown): PlanItem[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is PlanItem => {
    if (!item || typeof item !== "object") return false;
    const candidate = item as Record<string, unknown>;
    return typeof candidate.id === "string"
      && typeof candidate.question === "string"
      && typeof candidate.status === "string";
  });
}
