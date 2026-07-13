import type { ChatResponse } from "../../types";
import { TracePanel } from "../trace/TracePanel";
import { PaperList } from "./PaperList";


export function ResultDetails({ result }: { result: ChatResponse }) {
  return (
    <div className="result-details">
      <PaperList papers={result.papers} />
      <TracePanel events={result.trace} />
    </div>
  );
}
