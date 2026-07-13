import { scoreLabel } from "../../formatters";
import type { Paper } from "../../types";


export function PaperList({ papers }: { papers: Paper[] }) {
  return (
    <details className="detail-section">
      <summary>
        <span>论文证据</span>
        <span className="summary-count">{papers.length}</span>
      </summary>
      <div className="paper-grid">
        {papers.map((paper, index) => (
          <article className="paper-card" key={paper.arxiv_id}>
            <div className="paper-rank">
              {String(index + 1).padStart(2, "0")}
            </div>
            <div>
              <div className="paper-meta">
                <span>{paper.primary_category}</span>
                <span>{scoreLabel(paper)}</span>
              </div>
              <a href={paper.entry_url} target="_blank" rel="noreferrer">
                {paper.title}
              </a>
              <p>arXiv:{paper.arxiv_id}</p>
              <details className="abstract-details">
                <summary>查看标题与摘要</summary>
                <p>{paper.document}</p>
              </details>
            </div>
          </article>
        ))}
      </div>
    </details>
  );
}
