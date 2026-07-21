import { scoreLabel } from "../../formatters";
import { safeArxivUrl } from "../../lib/arxivUrl";
import type { Paper } from "../../types";


export function PaperList({ papers }: { papers: Paper[] }) {
  return (
    <details className="detail-section">
      <summary>
        <span>论文证据</span>
        <span className="summary-count">{papers.length}</span>
      </summary>
      <div className="paper-grid">
        {papers.map((paper, index) => {
          const safeUrl = safeArxivUrl(paper);
          return (
            <article className="paper-card" key={paper.arxiv_id}>
              <div className="paper-rank">
                {String(index + 1).padStart(2, "0")}
              </div>
              <div>
                <div className="paper-meta">
                  <span>{paper.primary_category}</span>
                  <span>{scoreLabel(paper)}</span>
                </div>
                {safeUrl ? (
                  <a href={safeUrl} target="_blank" rel="noreferrer noopener">
                    {paper.title}
                  </a>
                ) : (
                  <span className="paper-title-plain">{paper.title}</span>
                )}
                <p>arXiv:{paper.arxiv_id}</p>
                <details className="abstract-details">
                  <summary>查看标题与摘要</summary>
                  <p>{paper.document}</p>
                </details>
              </div>
            </article>
          );
        })}
      </div>
    </details>
  );
}
