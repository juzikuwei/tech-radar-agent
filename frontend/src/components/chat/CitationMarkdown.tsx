import { Children, Fragment, type ReactNode } from "react";
import ReactMarkdown, { type Components } from "react-markdown";

import type { Paper } from "../../types";


export function CitationMarkdown({
  content,
  papers,
}: {
  content: string;
  papers: Paper[];
}) {
  const citationLinks = buildCitationLinks(papers);
  const citationChildren = (children: ReactNode) => (
    <>{linkifyChildren(children, citationLinks)}</>
  );
  const components: Components = {
    p: ({ node: _node, children, ...props }) => (
      <p {...props}>{citationChildren(children)}</p>
    ),
    li: ({ node: _node, children, ...props }) => (
      <li {...props}>{citationChildren(children)}</li>
    ),
    strong: ({ node: _node, children, ...props }) => (
      <strong {...props}>{citationChildren(children)}</strong>
    ),
    em: ({ node: _node, children, ...props }) => (
      <em {...props}>{citationChildren(children)}</em>
    ),
    blockquote: ({ node: _node, children, ...props }) => (
      <blockquote {...props}>{citationChildren(children)}</blockquote>
    ),
    h1: ({ node: _node, children, ...props }) => (
      <h1 {...props}>{citationChildren(children)}</h1>
    ),
    h2: ({ node: _node, children, ...props }) => (
      <h2 {...props}>{citationChildren(children)}</h2>
    ),
    h3: ({ node: _node, children, ...props }) => (
      <h3 {...props}>{citationChildren(children)}</h3>
    ),
    h4: ({ node: _node, children, ...props }) => (
      <h4 {...props}>{citationChildren(children)}</h4>
    ),
    h5: ({ node: _node, children, ...props }) => (
      <h5 {...props}>{citationChildren(children)}</h5>
    ),
    h6: ({ node: _node, children, ...props }) => (
      <h6 {...props}>{citationChildren(children)}</h6>
    ),
    a: ({ node: _node, children, ...props }) => (
      <a {...props} target="_blank" rel="noreferrer noopener">
        {children}
      </a>
    ),
  };

  return <ReactMarkdown components={components}>{content}</ReactMarkdown>;
}


function buildCitationLinks(papers: Paper[]): Map<string, string> {
  const links = new Map<string, string>();
  for (const paper of papers) {
    const safeUrl = safeArxivUrl(paper);
    if (safeUrl !== null) {
      links.set(paper.arxiv_id, safeUrl);
    }
  }
  return links;
}


function safeArxivUrl(paper: Paper): string | null {
  try {
    const url = new URL(paper.entry_url);
    if (!["http:", "https:"].includes(url.protocol) || url.hostname !== "arxiv.org") {
      return null;
    }
    const prefix = "/abs/";
    if (!url.pathname.startsWith(prefix)) return null;
    const pathId = decodeURIComponent(url.pathname.slice(prefix.length))
      .replace(/\/$/, "")
      .replace(/v\d+$/i, "");
    if (pathId !== paper.arxiv_id) return null;
    return url.toString();
  } catch {
    return null;
  }
}


function linkifyChildren(
  children: ReactNode,
  citationLinks: Map<string, string>,
): ReactNode {
  return Children.map(children, (child) => {
    if (typeof child !== "string") return child;
    return linkifyText(child, citationLinks);
  });
}


function linkifyText(
  text: string,
  citationLinks: Map<string, string>,
): ReactNode {
  if (!citationLinks.size) return text;
  const ids = [...citationLinks.keys()]
    .sort((left, right) => right.length - left.length)
    .map(escapeRegExp);
  const pattern = new RegExp(`\\[(${ids.join("|")})\\]`, "g");
  const parts: ReactNode[] = [];
  let cursor = 0;

  for (const match of text.matchAll(pattern)) {
    const index = match.index ?? 0;
    const arxivId = match[1];
    if (index > cursor) parts.push(text.slice(cursor, index));
    parts.push(
      <a
        className="citation-link"
        href={citationLinks.get(arxivId)}
        key={`${arxivId}-${index}`}
        target="_blank"
        rel="noreferrer noopener"
        aria-label={`打开 arXiv 论文 ${arxivId}`}
      >
        [{arxivId}]
      </a>,
    );
    cursor = index + match[0].length;
  }
  if (cursor === 0) return text;
  if (cursor < text.length) parts.push(text.slice(cursor));
  return parts.map((part, index) => <Fragment key={index}>{part}</Fragment>);
}


function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
