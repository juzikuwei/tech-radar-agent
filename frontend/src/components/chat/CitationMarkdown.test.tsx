import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { Paper } from "../../types";
import { CitationMarkdown } from "./CitationMarkdown";


function paper(overrides: Partial<Paper> = {}): Paper {
  return {
    arxiv_id: "2501.09136",
    title: "Agentic Retrieval-Augmented Generation",
    document: "Title\nAbstract",
    entry_url: "https://arxiv.org/abs/2501.09136",
    primary_category: "cs.AI",
    similarity: 0.8,
    keyword_score: null,
    fusion_score: 0.7,
    rerank_score: 4.1,
    ...overrides,
  };
}


describe("CitationMarkdown", () => {
  it("links only citations backed by the current paper list", () => {
    render(
      <CitationMarkdown
        content="真实引用 [2501.09136]，未知引用 [9999.99999]。"
        papers={[paper()]}
      />,
    );

    const link = screen.getByRole("link", { name: "打开 arXiv 论文 2501.09136" });
    expect(link).toHaveAttribute("href", "https://arxiv.org/abs/2501.09136");
    expect(link).toHaveAttribute("target", "_blank");
    expect(screen.queryByRole("link", { name: /9999\.99999/ })).not.toBeInTheDocument();
  });

  it("does not link citations inside code or rewrite existing markdown links", () => {
    render(
      <CitationMarkdown
        content={"`[2501.09136]` 与 [已有链接](https://example.test)"}
        papers={[paper()]}
      />,
    );

    expect(screen.getByText("[2501.09136]").tagName).toBe("CODE");
    expect(screen.getByRole("link", { name: "已有链接" })).toHaveAttribute(
      "href",
      "https://example.test",
    );
  });

  it("rejects an untrusted paper entry URL", () => {
    render(
      <CitationMarkdown
        content="引用 [2501.09136]。"
        papers={[paper({ entry_url: "javascript:alert(1)" })]}
      />,
    );

    expect(screen.queryByRole("link")).not.toBeInTheDocument();
  });
});
