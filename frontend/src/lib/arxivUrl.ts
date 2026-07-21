import type { Paper } from "../types";


/**
 * 仅当 entry_url 是 https?://arxiv.org/abs/<arxiv_id> 形式且与论文 ID 一致时
 * 返回可安全渲染的链接，否则返回 null（调用方应降级为纯文本）。
 */
export function safeArxivUrl(paper: Paper): string | null {
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
