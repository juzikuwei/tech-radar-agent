import { useEffect, useState } from "react";

import { getKnowledgeBaseStats } from "../api";
import type { KnowledgeBaseStats } from "../types";

const RETRY_INTERVAL_MS = 30_000;


export function useKnowledgeBaseStats() {
  const [stats, setStats] = useState<KnowledgeBaseStats | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let retryTimer: number | undefined;

    const load = async () => {
      try {
        const value = await getKnowledgeBaseStats();
        if (!cancelled) {
          setStats(value);
          setError(null);
        }
      } catch (reason: unknown) {
        if (!cancelled) {
          setError(
            reason instanceof Error ? reason.message : "知识库状态加载失败",
          );
          retryTimer = window.setTimeout(() => {
            void load();
          }, RETRY_INTERVAL_MS);
        }
      }
    };
    void load();

    return () => {
      cancelled = true;
      window.clearTimeout(retryTimer);
    };
  }, []);

  return { stats, error };
}
