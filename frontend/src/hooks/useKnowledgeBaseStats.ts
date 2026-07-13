import { useEffect, useState } from "react";

import { getKnowledgeBaseStats } from "../api";
import type { KnowledgeBaseStats } from "../types";


export function useKnowledgeBaseStats() {
  const [stats, setStats] = useState<KnowledgeBaseStats | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getKnowledgeBaseStats()
      .then((value) => {
        setStats(value);
        setError(null);
      })
      .catch((reason: unknown) => {
        setError(
          reason instanceof Error ? reason.message : "知识库状态加载失败",
        );
      });
  }, []);

  return { stats, error };
}
