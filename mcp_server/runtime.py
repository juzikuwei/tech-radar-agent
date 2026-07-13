"""Lazily share one RAG runtime across all MCP tool calls."""

from collections.abc import Callable
from threading import Lock

from rag.runtime import RagRuntime


class SharedRuntime:
    """Load expensive local resources at most once per MCP process."""

    def __init__(self, loader: Callable[[], RagRuntime]) -> None:
        self._loader = loader
        self._runtime: RagRuntime | None = None
        self._lock = Lock()

    def get(self) -> RagRuntime:
        """Return the cached runtime, loading it on the first tool call."""
        if self._runtime is not None:
            return self._runtime
        with self._lock:
            if self._runtime is None:
                self._runtime = self._loader()
        return self._runtime
