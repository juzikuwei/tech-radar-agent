"""Attach the shared RAG runtime to FastAPI requests."""

from collections.abc import Callable

from fastapi import HTTPException, Request

from rag.runtime import RagRuntime, load_rag_runtime

RuntimeLoader = Callable[[], RagRuntime]


def runtime_from_request(request: Request) -> RagRuntime:
    """Return the initialized runtime attached during application startup."""
    runtime = getattr(request.app.state, "runtime", None)
    if not isinstance(runtime, RagRuntime):
        raise HTTPException(status_code=503, detail="API runtime is not ready")
    return runtime
