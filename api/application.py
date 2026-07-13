"""Assemble the FastAPI application and its process lifecycle."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers.chat import router as chat_router
from api.routers.system import router as system_router
from api.runtime import RuntimeLoader
from rag.runtime import load_rag_runtime


LOCAL_FRONTEND_ORIGINS = (
    "http://127.0.0.1:5173",
    "http://localhost:5173",
)


def create_app(runtime_loader: RuntimeLoader = load_rag_runtime) -> FastAPI:
    """Create an API application with injectable startup resources."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = runtime_loader()
        yield

    app = FastAPI(
        title="AI/Agent Tech Radar API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(LOCAL_FRONTEND_ORIGINS),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )
    app.include_router(system_router)
    app.include_router(chat_router)
    return app
