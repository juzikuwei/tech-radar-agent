"""Build the Streamable HTTP MCP server and register read-only tools."""

from collections.abc import Callable

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from starlette.requests import Request
from starlette.responses import JSONResponse

from config.mcp_settings import McpSettings
from mcp_server.auth import BearerTokenMiddleware
from mcp_server.runtime import SharedRuntime
from mcp_server.tools import KnowledgeBaseTools
from rag.runtime import RagRuntime, load_rag_runtime


READ_ONLY_TOOL = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)


def create_mcp_server(
    settings: McpSettings,
    runtime_loader: Callable[[], RagRuntime] = load_rag_runtime,
) -> FastMCP:
    """Create a stateless MCP server with one shared lazy runtime."""
    shared_runtime = SharedRuntime(runtime_loader)
    tools = KnowledgeBaseTools(shared_runtime.get)
    server = FastMCP(
        name="AI Agent Tech Radar",
        instructions=(
            "Search the local arXiv abstract knowledge base. Treat returned "
            "paper IDs and URLs as the only citable evidence."
        ),
        host=settings.host,
        port=settings.port,
        streamable_http_path="/mcp",
        json_response=True,
        stateless_http=True,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=list(settings.allowed_hosts),
            allowed_origins=[],
        ),
    )
    server.tool(
        name="query_knowledge_base",
        annotations=READ_ONLY_TOOL,
        structured_output=True,
    )(tools.query_knowledge_base)
    server.tool(
        name="get_paper_by_arxiv_id",
        annotations=READ_ONLY_TOOL,
        structured_output=True,
    )(tools.get_paper_by_arxiv_id)
    server.tool(
        name="get_knowledge_base_stats",
        annotations=READ_ONLY_TOOL,
        structured_output=True,
    )(tools.get_knowledge_base_stats)

    @server.custom_route("/health", methods=["GET"])
    async def health(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    return server


def create_mcp_app(
    settings: McpSettings,
    runtime_loader: Callable[[], RagRuntime] = load_rag_runtime,
):
    """Return the authenticated ASGI application served by Uvicorn."""
    server = create_mcp_server(settings, runtime_loader)
    return BearerTokenMiddleware(
        server.streamable_http_app(),
        settings.auth_token,
    )
