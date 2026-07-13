"""ASGI and command-line entry point for the remote MCP server."""

import uvicorn

from config.mcp_settings import McpSettings
from mcp_server.server import create_mcp_app


settings = McpSettings.from_env()
app = create_mcp_app(settings)


def main() -> None:
    """Run the authenticated Streamable HTTP server."""
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
