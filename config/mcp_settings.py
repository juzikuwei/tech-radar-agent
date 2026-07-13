"""Load and validate remote MCP server configuration."""

from dataclasses import dataclass
import os
from pathlib import Path

from config.environment import DEFAULT_ENV_PATH, load_repository_env


@dataclass(frozen=True)
class McpSettings:
    """Network and authentication settings for Streamable HTTP MCP."""

    auth_token: str
    host: str = "127.0.0.1"
    port: int = 8100
    allowed_hosts: tuple[str, ...] = (
        "127.0.0.1:*",
        "localhost:*",
    )

    @classmethod
    def from_env(cls, env_path: Path = DEFAULT_ENV_PATH) -> "McpSettings":
        """Read MCP settings from process variables or the ignored `.env`."""
        load_repository_env(env_path)
        auth_token = os.getenv("MCP_AUTH_TOKEN", "").strip()
        if len(auth_token) < 16:
            raise ValueError("MCP_AUTH_TOKEN must contain at least 16 characters")

        host = os.getenv("MCP_HOST", "127.0.0.1").strip() or "127.0.0.1"
        try:
            port = int(os.getenv("MCP_PORT", "8100"))
        except ValueError as error:
            raise ValueError("MCP_PORT must be an integer") from error
        if not 1 <= port <= 65_535:
            raise ValueError("MCP_PORT must be between 1 and 65535")

        configured_hosts = os.getenv("MCP_ALLOWED_HOSTS", "").strip()
        allowed_hosts = (
            tuple(
                item.strip()
                for item in configured_hosts.split(",")
                if item.strip()
            )
            if configured_hosts
            else ("127.0.0.1:*", "localhost:*")
        )
        if not allowed_hosts:
            raise ValueError("MCP_ALLOWED_HOSTS must not be empty")
        return cls(
            auth_token=auth_token,
            host=host,
            port=port,
            allowed_hosts=allowed_hosts,
        )
