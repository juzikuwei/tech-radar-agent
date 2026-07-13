"""Small interim Bearer Token boundary for the remote MCP endpoint."""

from secrets import compare_digest
from typing import Any


class BearerTokenMiddleware:
    """Require one configured Bearer Token for the `/mcp` endpoint."""

    def __init__(self, app: Any, auth_token: str) -> None:
        if len(auth_token) < 16:
            raise ValueError("auth_token must contain at least 16 characters")
        self._app = app
        self._auth_token = auth_token

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http" or not _is_mcp_path(str(scope.get("path", ""))):
            await self._app(scope, receive, send)
            return

        supplied_token = _bearer_token(scope.get("headers", []))
        if supplied_token is not None and compare_digest(
            supplied_token,
            self._auth_token,
        ):
            await self._app(scope, receive, send)
            return

        body = b'{"error":"unauthorized"}'
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                    (b"www-authenticate", b"Bearer"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def _is_mcp_path(path: str) -> bool:
    return path.rstrip("/") == "/mcp"


def _bearer_token(headers: list[tuple[bytes, bytes]]) -> str | None:
    for name, value in headers:
        if name.lower() != b"authorization":
            continue
        scheme, separator, token = value.decode("latin-1").partition(" ")
        if separator and scheme.lower() == "bearer" and token.strip():
            return token.strip()
    return None
