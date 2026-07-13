# Remote MCP Server

The project exposes a read-only Streamable HTTP MCP endpoint at `/mcp`.
It is a separate adapter that reuses the same `rag/` runtime and retrieval
services as FastAPI.

## Tools

### `query_knowledge_base(query, top_k=3)`

Returns at most five results. Each result contains only:

- `arxiv_id`
- `title`
- `abstract_excerpt` (at most 600 characters)
- `primary_category`
- `score` (Cross-encoder score)
- `entry_url`

It does not return prompts, traces, embeddings, BM25/RRF intermediate scores,
database fields, or filesystem paths.

### `get_paper_by_arxiv_id(arxiv_id)`

Returns only:

- `arxiv_id`
- `title`
- `abstract`
- `primary_category`
- `published_at`
- `entry_url`

### `get_knowledge_base_stats()`

Returns only `paper_count` and `vector_count`.

## Configuration

Configure these variables privately in the ignored `.env` file or in the
process environment:

- `MCP_AUTH_TOKEN` (required, at least 16 characters)
- `MCP_HOST` (optional; defaults to loopback)
- `MCP_PORT` (optional; defaults to 8100)
- `MCP_ALLOWED_HOSTS` (optional comma-separated Host allowlist)

Do not commit tokens. Use a different token per environment. The current token
boundary is intended for development and limited invitations; public onboarding
should migrate to OAuth 2.1.

## Run locally

```powershell
.\.venv\Scripts\Activate.ps1
python -m mcp_server.main
```

The health endpoint is `/health`; MCP clients connect to `/mcp` and send:

```text
Authorization: Bearer <private token>
```

For VPS deployment, keep the Python service on loopback and publish only HTTPS
through Caddy or Nginx. Add the public domain to `MCP_ALLOWED_HOSTS`.
