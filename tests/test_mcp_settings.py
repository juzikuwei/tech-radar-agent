from config.mcp_settings import McpSettings


def test_loads_mcp_settings_from_environment(monkeypatch: object) -> None:
    monkeypatch.setenv("MCP_AUTH_TOKEN", "a-secure-local-token")
    monkeypatch.setenv("MCP_PORT", "8200")
    monkeypatch.setenv("MCP_ALLOWED_HOSTS", "mcp.example.com,127.0.0.1:*")

    settings = McpSettings.from_env()

    assert settings.host == "127.0.0.1"
    assert settings.port == 8200
    assert settings.allowed_hosts == ("mcp.example.com", "127.0.0.1:*")
