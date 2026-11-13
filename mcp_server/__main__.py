"""Run the MCP server: python -m mcp_server  (Streamable HTTP on MCP_HOST:MCP_PORT, path /mcp)."""

import hmac
import logging
import sys

import uvicorn
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.responses import JSONResponse

from mcp_server.server import mcp


class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    mcp_api_key: SecretStr = SecretStr("")
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8765


class RequireServiceKey:
    """ASGI middleware: every HTTP request must carry the shared service key."""

    def __init__(self, app, key: str):
        self.app, self.expected = app, f"Bearer {key}".encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            given = dict(scope.get("headers") or []).get(b"authorization", b"")
            if not hmac.compare_digest(given, self.expected):  # constant time
                await JSONResponse({"error": "unauthorized"}, status_code=401)(scope, receive, send)
                return
        await self.app(scope, receive, send)


def build_app(key: str):
    if not key:
        raise SystemExit("MCP_API_KEY is not set – refusing to start an unauthenticated server.")
    return RequireServiceKey(mcp.streamable_http_app(streamable_http_path="/mcp"), key)


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(name)s %(message)s")
    s = ServerSettings()
    uvicorn.run(build_app(s.mcp_api_key.get_secret_value()), host=s.mcp_host, port=s.mcp_port, log_level="warning")


if __name__ == "__main__":
    main()
