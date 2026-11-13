"""Application settings loaded from environment variables / .env."""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App
    app_name: str = "ComBank Knowledge Assistant"
    app_env: str = "dev"
    log_level: str = "INFO"
    jwt_secret: SecretStr = SecretStr("change-me")

    # LLM gateway (9router, OpenAI-compatible)
    llm_base_url: str = "http://localhost:20128/v1"
    llm_api_key: SecretStr = SecretStr("")
    llm_model_strong: str = "orysys-fast"
    llm_model_fast: str = "orysys-fast"
    llm_timeout_seconds: float = 60

    # Fallback LLM (direct provider, used when the gateway is down)
    fallback_llm_base_url: str = ""
    fallback_llm_api_key: SecretStr = SecretStr("")
    fallback_llm_model: str = ""

    # Pinecone
    pinecone_api_key: SecretStr = SecretStr("")
    pinecone_index: str = "orysys-kb"

    # LangSmith
    langsmith_api_key: SecretStr = SecretStr("")
    langsmith_project: str = "orysys-ai-assistant"
    langsmith_tracing: bool = True

    # Rate limiting
    rate_limit_capacity: int = 10
    rate_limit_refill_per_sec: float = 0.2

    # MCP
    mcp_server_url: str = "http://localhost:8765/mcp"

    # Health checks
    health_check_timeout_seconds: float = 6.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
