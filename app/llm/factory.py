"""LLM factory: the only place that knows how models are reached.

All agents call `get_llm("strong")` or `get_llm("fast")`. The gateway (9router) is
OpenAI-compatible, so swapping it for LiteLLM or a direct provider is a config change.

- strong: supervisor routing and the final answer (quality matters)
- fast:   RLM batch workers, guards, summaries (many small calls, cost matters)
"""

from functools import lru_cache
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.config import get_settings

Tier = Literal["strong", "fast"]


def _gateway_llm(model: str, temperature: float, streaming: bool) -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        base_url=s.llm_base_url,
        api_key=s.llm_api_key.get_secret_value() or "not-set",
        model=model,
        temperature=temperature,
        timeout=s.llm_timeout_seconds,
        max_retries=s.llm_max_retries,  # retries with backoff inside the OpenAI client
        streaming=streaming,
    )


def _fallback_llm(temperature: float, streaming: bool) -> ChatOpenAI | None:
    """Direct provider used when the gateway itself is down. Optional."""
    s = get_settings()
    if not (s.fallback_llm_base_url and s.fallback_llm_model and s.fallback_llm_api_key.get_secret_value()):
        return None
    return ChatOpenAI(
        base_url=s.fallback_llm_base_url,
        api_key=s.fallback_llm_api_key.get_secret_value(),
        model=s.fallback_llm_model,
        temperature=temperature,
        timeout=s.llm_timeout_seconds,
        max_retries=1,
        streaming=streaming,
    )


@lru_cache
def get_llm(tier: Tier = "strong", temperature: float = 0.0, streaming: bool = False) -> BaseChatModel:
    s = get_settings()
    model = s.llm_model_strong if tier == "strong" else s.llm_model_fast
    primary = _gateway_llm(model, temperature, streaming)
    fallback = _fallback_llm(temperature, streaming)
    # with_fallbacks: if the gateway call fails after its retries, the same request goes to the fallback.
    return primary.with_fallbacks([fallback]) if fallback else primary
