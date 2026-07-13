# LOCKED BY Worker B

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from app.core.config import Settings
from app.core.config import settings as default_settings

from langchain_openai import ChatOpenAI


class LLMConfigError(RuntimeError):
    pass


def _build_llm(
    settings: Settings,
    *,
    temperature: float | None,
    max_tokens: int | None,
):
    if not settings.llm_api_key:
        raise LLMConfigError("LLM_API_KEY is not set. Add it to your environment or .env file.")

    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        max_tokens=max_tokens if max_tokens is not None else settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
        max_retries=2,
    )


def get_llm(
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    settings: Settings | None = None,
):
    cfg = settings or default_settings
    return _build_llm(cfg, temperature=temperature, max_tokens=max_tokens)


@lru_cache
def get_default_llm():
    return get_llm()


def reset_llm_cache() -> None:
    get_default_llm.cache_clear()
    LLMFactory.reset()


class LLMFactory:
    _instance: Optional[ChatOpenAI] = None
    _instance_temperature: float | None = None
    _instance_model: str | None = None

    @classmethod
    def get_llm(cls, temperature: float | None = None) -> ChatOpenAI:
        cfg = default_settings
        model = cfg.llm_model
        if (cls._instance is None or
            cls._instance_temperature != temperature or
            cls._instance_model != model):
            cls._instance = get_llm(temperature=temperature)
            cls._instance_temperature = temperature
            cls._instance_model = model
        return cls._instance

    @classmethod
    def get_structured_llm(cls, schema, temperature: float | None = None):
        return cls.get_llm(temperature=temperature).with_structured_output(schema)

    @classmethod
    def reset(cls) -> None:
        cls._instance = None
        cls._instance_temperature = None