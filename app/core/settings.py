"""Backwards-compatibility shim.

An earlier Worker B draft exposed three distinct ``BaseSettings`` classes
(``LLMSettings``, ``SerperSettings``, ``PipelineSettings``) and getter
helpers. The settings have since been consolidated into the unified
:class:`app.core.config.Settings`. This module re-exports thin wrappers
under the old names so any code (and tests) written against the original
API keep working.

Prefer ``from app.core.config import settings`` in new code.
"""

from __future__ import annotations

from functools import lru_cache

from app.core.config import Settings


# Old class names — they're all aliases for the unified Settings now. Each
# alias preserves the constructor surface so calls like
# ``PipelineSettings(MAX_RESEARCH_TASKS=3)`` continue to work.
class LLMSettings(Settings):
    """Alias for the LLM-related slice of :class:`Settings`."""


class SerperSettings(Settings):
    """Alias for the Serper-related slice of :class:`Settings`."""


class PipelineSettings(Settings):
    """Alias for the pipeline-limits slice of :class:`Settings`."""


@lru_cache
def get_llm_settings() -> LLMSettings:
    return LLMSettings()


@lru_cache
def get_serper_settings() -> SerperSettings:
    return SerperSettings()


@lru_cache
def get_pipeline_settings() -> PipelineSettings:
    return PipelineSettings()


__all__ = [
    "LLMSettings",
    "PipelineSettings",
    "SerperSettings",
    "get_llm_settings",
    "get_pipeline_settings",
    "get_serper_settings",
]
