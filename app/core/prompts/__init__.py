"""Agent prompts package.

The ``.md`` files in this directory are the source of truth for each agent's
system prompt. Use :func:`load_prompt` to read them at runtime so edits to
the Markdown take effect on the next call (the result is cached for the
lifetime of the process).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent


class PromptNotFoundError(FileNotFoundError):
    pass


@lru_cache
def load_prompt(name: str) -> str:
    """Load a prompt by name (without extension).

    Example: ``load_prompt("planner")`` -> contents of ``planner.md``.
    """
    path = _PROMPTS_DIR / f"{name}.md"
    if not path.is_file():
        raise PromptNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def prompts_dir() -> Path:
    """Return the directory holding prompt files (for tooling/debug)."""
    return _PROMPTS_DIR


__all__ = ["PromptNotFoundError", "load_prompt", "prompts_dir"]
