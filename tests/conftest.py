"""Shared pytest fixtures for Worker B tests.

The agents in this package use ``llm.with_structured_output(Schema)`` and
then ``await structured.ainvoke(messages)`` to get a typed Pydantic object
back. We don't want to call a real LLM in tests, so this module defines a
``FakeStructuredLLM`` that returns canned Pydantic outputs in order.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel


class _StructuredRunnable:
    """Mimics the runnable returned by ``BaseChatModel.with_structured_output``.

    Pulls from a *shared* FIFO so multiple runnables created from the same
    ``FakeStructuredLLM`` consume the same queue of canned outputs in order.
    """

    def __init__(self, shared_outputs: list[BaseModel | Exception], calls: list[Any]):
        self._outputs = shared_outputs
        self._calls = calls

    async def ainvoke(self, messages, *args, **kwargs):
        self._calls.append(messages)
        if not self._outputs:
            raise AssertionError("FakeStructuredLLM ran out of canned outputs")
        nxt = self._outputs.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    def invoke(self, messages, *args, **kwargs):
        self._calls.append(messages)
        if not self._outputs:
            raise AssertionError("FakeStructuredLLM ran out of canned outputs")
        nxt = self._outputs.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class FakeStructuredLLM:
    """Drop-in replacement for a chat model in agent tests.

    Only implements the surface our agents touch:
        * ``with_structured_output(schema)`` -> awaitable runnable

    All runnables created by this fake share one FIFO of canned outputs,
    so an agent that calls ``with_structured_output`` once per sub-task
    (e.g. the Researcher) will consume one output per call.
    """

    def __init__(self, outputs: list[BaseModel | Exception] | None = None):
        self._outputs: list[BaseModel | Exception] = list(outputs or [])
        self.invocations: list[Any] = []
        self.schemas: list[Any] = []

    def queue(self, output: BaseModel | Exception) -> None:
        self._outputs.append(output)

    @property
    def remaining(self) -> int:
        return len(self._outputs)

    def with_structured_output(self, schema, **_kwargs):
        self.schemas.append(schema)
        return _StructuredRunnable(self._outputs, self.invocations)


@pytest.fixture
def fake_llm() -> FakeStructuredLLM:
    """A fresh fake LLM with no queued outputs."""
    return FakeStructuredLLM()


@pytest.fixture
def async_mock() -> AsyncMock:
    return AsyncMock()
