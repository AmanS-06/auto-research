"""Tests for the Serper web search client."""

from __future__ import annotations

import httpx
import pytest
import respx

from app.core.config import Settings
from app.core.langgraph.tools.web_search import (
    SerperClient,
    SerperSearchResult,
)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        serper_api_key="test-key",
        serper_base_url="https://google.serper.dev",
        serper_timeout_seconds=5.0,
    )


@pytest.mark.asyncio
async def test_search_parses_organic_results(settings):
    sample = {
        "organic": [
            {
                "title": "Result A",
                "link": "https://example.com/a",
                "snippet": "snippet a",
                "position": 1,
            },
            {
                "title": "Result B",
                "link": "https://example.com/b",
                "snippet": "snippet b",
                "position": 2,
            },
        ]
    }

    with respx.mock(base_url=settings.serper_base_url) as router:
        router.post("/search").respond(200, json=sample)
        async with SerperClient(settings) as client:
            results = await client.search("anything")

    assert len(results) == 2
    assert all(isinstance(r, SerperSearchResult) for r in results)
    assert str(results[0].link) == "https://example.com/a"
    assert results[1].position == 2


@pytest.mark.asyncio
async def test_search_returns_empty_on_blank_query(settings):
    async with SerperClient(settings) as client:
        results = await client.search("   ")
    assert results == []


@pytest.mark.asyncio
async def test_search_returns_empty_on_404(settings):
    with respx.mock(base_url=settings.serper_base_url) as router:
        router.post("/search").respond(404, text="not found")
        async with SerperClient(settings) as client:
            results = await client.search("q")

    assert results == []


@pytest.mark.asyncio
async def test_search_retries_on_5xx_then_succeeds(settings):
    sample = {"organic": [{"title": "x", "link": "https://x.example", "snippet": ""}]}

    with respx.mock(base_url=settings.serper_base_url) as router:
        route = router.post("/search")
        route.side_effect = [
            httpx.Response(500, text="boom"),
            httpx.Response(500, text="boom"),
            httpx.Response(200, json=sample),
        ]
        async with SerperClient(settings) as client:
            results = await client.search("q")

    assert len(results) == 1
    assert route.call_count == 3


@pytest.mark.asyncio
async def test_search_returns_empty_when_api_key_missing():
    settings = Settings(serper_api_key="")
    async with SerperClient(settings) as client:
        results = await client.search("q")
    assert results == []


@pytest.mark.asyncio
async def test_search_skips_malformed_results(settings):
    bad = {
        "organic": [
            {"title": "ok", "link": "https://ok.example", "snippet": "s"},
            {"title": "missing link", "snippet": "s"},
            {"title": "bad url", "link": "not-a-url", "snippet": "s"},
        ]
    }
    with respx.mock(base_url=settings.serper_base_url) as router:
        router.post("/search").respond(200, json=bad)
        async with SerperClient(settings) as client:
            results = await client.search("q")

    # Only the first one is well-formed.
    assert len(results) == 1
    assert str(results[0].link) == "https://ok.example/"
