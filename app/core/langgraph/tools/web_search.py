"""Serper web search client.

Thin async wrapper around https://serper.dev/. Returns typed results so the
Researcher agent doesn't have to deal with raw dicts.

API reference (POST /search):
    Headers:  X-API-KEY: <key>, Content-Type: application/json
    Body:     {"q": "<query>", "num": 10, "gl": "us", "hl": "en"}
    Response: {"organic": [{"title", "link", "snippet", "position"}, ...], ...}
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, HttpUrl
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import Settings
from app.core.config import settings as default_settings

logger = logging.getLogger(__name__)


class SerperError(RuntimeError):
    """Raised for Serper API failures we cannot recover from."""


class SerperSearchResult(BaseModel):
    """One organic search result from Serper."""

    model_config = ConfigDict(extra="ignore")

    title: str
    link: HttpUrl = Field(..., description="Result URL")
    snippet: str = ""
    position: int = 0


class SerperClient:
    """Async client for Serper search.

    Designed to be reused (one client per process). Holds an internal
    ``httpx.AsyncClient`` that should be closed via :meth:`aclose` or as an
    async context manager.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings or default_settings
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self._settings.serper_base_url,
            timeout=self._settings.serper_timeout_seconds,
            headers={
                "X-API-KEY": self._settings.serper_api_key,
                "Content-Type": "application/json",
            },
        )

    async def __aenter__(self) -> SerperClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    )
    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self._settings.serper_api_key:
            raise SerperError("SERPER_API_KEY is not set. Add it to your environment or .env file.")
        response = await self._client.post(path, json=payload)
        # Retry on 5xx, fail fast on 4xx.
        if 500 <= response.status_code < 600:
            response.raise_for_status()
        if response.status_code >= 400:
            raise SerperError(
                f"Serper request failed ({response.status_code}): {response.text[:300]}"
            )
        return response.json()

    async def search(
        self,
        query: str,
        *,
        num: int = 10,
        gl: str = "us",
        hl: str = "en",
    ) -> list[SerperSearchResult]:
        """Run a search and return organic results.

        Args:
            query: Search query string.
            num: Number of results to request (Serper caps this).
            gl: Country code (geo-localization).
            hl: Interface language.
        """
        query = query.strip()
        if not query:
            return []

        payload = {"q": query, "num": num, "gl": gl, "hl": hl}
        logger.debug("Serper search: %r (num=%d)", query, num)
        try:
            data = await self._post("/search", payload)
        except (httpx.HTTPError, SerperError) as exc:
            logger.warning("Serper search failed for %r: %s", query, exc)
            return []

        organic = data.get("organic") or []
        results: list[SerperSearchResult] = []
        for idx, item in enumerate(organic):
            try:
                results.append(
                    SerperSearchResult(
                        title=item.get("title", "").strip() or "(untitled)",
                        link=item["link"],
                        snippet=(item.get("snippet") or "").strip(),
                        position=int(item.get("position", idx + 1)),
                    )
                )
            except (KeyError, ValueError) as exc:
                logger.debug("Skipping malformed Serper result: %s", exc)
                continue

        return results
