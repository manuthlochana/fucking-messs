"""External web-search client for real, source-backed defect mining.

Phase 3 of the forensic pipeline must cite *real* corroborating sources, not
LLM-hallucinated ones. This module wraps three interchangeable providers behind
one async API:

* **Tavily**   — ``https://api.tavily.com/search`` (POST, JSON body).
* **Brave**    — ``https://api.search.brave.com/res/v1/web/search`` (GET).
* **SerpAPI**  — ``https://serpapi.com/search.json`` (GET, Google engine).

The provider is auto-selected from whichever API key is present (or forced via
``SEARCH_PROVIDER``). When no key is configured, :func:`search` returns an empty
list and the caller is expected to log ``search_skipped_no_api_key`` and degrade
gracefully — never to invent sources.

Dependency-light: uses ``httpx`` (already a project dep). Never raises on a
network/provider error — a failed search yields ``[]`` so the pipeline continues.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import Settings, settings as default_settings
from logging_utils import log


@dataclass
class SearchResult:
    """One normalized search hit, provider-agnostic."""

    title: str = ""
    url: str = ""
    snippet: str = ""
    source: str = ""  # provider name (tavily/brave/serpapi)

    def as_dict(self) -> Dict[str, str]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
        }


@dataclass
class SearchClient:
    """Async multi-provider search client with graceful no-key degradation."""

    settings: Settings = field(default_factory=lambda: default_settings)

    @property
    def provider(self) -> Optional[str]:
        return self.settings.active_search_provider

    @property
    def enabled(self) -> bool:
        return self.provider is not None

    async def search(self, query: str, max_results: Optional[int] = None) -> List[SearchResult]:
        """Run a web search and return up to ``max_results`` normalized hits.

        Returns ``[]`` when no provider key is configured or the call fails.
        """
        provider = self.provider
        n = max_results or self.settings.search_max_results
        if provider is None:
            log.debug("[Search] search_skipped_no_api_key — no provider key configured.")
            return []
        try:
            import httpx

            async with httpx.AsyncClient(timeout=15.0) as client:
                if provider == "tavily":
                    return await self._tavily(client, query, n)
                if provider == "brave":
                    return await self._brave(client, query, n)
                if provider == "serpapi":
                    return await self._serpapi(client, query, n)
        except Exception as exc:  # network / provider / parse error
            log.warn(f"[Search] provider={provider} query={query!r} failed: {exc}")
        return []

    # ------------------------------------------------------------------ #
    # Provider adapters
    # ------------------------------------------------------------------ #
    async def _tavily(self, client, query: str, n: int) -> List[SearchResult]:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": self.settings.tavily_api_key,
                "query": query,
                "max_results": n,
                "search_depth": "basic",
                "include_answer": False,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        out: List[SearchResult] = []
        for r in (data.get("results") or [])[:n]:
            out.append(
                SearchResult(
                    title=str(r.get("title") or ""),
                    url=str(r.get("url") or ""),
                    snippet=str(r.get("content") or r.get("snippet") or ""),
                    source="tavily",
                )
            )
        return out

    async def _brave(self, client, query: str, n: int) -> List[SearchResult]:
        resp = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": n},
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self.settings.brave_api_key or "",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        results = ((data.get("web") or {}).get("results")) or []
        out: List[SearchResult] = []
        for r in results[:n]:
            out.append(
                SearchResult(
                    title=str(r.get("title") or ""),
                    url=str(r.get("url") or ""),
                    snippet=str(r.get("description") or ""),
                    source="brave",
                )
            )
        return out

    async def _serpapi(self, client, query: str, n: int) -> List[SearchResult]:
        resp = await client.get(
            "https://serpapi.com/search.json",
            params={
                "q": query,
                "engine": "google",
                "num": n,
                "api_key": self.settings.serpapi_api_key or "",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        out: List[SearchResult] = []
        for r in (data.get("organic_results") or [])[:n]:
            out.append(
                SearchResult(
                    title=str(r.get("title") or ""),
                    url=str(r.get("link") or ""),
                    snippet=str(r.get("snippet") or ""),
                    source="serpapi",
                )
            )
        return out


def build_defect_queries(brand: Optional[str], canonical_name: str,
                         model_number: Optional[str] = None) -> List[str]:
    """Construct the real defect-mining search queries for a product."""
    ident = (model_number or canonical_name or "").strip()
    brand = (brand or "").strip()
    name = canonical_name.strip()
    queries = []
    if brand or ident:
        queries.append(f"{brand} {ident} hardware problems site:reddit.com".strip())
    if name:
        queries.append(f"{name} known defects issues teardown".strip())
    # De-dup while preserving order; drop empties.
    seen: set = set()
    out: List[str] = []
    for q in queries:
        q = " ".join(q.split())
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out
