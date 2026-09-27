"""
agent/tools.py

Web-search fallback tool for the agentic RAG pipeline (M5). Used only
after retrieval + retries have failed to find enough relevant chunks
in the local Chroma store.

Scoped to the target docs site (and GitHub, for issues/discussions)
rather than the open web, so a fallback answer still traces back to
project-relevant sources rather than the general internet.

Prefers Tavily (needs TAVILY_API_KEY, free tier: 1,000 credits/month).
Falls back to a no-key DuckDuckGo scrape if TAVILY_API_KEY is unset, per
the project's env var documentation.
"""

import os
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

TARGET_DOCS_URL = os.getenv("TARGET_DOCS_URL", "https://fastapi.tiangolo.com/")
_docs_domain = urlparse(TARGET_DOCS_URL).netloc  # e.g. "fastapi.tiangolo.com"
SEARCH_DOMAINS = [_docs_domain, "github.com"]
MAX_RESULTS = 5


def _search_tavily(query: str) -> list[dict]:
    from tavily import TavilyClient

    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    response = client.search(
        query=query,
        include_domains=SEARCH_DOMAINS,
        max_results=MAX_RESULTS,
        search_depth="basic",
    )
    return [
        {
            "text": r.get("content", ""),
            "url": r.get("url", ""),
            "title": r.get("title", r.get("url", "")),
            "section": "(web search result)",
        }
        for r in response.get("results", [])
    ]


def _search_duckduckgo(query: str) -> list[dict]:
    from duckduckgo_search import DDGS

    domain_filter = " OR ".join(f"site:{d}" for d in SEARCH_DOMAINS)
    scoped_query = f"{query} ({domain_filter})"

    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(scoped_query, max_results=MAX_RESULTS):
            results.append({
                "text": r.get("body", ""),
                "url": r.get("href", ""),
                "title": r.get("title", r.get("href", "")),
                "section": "(web search result)",
            })
    return results


def web_search(query: str) -> list[dict]:
    """Returns chunk-shaped dicts (text/url/title/section) from either
    Tavily (preferred) or a DuckDuckGo scrape (no-key fallback), scoped
    to the target docs site and GitHub."""
    if os.getenv("TAVILY_API_KEY"):
        try:
            results = _search_tavily(query)
            if results:
                return results
        except Exception as e:
            print(f"  [web_search] Tavily failed, falling back to DuckDuckGo: {e}")

    try:
        return _search_duckduckgo(query)
    except Exception as e:
        print(f"  [web_search] DuckDuckGo fallback also failed: {e}")
        return []