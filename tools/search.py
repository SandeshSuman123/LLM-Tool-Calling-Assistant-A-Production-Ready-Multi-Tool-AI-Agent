"""
tools/search.py
=================

WHY THIS FILE EXISTS
---------------------
LLMs have a training cutoff and no knowledge of events after it. A web
search tool lets the assistant answer questions about current events,
recent releases, or anything outside the model's training data — one of
the highest-value real-world use cases for tool calling.

RESPONSIBILITY
---------------
Perform a lightweight web search and return the top N result titles,
snippets, and URLs.

By default this uses DuckDuckGo's HTML endpoint, which requires NO API
key — ideal for a free/portfolio project. If `SEARCH_API_KEY` is set in
`.env`, you can swap in a provider like Tavily or Serper (see the
`_search_with_provider_api` stub note below).

CONCEPTS DEMONSTRATED
-----------------------
✓ Tool Functions
✓ Tool Schemas
✓ Error Handling
"""

from __future__ import annotations

from typing import Any, Dict, List

import requests
from pydantic import BaseModel, Field

from agent.schemas import ToolResult, pydantic_to_tool_schema
from config import settings
from utils.errors import ToolExecutionError
from utils.logger import get_logger

logger = get_logger(__name__)

_DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/"


class SearchArgs(BaseModel):
    """Arguments for the web search tool."""

    query: str = Field(..., description="The search query, e.g. 'latest Python 3.13 features'.")
    max_results: int = Field(5, ge=1, le=10, description="Maximum number of results to return.")


def _parse_duckduckgo_html(html: str, max_results: int) -> List[Dict[str, str]]:
    """
    Extract result titles + links from DuckDuckGo's HTML search response.

    WHY THIS EXISTS: DuckDuckGo's free HTML endpoint doesn't return JSON,
    so we do a minimal, dependency-light regex-based extraction rather
    than pulling in a full HTML parser for a handful of fields.
    """
    import re

    results: List[Dict[str, str]] = []
    # Each result link looks like: <a rel="nofollow" class="result__a" href="...">Title</a>
    pattern = re.compile(
        r'<a[^>]*class="result__a"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>',
        re.DOTALL,
    )
    for match in pattern.finditer(html):
        title = re.sub(r"<.*?>", "", match.group("title")).strip()
        url = match.group("url")
        if title and url:
            results.append({"title": title, "url": url})
        if len(results) >= max_results:
            break
    return results


def run(args: SearchArgs) -> ToolResult:
    """
    Perform a web search for `args.query` and return the top results.
    """
    logger.info("Search tool invoked with query=%r", args.query)
    try:
        response = requests.post(
            _DUCKDUCKGO_HTML_URL,
            data={"q": args.query},
            headers={"User-Agent": "Mozilla/5.0 (compatible; AIPersonalAssistant/1.0)"},
            timeout=settings.REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        results = _parse_duckduckgo_html(response.text, args.max_results)

        if not results:
            raise ToolExecutionError("No search results found for this query.")

        return ToolResult(success=True, data={"query": args.query, "results": results})

    except ToolExecutionError as exc:
        return ToolResult(success=False, error=str(exc))
    except requests.exceptions.RequestException as exc:
        logger.exception("Search request failed")
        return ToolResult(success=False, error=f"Search service unavailable: {exc}")


TOOL_SCHEMA = pydantic_to_tool_schema(
    name="web_search",
    description=(
        "Search the web for current information, news, or facts that may be "
        "beyond the model's training data. Returns a list of titles and URLs."
    ),
    args_model=SearchArgs,
)
