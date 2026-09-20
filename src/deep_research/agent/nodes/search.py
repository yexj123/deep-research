"""search node: one arXiv search for the question (milestone 2: a single subtopic)."""

from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from deep_research.agent.config import ARXIV_MAX_RESULTS
from deep_research.agent.sources.arxiv import build_search_query, search_arxiv
from deep_research.agent.state import ResearchState

SearchNode = Callable[[ResearchState], Awaitable[dict[str, Any]]]


def make_search(http_client: httpx.AsyncClient) -> SearchNode:
    """Build the search node with its HTTP client captured in a closure (D-049)."""

    async def search(state: ResearchState) -> dict[str, Any]:
        # Catches nothing: at milestone 2 a failed search fails the run (D-053).
        query = build_search_query(state.question)
        result = await search_arxiv(
            client=http_client,
            query=query,
            max_results=ARXIV_MAX_RESULTS,
        )
        return {
            "sources": list(result.sources),
            "skipped_entries": result.skipped,
        }

    return search
