"""research_worker node: one arXiv search per subtopic, run in parallel via Send (D-072)."""

from collections.abc import Awaitable, Callable
from typing import Any, TypedDict
from xml.etree.ElementTree import ParseError

import httpx
import pydantic
from defusedxml import DefusedXmlException
from langgraph.config import get_stream_writer

from deep_research.agent.config import ARXIV_MAX_RESULTS
from deep_research.agent.sources.arxiv import ArxivAPIError, build_search_query, search_arxiv
from deep_research.agent.sources.rate_limit import ArxivRateLimiter


class SubtopicTask(TypedDict):
    """The Send payload. A TypedDict, because Send payloads are checkpointed too (D-071)."""

    subtopic: str
    seen_paper_ids: set[str]


WorkerNode = Callable[[SubtopicTask], Awaitable[dict[str, Any]]]

# Bad external data, not bugs: each becomes a failed subtopic (D-048, D-053, D-072).
EXTERNAL_FAILURES = (
    httpx.TransportError,
    pydantic.ValidationError,
    ParseError,
    DefusedXmlException,
    ArxivAPIError,
)


def _progress_writer() -> Callable[[dict[str, Any]], None]:
    """The `custom` stream writer, or a no-op when there isn't one (O-5).

    `get_stream_writer()` raises RuntimeError outside a runnable context -- confirmed
    2026-09-21: "Called get_config outside of a runnable context". Two reasons this falls back
    rather than propagating:

    - **Progress must never be able to break a run.** A node whose research fails because its
      telemetry could not initialize is badly designed. Reporting is strictly additive.
    - It keeps the worker callable directly, so its contract stays testable without graph
      scaffolding -- which is what test_research_worker.py exists to do.

    The only cause of that RuntimeError is "not running inside a graph", so nothing diagnostic
    is being swallowed.
    """
    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda _: None


def _is_external_status(exc: httpx.HTTPStatusError) -> bool:
    """429 and 5xx are arXiv's problem; every other 4xx is ours (D-065)."""
    status = exc.response.status_code
    return status == 429 or status >= 500


def make_research_worker(
    http_client: httpx.AsyncClient, limiter: ArxivRateLimiter
) -> WorkerNode:
    """Build the worker with its client and limiter captured in a closure (D-032, D-049, D-064)."""

    async def research_worker(task: SubtopicTask) -> dict[str, Any]:
        subtopic = task["subtopic"]
        # Live progress for a long run (O-5). Derived from the same facts as the state
        # update below, so the two cannot drift: one place decides what happened.
        writer = _progress_writer()
        writer({"status": f"Searching arXiv for {subtopic!r}"})
        # Outside the try on purpose: decompose guarantees the subtopic is searchable (D-073),
        # so a ValueError here is a bug and must crash rather than burn the retry cap.
        query = build_search_query(subtopic)

        try:
            async with limiter:  # held across the request, not just its start (D-064)
                result = await search_arxiv(http_client, query, ARXIV_MAX_RESULTS)
        except httpx.HTTPStatusError as exc:
            if not _is_external_status(exc):
                raise
            writer({"status": f"Search failed for {subtopic!r}"})
            return {"failed_subtopics": [subtopic]}
        except EXTERNAL_FAILURES:
            writer({"status": f"Search failed for {subtopic!r}"})
            return {"failed_subtopics": [subtopic]}

        found = len(result.sources)
        writer(
            {
                "status": f"Found {found} paper(s) for {subtopic!r}"
                if found
                else f"No papers found for {subtopic!r}"
            }
        )

        update: dict[str, Any] = {
            "sources": list(result.sources),
            "skipped_entries": result.skipped,  # this worker's delta, never a total (D-067)
            "explored_subtopics": [subtopic],  # success only (D-018)
            "seen_paper_ids": {source.arxiv_id for source in result.sources},
        }
        if not found:
            # Explored, but nothing published: a finding worth reporting, not a silent
            # no-op (D-021, O-5).
            update["empty_subtopics"] = [subtopic]
        return update

    return research_worker