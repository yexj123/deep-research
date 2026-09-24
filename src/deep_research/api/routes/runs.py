"""Run routes: create a run, stream it, read a finished review (D-081, D-083)."""

import json
import uuid
from dataclasses import asdict
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from deep_research.agent.context import ProviderType
from deep_research.agent.coverage import summarize_coverage
from deep_research.agent.runner import RunState, get_review, get_run_state, stream_run
from deep_research.api.rendering import render_coverage, render_review
from deep_research.persistence.runs import Run, get_run, list_runs, record_run

router = APIRouter(prefix="/runs", tags=["runs"])


class CreateRun(BaseModel):
    """The request body. Pydantic at the boundary, as everywhere else (D-013)."""

    question: str
    provider: ProviderType = "openai"


class RunCreated(BaseModel):
    thread_id: str


def _sse(event: str, data: dict[str, Any]) -> str:
    """One Server-Sent Event. Hand-rolled framing, no dependency (D-083).

    The blank line is what terminates an event -- omit it and the browser buffers forever.
    """
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _event_for(chunk: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """Translate a LangGraph chunk into a browser event, or None to skip it.

    Deliberately *not* a passthrough of the raw update: those carry `Source` objects, which
    aren't JSON-serializable, and the browser has no use for full state. What it needs is a
    progress trail and the review text.
    """
    match chunk["type"]:
        case "updates":
            # One event per finished node: the "decompose -> searching -> checking" trail.
            node, update = next(iter(chunk["data"].items()))
            return "node", {"node": node, "fields": sorted(update or {})}
        case "custom":
            # Live status a node pushed itself (O-5): mid-search progress that no node
            # *finishing* would convey -- "searching X", "no papers found for X".
            return "progress", {"message": chunk["data"].get("status", "")}
        case "messages":
            message_chunk, metadata = chunk["data"]
            # Only the review streams to the reader. decompose also emits tokens from
            # milestone 3 onward (its JSON plan) -- unfiltered, that lands in the user's
            # review pane as raw JSON.
            if metadata.get("langgraph_node") != "synthesize":
                return None
            return "token", {"text": message_chunk.text}
        case _:
            return None


@router.post("", response_model=RunCreated, status_code=201)
async def create_run(request: Request, body: CreateRun) -> RunCreated:
    """Record a run and hand back its thread_id. Executes nothing (D-081)."""
    question = body.question.strip()
    if not question:
        # intake would reject this too (D-033), but failing here saves a thread_id and a
        # round trip for something we can see immediately.
        raise HTTPException(status_code=422, detail="question must not be empty")

    thread_id = str(uuid.uuid4())
    await record_run(request.app.state.conn, thread_id, question, body.provider)
    return RunCreated(thread_id=thread_id)


@router.get("")
async def list_recent_runs(request: Request) -> list[Run]:
    """Run history, newest first."""
    return await list_runs(request.app.state.conn)


@router.get("/{thread_id}")
async def read_run(request: Request, thread_id: str) -> dict[str, Any]:
    """The current state of a run: its question, its status, and the review once it exists."""
    run = await get_run(request.app.state.conn, thread_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"no run {thread_id}")

    graph = request.app.state.graph
    values = await get_review(graph, thread_id)
    return {
        "thread_id": thread_id,
        "question": run.question,
        "provider": run.provider,
        "status": (await get_run_state(graph, thread_id)).value,
        "review": values.get("review", ""),
        "citation_violations": values.get("citation_violations", []),
        "coverage": asdict(summarize_coverage(values)),
    }


@router.get("/{thread_id}/stream")
async def stream(request: Request, thread_id: str) -> StreamingResponse:
    """Drive the run and stream it as SSE (D-081, D-083).

    Starts, resumes or replays depending on the checkpoint -- `stream_run` owns that decision.
    Closing the connection closes this generator, which leaves the checkpoint resumable, so
    reconnecting to the same thread_id continues rather than restarting.
    """
    run = await get_run(request.app.state.conn, thread_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"no run {thread_id}")

    graph = request.app.state.graph

    async def events() -> AsyncIterator[str]:
        async for chunk in stream_run(graph, thread_id, run.question, run.provider):
            event = _event_for(chunk)
            if event is not None:
                yield _sse(*event)

        # Always terminate with the finished review, whether this call produced it or a
        # previous one did (the FINISHED case yields no chunks at all).
        values = await get_review(graph, thread_id)
        review = values.get("review", "")
        coverage = summarize_coverage(values)
        yield _sse(
            "done",
            {
                "thread_id": thread_id,
                "review": review,
                # Rendered and escaped on the server (O-8): the page assigns this to
                # innerHTML, so the browser must never be handed model-authored markdown.
                "review_html": render_review(review),
                # What the run covered and did NOT cover, rendered separately so the model's
                # prose stays exactly what it wrote -- and so the streamed text still equals
                # `review` (D-037's integration assertion). Always present since O-15: the
                # stop reason is the answer to "why only one round?", which D-096 made the
                # obvious question and which `is_complete` used to hide on exactly the runs
                # that prompt it.
                "coverage_html": render_coverage(coverage),
                "citation_violations": values.get("citation_violations", []),
            },
        )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        # Without this a proxy may buffer the whole stream and deliver it at the end,
        # which looks exactly like the agent hanging.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
