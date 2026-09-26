"""Run routes: create a run, stream it, read a finished review (D-081, D-083)."""

import json
import logging
import re
import uuid
from dataclasses import asdict
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from deep_research.agent.config import MODEL_NAME_PATTERN
from deep_research.agent.context import ProviderType
from deep_research.agent.coverage import summarize_coverage
from deep_research.agent.runner import RunState, get_review, get_run_state, stream_run
from deep_research.api.rendering import render_coverage, render_review
from deep_research.persistence.runs import (
    Run,
    delete_session,
    get_run,
    list_runs,
    list_session,
    record_run,
    rename_session,
    set_starred,
)

router = APIRouter(prefix="/runs", tags=["runs"])

logger = logging.getLogger(__name__)


class CreateRun(BaseModel):
    """The request body. Pydantic at the boundary, as everywhere else (D-013)."""

    question: str
    provider: ProviderType = "openai"
    # The exact model to use, chosen or typed by the user (D-125, D-126). `None` means the
    # provider's default, so a client that never sends this keeps working unchanged.
    #
    # Validated here as well as in `intake`: this turns a malformed name into a 422 the browser
    # can show, instead of a 500 from a run that started and then raised.
    model: str | None = Field(default=None, max_length=100)
    # The run this question follows up on, if any (D-121). A follow-up gets its own
    # thread and clean graph state; only its wording is inherited.
    follow_up_to: str | None = None

    @field_validator("model")
    @classmethod
    def _model_name_is_usable(cls, value: str | None) -> str | None:
        """Sanitize, then check the shape. **Not** checked against RECOMMENDED_MODELS.

        That list is what the UI offers, not what the API permits (D-126): a model released
        tomorrow must work by typing it, which is the whole point. What this rejects is
        whitespace, control characters and anything long enough to be a payload rather than a
        name -- input hygiene, not a policy on which models exist.
        """
        cleaned = (value or "").strip()
        if not cleaned:
            # "", "   " and an omitted field are the same fact; collapse them here so nothing
            # downstream has to tell three absent values apart.
            return None
        if not re.fullmatch(MODEL_NAME_PATTERN, cleaned):
            raise ValueError(
                f"{cleaned!r} is not a usable model name. Expected something like "
                "'gpt-4o-mini' or 'deepseek-chat': letters, digits, and . _ - : / only."
            )
        return cleaned


class RunCreated(BaseModel):
    thread_id: str
    # The question as stored, which for a follow-up is the *rewritten* one (D-121). Returned
    # so the page can show what it resolved to without a second request -- a rewrite the
    # reader can't see is a rewrite they can't trust.
    question: str


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
    """Record a run and hand back its thread_id. Executes nothing (D-081).

    With `follow_up_to`, this turn joins that turn's session and its question is rewritten to
    stand on its own (D-121). The rewrite happens **here**, before the thread exists, so what
    gets stored and later streamed is the standalone question -- one code path through the
    graph, and a history entry that reads as a real question rather than "What about it?".
    """
    question = body.question.strip()
    if not question:
        # intake would reject this too (D-033), but failing here saves a thread_id and a
        # round trip for something we can see immediately.
        raise HTTPException(status_code=422, detail="question must not be empty")

    parent = None
    if body.follow_up_to:
        parent = await get_run(request.app.state.conn, body.follow_up_to)
        if parent is None:
            raise HTTPException(status_code=404, detail=f"no run {body.follow_up_to}")
        values = await get_review(request.app.state.graph, parent.thread_id)
        review = values.get("review", "")
        if not review:
            # Following up on a run with no answer yet would rewrite against nothing, and the
            # rewriter would invent the context it was missing.
            raise HTTPException(
                status_code=409, detail="that run has no review to follow up on yet"
            )
        question = await request.app.state.rewrite_follow_up(
            question, parent.question, review, body.provider, body.model
        )

    thread_id = str(uuid.uuid4())
    await record_run(
        request.app.state.conn, thread_id, question, body.provider, parent, body.model
    )
    return RunCreated(thread_id=thread_id, question=question)


class UpdateSession(BaseModel):
    """A sidebar edit. Both fields optional: a request may set either, or both (D-127)."""

    model_config = {"extra": "forbid"}  # an unknown field is a contract change (D-061)

    # "" clears the title back to the question, which is how a rename is undone -- so this
    # has to distinguish "not sent" (None) from "sent as empty" (""), and cannot use "" as
    # its own absent value the way `model` does.
    title: str | None = Field(default=None, max_length=200)
    starred: bool | None = None


async def _session_of(request: Request, thread_id: str) -> str:
    """The session a thread belongs to, or 404. Shared by the two edit routes."""
    run = await get_run(request.app.state.conn, thread_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"no run {thread_id}")
    return run.session_id or run.thread_id


@router.patch("/{thread_id}/session", status_code=204)
async def update_session(request: Request, thread_id: str, body: UpdateSession) -> Response:
    """Rename or star the conversation this run belongs to (D-127).

    Scoped to the *session*, not the turn, because that is what the sidebar lists: renaming
    one turn of a conversation and leaving the rest would show a title that does not describe
    what opening it produces.
    """
    session_id = await _session_of(request, thread_id)
    conn = request.app.state.conn

    if body.title is not None:
        await rename_session(conn, session_id, body.title)
    if body.starred is not None:
        await set_starred(conn, session_id, body.starred)
    return Response(status_code=204)


@router.delete("/{thread_id}/session", status_code=204)
async def delete_session_route(request: Request, thread_id: str) -> Response:
    """Delete a whole conversation: its rows **and** every turn's checkpoint (D-127).

    **The checkpoints are the point.** A turn's row is a few hundred bytes; its checkpoint
    holds the sources, the review and the full graph state, and that is what actually grows
    the database. Deleting rows alone would empty the sidebar while leaving the bytes on disk
    forever -- a delete that looks like it worked, which is this project's recurring failure
    shape (D-062, D-069, O-5) applied to storage.

    One `adelete_thread` per turn, because each turn is its own thread (D-121). Deleting only
    the one the user clicked would orphan every follow-up's checkpoint.

    The corpus is untouched: its papers are shared with every other run and are the artifact
    worth keeping (D-105).
    """
    session_id = await _session_of(request, thread_id)
    thread_ids = await delete_session(request.app.state.conn, session_id)

    checkpointer = request.app.state.graph.checkpointer
    for turn_id in thread_ids:
        await checkpointer.adelete_thread(turn_id)
    return Response(status_code=204)


@router.get("/{thread_id}/session")
async def read_session(request: Request, thread_id: str) -> list[dict[str, Any]]:
    """Every turn of the conversation this run belongs to, oldest first (D-121)."""
    run = await get_run(request.app.state.conn, thread_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"no run {thread_id}")
    turns = await list_session(request.app.state.conn, run.session_id or run.thread_id)
    return [
        {
            "thread_id": turn.thread_id,
            "question": turn.question,
            "created_at": turn.created_at,
            "is_follow_up": turn.is_follow_up,
        }
        for turn in turns
    ]


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
        # "" means the provider's default was used (D-125).
        "model": run.model,
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
        try:
            async for chunk in stream_run(
                graph, thread_id, run.question, run.provider, run.model
            ):
                event = _event_for(chunk)
                if event is not None:
                    yield _sse(*event)
        except Exception as exc:  # noqa: BLE001 - the outermost boundary of a live stream
            # **Say why the run stopped.** Letting this propagate truncates the SSE response,
            # which reaches the browser as EventSource's generic error -- and the page then
            # offers "reload to resume", advice that for a bad model name (D-125) will fail
            # identically forever. The checkpoint is untouched, so the run stays genuinely
            # resumable for the failures where resuming *does* help (D-081).
            #
            # Logged with the traceback as well as reported, because catching it here is what
            # removes it from the server log.
            logger.exception("run %s failed", thread_id)
            yield _sse("failed", {"message": f"{type(exc).__name__}: {exc}"})
            return

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
