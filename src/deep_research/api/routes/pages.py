"""The HTML pages (D-080). Server-rendered Jinja2; no build step, no node_modules."""

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langgraph.graph.state import CompiledStateGraph

from fastapi import HTTPException

from deep_research.agent.coverage import summarize_coverage
from deep_research.agent.runner import get_review, get_run_state
from deep_research.api.rendering import render_coverage, render_review
from deep_research.persistence.runs import Run, get_run, list_session, list_session_heads

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter(tags=["pages"])


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """The single page: ask a question, watch it run, read the review."""
    return templates.TemplateResponse(request, "index.html")


@router.get("/history", response_class=HTMLResponse)
async def history(request: Request) -> HTMLResponse:
    """The conversation list as an HTML fragment, for htmx to swap in.

    A fragment rather than JSON because this is the one part of the page htmx genuinely
    handles: `hx-get` plus a `refresh-history` event after each run, no JavaScript involved.
    Jinja2 autoescaping is what keeps a question containing markup inert here.

    One entry per *conversation*, not per run (D-121): a follow-up belongs under the question
    it followed up on, not beside it.
    """
    sessions = await list_session_heads(request.app.state.conn)
    return templates.TemplateResponse(request, "_history.html", {"sessions": sessions})


@router.get("/runs/{thread_id}/view", response_class=HTMLResponse)
async def run_view(request: Request, thread_id: str) -> HTMLResponse:
    """A past conversation as an HTML fragment, for the sidebar to swap into the main pane.

    Server-rendered rather than JSON + client rendering, so the markdown stays escaped in
    Python (D-085) and the page needs no second rendering path (D-080). This is htmx's other
    real job here, alongside the history list.

    Renders every turn of the session, oldest first (D-121) -- opening a conversation should
    show the conversation. One checkpoint read per turn, which is why the history list still
    refuses to do the same thing per row (N+1 over *every* run, not over one session).
    """
    run = await get_run(request.app.state.conn, thread_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"no run {thread_id}")

    graph = request.app.state.graph
    turns = await list_session(request.app.state.conn, run.session_id or run.thread_id)
    rendered = [await _render_turn(graph, turn) for turn in turns]
    return templates.TemplateResponse(request, "_run_view.html", {"turns": rendered})


async def _render_turn(graph: CompiledStateGraph, run: Run) -> dict[str, Any]:
    """One turn of a conversation, with its review already rendered and escaped (D-085)."""
    values = await get_review(graph, run.thread_id)
    review = values.get("review", "")
    return {
        "run": run,
        "status": (await get_run_state(graph, run.thread_id)).value,
        "review_html": render_review(review) if review else "",
        # Always rendered since O-15 -- see the note in routes/runs.py.
        "coverage_html": render_coverage(summarize_coverage(values)),
        "citation_violations": values.get("citation_violations", []),
    }
