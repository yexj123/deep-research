"""The HTML pages (D-080). Server-rendered Jinja2; no build step, no node_modules."""

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from fastapi import HTTPException

from deep_research.agent.coverage import summarize_coverage
from deep_research.agent.runner import get_review, get_run_state
from deep_research.api.rendering import render_coverage, render_review
from deep_research.persistence.runs import get_run, list_runs

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter(tags=["pages"])


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """The single page: ask a question, watch it run, read the review."""
    return templates.TemplateResponse(request, "index.html")


@router.get("/history", response_class=HTMLResponse)
async def history(request: Request) -> HTMLResponse:
    """The run list as an HTML fragment, for htmx to swap in.

    A fragment rather than JSON because this is the one part of the page htmx genuinely
    handles: `hx-get` plus a `refresh-history` event after each run, no JavaScript involved.
    Jinja2 autoescaping is what keeps a question containing markup inert here.
    """
    runs = await list_runs(request.app.state.conn)
    return templates.TemplateResponse(request, "_history.html", {"runs": runs})


@router.get("/runs/{thread_id}/view", response_class=HTMLResponse)
async def run_view(request: Request, thread_id: str) -> HTMLResponse:
    """A past run as an HTML fragment, for the sidebar to swap into the main pane (D-080).

    Server-rendered rather than JSON + client rendering, so the markdown stays escaped in
    Python (D-085) and the page needs no second rendering path. This is htmx's other real
    job here, alongside the history list.
    """
    run = await get_run(request.app.state.conn, thread_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"no run {thread_id}")

    graph = request.app.state.graph
    values = await get_review(graph, thread_id)
    coverage = summarize_coverage(values)
    review = values.get("review", "")

    return templates.TemplateResponse(
        request,
        "_run_view.html",
        {
            "run": run,
            "status": (await get_run_state(graph, thread_id)).value,
            "review_html": render_review(review) if review else "",
            "coverage_html": "" if coverage.is_complete else render_coverage(coverage),
            "citation_violations": values.get("citation_violations", []),
        },
    )
