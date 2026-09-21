"""The HTML pages (D-080). Server-rendered Jinja2; no build step, no node_modules."""

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from deep_research.persistence.runs import list_runs

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
