"""Page and static-asset tests (D-080).

Deliberately shallow: these assert the page is served, the assets exist, and the history
fragment escapes what it renders. Anything about how the page *behaves* lives in the browser
and is not testable here -- a test that pretended otherwise would be the ASGITransport mistake
again (see tests/agent/test_runner.py).
"""

import pytest

from tests.agent.fakes import DEFAULT_REPLY
from tests.api.test_runs import start_run


@pytest.mark.asyncio
async def test_the_index_page_is_served(api) -> None:
    """GET / returns the page with the form the run flow needs."""
    client, _ = api
    response = await client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert 'id="ask"' in response.text
    assert 'id="review"' in response.text
    assert 'id="loaded-run"' in response.text


@pytest.mark.asyncio
async def test_the_page_loads_its_own_assets_not_a_framework_bundle(api) -> None:
    """The stylesheet and the run-flow script are served from this app (D-080).

    htmx comes from a CDN with an integrity hash; everything else is local, because there is
    no build step and nothing to bundle.
    """
    client, _ = api
    page = (await client.get("/")).text

    assert "/static/app.js" in page
    assert "/static/app.css" in page
    assert (await client.get("/static/app.js")).status_code == 200
    assert (await client.get("/static/app.css")).status_code == 200


@pytest.mark.asyncio
async def test_the_history_fragment_lists_runs_newest_first(api) -> None:
    """htmx swaps this in; it is HTML, not JSON, which is the one place htmx earns its keep."""
    client, _ = api
    await client.post("/runs", json={"question": "first question"})
    await client.post("/runs", json={"question": "second question"})

    body = (await client.get("/history")).text
    assert body.index("second question") < body.index("first question")


@pytest.mark.asyncio
async def test_the_history_fragment_escapes_the_question(api) -> None:
    """A question containing markup renders as text (O-8).

    The review is not the only model- or user-supplied string reaching the DOM. Jinja2
    autoescaping handles this one; this test is what would catch autoescaping being turned
    off or the fragment being built by string concatenation instead.
    """
    client, _ = api
    await client.post("/runs", json={"question": "<script>alert('xss')</script>"})

    body = (await client.get("/history")).text
    assert "<script>alert" not in body
    assert "&lt;script&gt;" in body


@pytest.mark.asyncio
async def test_an_empty_history_says_so(api) -> None:
    """A fresh install shows a message rather than an empty list with no explanation."""
    client, _ = api
    assert "No runs yet" in (await client.get("/history")).text


# ---- the sidebar ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_page_has_a_sidebar_that_loads_history(api) -> None:
    """The history lives in a sidebar, loaded by htmx on page load."""
    client, _ = api
    page = (await client.get("/")).text

    assert 'class="sidebar"' in page
    assert 'hx-get="/history"' in page
    assert 'id="new-run"' in page


@pytest.mark.asyncio
async def test_a_history_entry_loads_that_run_into_the_main_pane(api) -> None:
    """Each entry is an htmx trigger targeting #loaded-run, not a link that reloads the page."""
    client, _ = api
    thread_id = await start_run(client, "a past question")

    fragment = (await client.get("/history")).text
    assert f'hx-get="/runs/{thread_id}/view"' in fragment
    assert 'hx-target="#loaded-run"' in fragment


@pytest.mark.asyncio
async def test_opening_a_finished_run_shows_its_review(api) -> None:
    """Reopening a past run renders its saved review -- without re-running it (D-081)."""
    client, factory = api
    thread_id = await start_run(client)
    await client.get(f"/runs/{thread_id}/stream")
    models_after_run = factory.models_built

    view = (await client.get(f"/runs/{thread_id}/view")).text

    assert DEFAULT_REPLY in view
    assert "finished" in view
    assert "Resume it" not in view
    assert "Start it" not in view
    assert factory.models_built == models_after_run, "opening a run must not execute it"


@pytest.mark.asyncio
async def test_opening_an_unstarted_run_offers_to_start_it(api) -> None:
    """A run that was never executed says so, and offers to start it (D-081).

    Wording matters here: POST /runs records a run without running it, so "this run didn't
    finish" would report a failure that never happened.
    """
    client, _ = api
    thread_id = await start_run(client, "never streamed")

    view = (await client.get(f"/runs/{thread_id}/view")).text

    assert "hasn't been started yet" in view
    assert "Start it" in view
    assert "stopped before finishing" not in view
    assert f'data-resume="{thread_id}"' in view
    assert "No review yet" in view


@pytest.mark.asyncio
async def test_a_finished_run_offers_neither_start_nor_resume(api) -> None:
    """Nothing to start or resume once the run is done (D-084)."""
    client, _ = api
    thread_id = await start_run(client)
    await client.get(f"/runs/{thread_id}/stream")

    view = (await client.get(f"/runs/{thread_id}/view")).text
    assert "data-resume" not in view


@pytest.mark.asyncio
async def test_the_run_view_escapes_the_question(api) -> None:
    """A question containing markup renders as text here too (D-085).

    The review is rendered by api/rendering.py; everything else in the fragment relies on
    Jinja2 autoescaping, and the template uses |safe on exactly two already-escaped values.
    """
    client, _ = api
    response = await client.post("/runs", json={"question": "<script>alert('xss')</script>"})
    thread_id = response.json()["thread_id"]

    view = (await client.get(f"/runs/{thread_id}/view")).text
    assert "<script>alert" not in view
    assert "&lt;script&gt;" in view


@pytest.mark.asyncio
async def test_opening_an_unknown_run_is_404(api) -> None:
    """A thread_id nobody created has no view, rather than an empty page."""
    client, _ = api
    assert (await client.get("/runs/never-created/view")).status_code == 404
