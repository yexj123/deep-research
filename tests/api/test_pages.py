"""Page and static-asset tests (D-080).

Deliberately shallow: these assert the page is served, the assets exist, and the history
fragment escapes what it renders. Anything about how the page *behaves* lives in the browser
and is not testable here -- a test that pretended otherwise would be the ASGITransport mistake
again (see tests/agent/test_runner.py).
"""

import pytest


@pytest.mark.asyncio
async def test_the_index_page_is_served(api) -> None:
    """GET / returns the page with the form the run flow needs."""
    client, _ = api
    response = await client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert 'id="ask"' in response.text
    assert 'id="review"' in response.text


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
