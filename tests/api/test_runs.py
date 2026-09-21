"""Run route tests (D-081, D-083). No server, no network: httpx.ASGITransport + fakes.

The agent underneath is real -- a real graph over a real SQLite checkpoint file -- so these
cover the seam the agent tests can't: that the web layer starts, resumes and replays a run
correctly, and that what reaches the browser is JSON-serializable.
"""

import httpx
import pytest

from tests.agent.fakes import DEFAULT_REPLY, RecordingFactory


def parse_sse(body: str) -> list[tuple[str, str]]:
    """Split an SSE body into (event, data) pairs.

    Hand-rolled to match the hand-rolled framing (D-083): if the format is ours to defend,
    the test shouldn't depend on a library to read it.
    """
    events: list[tuple[str, str]] = []
    for block in body.split("\n\n"):
        if not block.strip():
            continue
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        events.append((lines.get("event", "message"), lines.get("data", "")))
    return events


async def start_run(client: httpx.AsyncClient, question: str = "What is attention?") -> str:
    response = await client.post("/runs", json={"question": question})
    assert response.status_code == 201, response.text
    return response.json()["thread_id"]


# ---- creating a run ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_creating_a_run_returns_a_thread_id_and_executes_nothing(api) -> None:
    """POST /runs records the run and returns immediately (D-081).

    No model is built and no arXiv request is made: the stream route drives the run, so
    creating one is cheap and a client can create several before streaming any.
    """
    client, factory = api
    thread_id = await start_run(client)

    assert thread_id
    assert factory.providers == [], "creating a run must not call the model"


@pytest.mark.asyncio
async def test_a_blank_question_is_rejected(api) -> None:
    """A whitespace-only question fails at the boundary, before a thread_id exists (D-033)."""
    client, _ = api
    response = await client.post("/runs", json={"question": "   "})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_an_invalid_provider_is_rejected_by_the_schema(api) -> None:
    """ProviderType is a Literal, so Pydantic rejects an unknown provider (D-013, D-015).

    intake also checks this (D-033), but the request body is the boundary -- catching it here
    means no thread_id is created for a run that could never work.
    """
    client, _ = api
    response = await client.post("/runs", json={"question": "q", "provider": "gemini"})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_listing_runs_returns_them_newest_first(api) -> None:
    """The history list. Ordering is what the UI shows, so it's part of the contract."""
    client, _ = api
    await start_run(client, "first question")
    await start_run(client, "second question")

    runs = (await client.get("/runs")).json()
    assert [r["question"] for r in runs] == ["second question", "first question"]


@pytest.mark.asyncio
async def test_reading_an_unknown_thread_id_is_404(api) -> None:
    """A thread_id nobody created is not found, rather than an empty review (D-081)."""
    client, _ = api
    assert (await client.get("/runs/never-created")).status_code == 404
    assert (await client.get("/runs/never-created/stream")).status_code == 404


# ---- streaming -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_a_new_run_produces_progress_then_tokens_then_done(api) -> None:
    """The stream carries a node trail, the review's tokens, and a terminating done event."""
    client, _ = api
    thread_id = await start_run(client)

    body = (await client.get(f"/runs/{thread_id}/stream")).text
    events = parse_sse(body)
    kinds = [event for event, _ in events]

    assert "node" in kinds, "expected a progress trail from `updates`"
    assert "token" in kinds, "expected the review to stream from `messages`"
    assert kinds[-1] == "done", "the stream must terminate with done"


@pytest.mark.asyncio
async def test_only_the_review_streams_as_tokens(api) -> None:
    """decompose's tokens must not reach the reader (D-080).

    From milestone 3 the planner calls the model too, so an unfiltered `messages` stream
    carries its JSON plan. Unfiltered, that lands in the user's review pane as raw JSON.
    """
    client, _ = api
    thread_id = await start_run(client)

    body = (await client.get(f"/runs/{thread_id}/stream")).text
    streamed = "".join(
        __import__("json").loads(data)["text"] for event, data in parse_sse(body) if event == "token"
    )
    assert streamed == DEFAULT_REPLY
    assert "subtopics" not in streamed, "the planner's JSON must not reach the review pane"


@pytest.mark.asyncio
async def test_the_stream_is_marked_as_an_event_stream(api) -> None:
    """The content type is what makes a browser's EventSource work at all (D-083)."""
    client, _ = api
    thread_id = await start_run(client)

    response = await client.get(f"/runs/{thread_id}/stream")
    assert response.headers["content-type"].startswith("text/event-stream")
    # A buffering proxy would deliver the whole stream at the end, which looks like a hang.
    assert response.headers["x-accel-buffering"] == "no"


@pytest.mark.asyncio
async def test_every_streamed_event_is_json(api) -> None:
    """Raw LangGraph updates carry Source objects, which aren't JSON-serializable.

    The route translates chunks into browser events rather than passing them through, and
    this is what would catch a passthrough being reintroduced.
    """
    client, _ = api
    thread_id = await start_run(client)

    import json

    for event, data in parse_sse((await client.get(f"/runs/{thread_id}/stream")).text):
        json.loads(data)  # raises if the route ever emits something unserializable


# ---- replay: the third state (D-081) -------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_a_finished_run_replays_without_re_running(api) -> None:
    """Reconnecting to a completed run must not re-execute the graph (D-081).

    A finished run has `next == ()` just like one that never started, so without the
    created_at check this would re-run everything and re-bill every model call.
    """
    client, factory = api
    thread_id = await start_run(client)

    await client.get(f"/runs/{thread_id}/stream")
    models_after_first = factory.models_built

    body = (await client.get(f"/runs/{thread_id}/stream")).text
    events = parse_sse(body)

    assert factory.models_built == models_after_first, "a finished run must not call the model again"
    assert [event for event, _ in events] == ["done"], "replay sends only the saved review"
    assert DEFAULT_REPLY in events[0][1]


@pytest.mark.asyncio
async def test_the_review_is_readable_after_the_stream_ends(api) -> None:
    """GET /runs/{id} returns the saved review and the grounding result (D-046)."""
    client, _ = api
    thread_id = await start_run(client)
    await client.get(f"/runs/{thread_id}/stream")

    run = (await client.get(f"/runs/{thread_id}")).json()
    assert run["status"] == "finished"
    assert run["review"] == DEFAULT_REPLY
    assert run["citation_violations"] == []
    # What the run covered is reported as a structured summary rather than loose fields, so
    # the same shape serves the UI, the API and the thesis notebooks (O-5).
    coverage = run["coverage"]
    assert sorted(coverage["explored"]) == ["attention mechanisms", "positional encoding"]
    assert coverage["papers"] == 3
    assert coverage["empty"] == []
    assert coverage["failed"] == {}


@pytest.mark.asyncio
async def test_a_run_that_has_not_streamed_yet_reports_not_started(api) -> None:
    """Status distinguishes "created" from "finished", which the UI needs (D-081)."""
    client, _ = api
    thread_id = await start_run(client)

    run = (await client.get(f"/runs/{thread_id}")).json()
    assert run["status"] == "not_started"
    assert run["review"] == ""


@pytest.mark.asyncio
async def test_the_provider_from_the_request_reaches_the_model_factory(api) -> None:
    """The per-run provider travels from the request body to the factory (D-015).

    It must not come from a module-level global: two concurrent runs would overwrite it.
    """
    client, factory = api
    response = await client.post("/runs", json={"question": "q", "provider": "deepseek"})
    thread_id = response.json()["thread_id"]
    await client.get(f"/runs/{thread_id}/stream")

    assert set(factory.providers) == {"deepseek"}


# ---- coverage: the run must not hide what it lost (O-5) ------------------------------


@pytest.mark.asyncio
async def test_a_clean_run_sends_no_coverage_section(api) -> None:
    """A run that lost nothing isn't padded with a list of nothing (O-5)."""
    client, _ = api
    thread_id = await start_run(client)

    body = (await client.get(f"/runs/{thread_id}/stream")).text
    done = [data for event, data in parse_sse(body) if event == "done"][0]

    import json

    assert json.loads(done)["coverage_html"] == ""


@pytest.mark.asyncio
async def test_a_run_with_a_failed_subtopic_reports_it(api_with) -> None:
    """A failed search must appear in the report, not just in state (O-5).

    This is the last form of the project's recurring failure shape: without it, a review
    missing a third of its subtopics reads exactly like a complete one.
    """
    import json

    client, _ = await api_with(alpha_status=503)
    thread_id = await start_run(client)

    body = (await client.get(f"/runs/{thread_id}/stream")).text
    done = json.loads([data for event, data in parse_sse(body) if event == "done"][0])

    assert "Searches that failed" in done["coverage_html"]
    assert "alpha topic" in done["coverage_html"]
    assert "coverage is incomplete" in done["coverage_html"]


@pytest.mark.asyncio
async def test_a_run_where_nothing_was_published_says_so(api_with) -> None:
    """Zero results is a finding, and the reader has to be told (D-021, O-5)."""
    import json

    client, _ = await api_with(empty=True)
    thread_id = await start_run(client)

    body = (await client.get(f"/runs/{thread_id}/stream")).text
    done = json.loads([data for event, data in parse_sse(body) if event == "done"][0])

    assert "No papers found for" in done["coverage_html"]
    assert "gap in the literature" in done["coverage_html"]


@pytest.mark.asyncio
async def test_the_stream_carries_live_progress_from_the_workers(api) -> None:
    """`custom` events give mid-search progress that node completion can't (O-5).

    A worker searching arXiv under a 3-second rate limit is the longest silent stretch of a
    run; without these the UI shows nothing between "Planning" and "Writing".
    """
    import json

    client, _ = api
    thread_id = await start_run(client)

    body = (await client.get(f"/runs/{thread_id}/stream")).text
    messages = [json.loads(data)["message"] for event, data in parse_sse(body) if event == "progress"]

    assert any("Searching arXiv" in m for m in messages)
    assert any("Found 3 paper(s)" in m for m in messages)
