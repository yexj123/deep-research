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


async def finished_run(
    client: httpx.AsyncClient,
    question: str = "What is attention?",
    factory: RecordingFactory | None = None,
) -> str:
    """A run that has actually produced a review -- the precondition for following up.

    Pass `factory` when the test runs the graph more than once: the scripted replies are one
    run long, and the second run would otherwise be handed the tail of the first one's script.
    """
    thread_id = await start_run(client, question)
    if factory is not None:
        factory.restart()
    await client.get(f"/runs/{thread_id}/stream")
    return thread_id


async def follow_up(
    client: httpx.AsyncClient, parent: str, question: str = "What about it?"
) -> str:
    """Ask `question` as a follow-up to `parent` (D-121).

    Stubs the rewriter: the shared fake model replies from a fixed script, so left to itself
    it would hand the rewriter whatever reply came next and the test would pass or fail for a
    reason that has nothing to do with sessions.
    """

    async def fake_rewrite(
        q: str, parent_question: str, review: str, provider: str, model: str = ""
    ) -> str:
        return f"{q} (resolved)"

    client.app.state.rewrite_follow_up = fake_rewrite
    response = await client.post(
        "/runs", json={"question": question, "follow_up_to": parent}
    )
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
async def test_a_clean_run_still_says_why_it_stopped(api) -> None:
    """A clean run reports its stop reason but no list of losses (O-15, replacing O-5's rule).

    O-5 hid this panel entirely when nothing was lost, which was right while it only listed
    losses. D-096 changed what it has to say: runs now stop after a single round, and a reader
    watching three searches and then synthesis has no way to learn why it did not go deeper.
    The sentence existed, was tested, and was never shown -- on exactly the runs that prompt
    the question.

    So the split is now: stop reason unconditional, loss list conditional. The heading follows,
    so a clean run does not announce limitations it does not have.
    """
    import json

    client, _ = api
    thread_id = await start_run(client)

    body = (await client.get(f"/runs/{thread_id}/stream")).text
    done = [data for event, data in parse_sse(body) if event == "done"][0]
    coverage_html = json.loads(done)["coverage_html"]

    assert "The run stopped because" in coverage_html
    assert "<h2>Coverage</h2>" in coverage_html, "a clean run must not be headed 'limitations'"
    # The loss list stays conditional -- this is the part O-5 got right.
    for loss in ("No papers found for", "Searches that failed", "were skipped"):
        assert loss not in coverage_html, f"clean run should not mention {loss!r}"


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


# ---- follow-up questions (D-121) ------------------------------------------------------


@pytest.mark.asyncio
async def test_a_follow_up_joins_its_parents_session(api) -> None:
    """A follow-up is its own thread with its own clean state, tied only by session_id.

    Carrying graph state forward was the original plan and is not what shipped: the corpus
    already supplies research continuity across every thread, filtered per subtopic, and
    carrying `explored_subtopics` would stop the planner revisiting the very topic the
    follow-up asks about.
    """
    import json

    client, _ = api
    first = await start_run(client)
    await client.get(f"/runs/{first}/stream")  # produce a review to follow up on

    created = await client.post(
        "/runs", json={"question": "What about it?", "provider": "openai", "follow_up_to": first}
    )
    assert created.status_code == 201
    second = created.json()["thread_id"]
    assert second != first, "a follow-up gets its own thread, never the parent's"

    session = (await client.get(f"/runs/{second}/session")).json()
    assert [t["thread_id"] for t in session] == [first, second], "oldest first"
    assert [t["is_follow_up"] for t in session] == [False, True]


@pytest.mark.asyncio
async def test_a_follow_up_is_stored_as_the_rewritten_question(api) -> None:
    """What gets stored is the rewriter's output, not "What about it?" (D-121).

    Rewriting at creation keeps one code path through the graph, and makes the history entry
    read as a real question rather than a fragment nobody can interpret later.

    The rewriter is stubbed rather than driven through the shared fake factory. With the
    factory, this test passed by storing leftover claim-checker JSON as the question -- green
    for a reason that had nothing to do with the behaviour being claimed. The rewriter's own
    contract is covered in `tests/agent/test_followup.py`; what matters here is that the route
    calls it and stores what it returns.
    """
    client, _ = api
    first = await start_run(client)
    await client.get(f"/runs/{first}/stream")

    seen: list[tuple[str, str]] = []

    async def fake_rewrite(question, parent_question, parent_review, provider, model=""):
        seen.append((question, parent_question))
        return "How does quantization compare to speculative decoding?"

    client.app.state.rewrite_follow_up = fake_rewrite

    second = (
        await client.post(
            "/runs",
            json={"question": "What about it?", "provider": "openai", "follow_up_to": first},
        )
    ).json()["thread_id"]

    assert seen and seen[0][0] == "What about it?", "the rewriter must see the raw follow-up"
    stored = (await client.get(f"/runs/{second}")).json()["question"]
    assert stored == "How does quantization compare to speculative decoding?"


@pytest.mark.asyncio
async def test_creating_a_run_returns_the_question_it_stored(api) -> None:
    """POST /runs hands back the question, which for a follow-up is the rewritten one (D-121).

    The page shows this. A rewrite the reader can't see is a rewrite they can't trust -- and
    returning it here is what lets the page show it without a second request.
    """
    client, factory = api
    first = await finished_run(client, factory=factory)

    plain = await client.post("/runs", json={"question": "  What is attention?  "})
    assert plain.json()["question"] == "What is attention?", "stored stripped, returned stripped"

    async def fake_rewrite(
        q: str, parent_question: str, review: str, provider: str, model: str = ""
    ) -> str:
        return "How does quantization speed up inference?"

    client.app.state.rewrite_follow_up = fake_rewrite
    created = (
        await client.post(
            "/runs", json={"question": "What about quantization?", "follow_up_to": first}
        )
    ).json()

    assert created["question"] == "How does quantization speed up inference?"
    stored = (await client.get(f"/runs/{created['thread_id']}")).json()
    assert stored["question"] == created["question"], "returned == stored"


# ---- the typed model (D-125) ----------------------------------------------------------


@pytest.mark.asyncio
async def test_a_typed_model_reaches_the_factory_for_every_node(api) -> None:
    """The model is a per-run choice, so it must travel in runtime context (D-015, D-125).

    Asserted on *every* model the run built, not just the first: a global would work for the
    planner and then leak into a concurrent run, and checking only one call would not see it.
    """
    client, factory = api
    thread_id = (
        await client.post("/runs", json={"question": "What is attention?", "model": "gpt-4o-mini"})
    ).json()["thread_id"]
    await client.get(f"/runs/{thread_id}/stream")

    assert factory.models, "no models were built"
    assert set(factory.models) == {"gpt-4o-mini"}


@pytest.mark.asyncio
async def test_omitting_the_model_reaches_the_factory_as_none(api) -> None:
    """Omitted means "the provider's default", and arrives as None (D-126).

    The control for the test above: without it, a factory receiving *something* either way
    would look correct while silently ignoring the override. `None` rather than `""` because
    "not chosen" and "chosen as blank" are different facts and only one is reachable.
    """
    client, factory = api
    thread_id = await start_run(client)
    await client.get(f"/runs/{thread_id}/stream")

    assert set(factory.models) == {None}


@pytest.mark.asyncio
@pytest.mark.parametrize("sent", ["", "   ", None])
async def test_a_blank_model_collapses_to_the_default(api, sent) -> None:
    """"", "   " and an omitted field are the same fact (D-126).

    Collapsed at the API boundary so nothing downstream has to tell three absent values
    apart -- the kind of near-duplicate state that ends up handled in two places and one of
    them wrong.
    """
    client, factory = api
    created = await client.post("/runs", json={"question": "What is attention?", "model": sent})
    assert created.status_code == 201, created.text

    body = (await client.get(f"/runs/{created.json()['thread_id']}")).json()
    assert body["model"] is None


@pytest.mark.asyncio
async def test_the_model_is_stored_and_survives_a_resume(api) -> None:
    """A resumed run must finish on the model it started on (D-125).

    `stream_run` resumes from a checkpoint days later. If the model were re-read from config
    rather than from the run, half the review would be written by one model and half by
    another -- and nothing would report it.
    """
    client, _ = api
    thread_id = (
        await client.post("/runs", json={"question": "What is attention?", "model": "gpt-4o-mini"})
    ).json()["thread_id"]

    body = (await client.get(f"/runs/{thread_id}")).json()
    assert body["model"] == "gpt-4o-mini"

    session = (await client.get(f"/runs/{thread_id}/session")).json()
    assert session[0]["thread_id"] == thread_id


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt 4o", "x" * 101, "-leading", "a\nb"])
async def test_a_malformed_model_is_refused_before_a_thread_exists(api, model: str) -> None:
    """422 at the boundary, not a 500 from a run that started and then raised (D-013).

    `intake` would also reject it, but only after POST /runs had handed back a thread_id the
    caller would then stream into an exception.
    """
    client, _ = api
    response = await client.post("/runs", json={"question": "What is attention?", "model": model})
    assert response.status_code == 422, response.text


@pytest.mark.asyncio
async def test_a_model_with_surrounding_whitespace_is_accepted_and_trimmed(api) -> None:
    """Typing into a text field produces stray spaces; that is a UI artifact, not an error."""
    client, _ = api
    created = await client.post(
        "/runs", json={"question": "What is attention?", "model": "  gpt-4o-mini  "}
    )
    assert created.status_code == 201, created.text
    body = (await client.get(f"/runs/{created.json()['thread_id']}")).json()
    assert body["model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_a_run_that_raises_reports_why_instead_of_dying_silently(api, monkeypatch) -> None:
    """A failing run explains itself over SSE (D-125).

    Letting the exception propagate truncates the stream, which reaches the browser as
    EventSource's generic error -- and the page then offers "reload to resume", advice that
    for a bad model name will fail identically forever. Found by typing a nonexistent model
    into a live server: the provider's 404 named the model, and the browser saw none of it.
    """
    client, _ = api
    thread_id = await start_run(client)

    async def boom(*args, **kwargs):
        raise RuntimeError("The model `gpt-4o-imaginary` does not exist")
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr("deep_research.api.routes.runs.stream_run", boom)

    events = parse_sse((await client.get(f"/runs/{thread_id}/stream")).text)
    failures = [data for name, data in events if name == "failed"]

    assert failures, f"no failure event; got {[n for n, _ in events]}"
    assert "gpt-4o-imaginary" in failures[0], "the provider's own message must reach the reader"


@pytest.mark.asyncio
async def test_a_follow_up_to_an_unfinished_run_is_refused(api) -> None:
    """Rewriting against a review that does not exist invents the missing context.

    409 rather than 404: the run is real, it just has nothing to follow up on yet.
    """
    client, _ = api
    first = await start_run(client)  # created, never streamed

    response = await client.post(
        "/runs", json={"question": "What about it?", "provider": "openai", "follow_up_to": first}
    )
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_a_follow_up_to_a_missing_run_is_a_404(api) -> None:
    """The control for the test above: an unknown parent is a different failure."""
    client, _ = api
    response = await client.post(
        "/runs",
        json={"question": "What about it?", "provider": "openai", "follow_up_to": "nope"},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_a_first_question_is_its_own_session(api) -> None:
    """session_id is never empty, so the UI needs no special case for a first question."""
    client, _ = api
    first = await start_run(client)

    session = (await client.get(f"/runs/{first}/session")).json()
    assert [t["thread_id"] for t in session] == [first]
    assert session[0]["is_follow_up"] is False
