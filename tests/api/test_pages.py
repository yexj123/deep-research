"""Page and static-asset tests (D-080).

Deliberately shallow: these assert the page is served, the assets exist, and the history
fragment escapes what it renders. Anything about how the page *behaves* lives in the browser
and is not testable here -- a test that pretended otherwise would be the ASGITransport mistake
again (see tests/agent/test_runner.py).
"""

import pytest

from deep_research.agent.config import RECOMMENDED_MODELS
from tests.agent.fakes import DEFAULT_REPLY
from tests.api.test_runs import finished_run, follow_up, start_run


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


# ---- the model picker (D-126) ---------------------------------------------------------


@pytest.mark.asyncio
async def test_the_model_options_come_from_config_not_the_template(api) -> None:
    """Adding a model must be a one-line config change (D-126).

    If the template held the list, `RECOMMENDED_MODELS` and the page would drift, and the
    drift would be invisible -- the form would keep working while offering the wrong models.
    """
    client, _ = api
    page = (await client.get("/")).text

    for provider, models in RECOMMENDED_MODELS.items():
        for model in models:
            assert f'value="{model}"' in page, f"{model} missing from the form"
            assert f'data-provider="{provider}"' in page


@pytest.mark.asyncio
async def test_each_option_declares_its_provider(api) -> None:
    """The browser filters the list by provider, and needs the mapping to do it (D-126).

    Carried on the options themselves rather than duplicated into app.js, so the server stays
    the only source of truth for what the page offers.
    """
    client, _ = api
    page = (await client.get("/")).text

    assert 'value="gpt-4o" data-provider="openai"' in page.replace("\n", " ")
    assert 'data-provider="deepseek"' in page


@pytest.mark.asyncio
async def test_the_form_offers_a_custom_model_escape_hatch(api) -> None:
    """A model released tomorrow must work without a code change (D-126).

    This is the reason the list is a suggestion and not a whitelist, so it is worth pinning
    that the escape hatch actually exists in the markup.
    """
    client, _ = api
    page = (await client.get("/")).text

    assert 'value="__custom__"' in page
    assert 'id="custom-model"' in page
    assert 'id="custom-model-row"' in page


@pytest.mark.asyncio
async def test_the_default_model_is_marked_in_the_list(api) -> None:
    """The reader should be able to see which option an omitted model would have used."""
    client, _ = api
    page = (await client.get("/")).text
    assert "(default)" in page


@pytest.mark.asyncio
async def test_every_graph_node_has_a_human_label(api) -> None:
    """The progress trail must never show a raw node name (O-5).

    `check_claims` shipped in D-113 without one, so the reader saw literally "check_claims"
    in the trail -- found by looking at a real run in a browser, not by any test. A node added
    later will do the same unless something compares the two lists.
    """
    client, _ = api
    script = (await client.get("/static/app.js")).text

    # Read from the compiled graph, not a list written here: a hardcoded list would need
    # updating alongside the thing it is supposed to police, which is how check_claims got
    # missed in the first place.
    nodes = [n for n in client.app.state.graph.nodes if not n.startswith("__")]
    assert nodes, "no nodes found; this check would pass vacuously"

    for node in nodes:
        assert f"{node}:" in script, f"{node} has no entry in NODE_LABELS"


@pytest.mark.asyncio
async def test_the_worker_label_does_not_claim_a_search(api) -> None:
    """A worker may answer entirely from the corpus, so the label must not say "arXiv" (D-118).

    The `progress` events name what actually happened ("Answered X from 15 local paper(s)").
    A node label reading "Searching arXiv" directly above them contradicts the truth the run
    just reported -- the same bug D-118 fixed in the worker and left standing in the UI.
    """
    client, _ = api
    script = (await client.get("/static/app.js")).text
    worker_label = script.split("research_worker:")[1].split(",")[0]

    assert "arXiv" not in worker_label, worker_label


# ---- conversations in the UI (D-121) --------------------------------------------------


@pytest.mark.asyncio
async def test_the_history_lists_conversations_not_turns(api) -> None:
    """A follow-up does not get its own sidebar entry (D-121).

    This is the regression the feature would otherwise introduce: `list_runs` would show a
    rewritten follow-up beside the question it followed up on, as though the two were
    unrelated research questions -- the exact confusion sessions exist to remove.
    """
    client, _ = api
    first = await finished_run(client, "How does speculative decoding work?")
    await follow_up(client, first, "What about quantization?")

    body = (await client.get("/history")).text

    assert body.count('class="entry"') == 1, "one entry per conversation, not per turn"
    assert "How does speculative decoding work?" in body
    assert "What about quantization?" not in body
    assert "2 turns" in body, "the entry says how far the conversation got"


@pytest.mark.asyncio
async def test_a_one_turn_conversation_is_not_labelled_with_a_turn_count(api) -> None:
    """"1 turns" is noise on every row of a fresh install (D-121)."""
    client, _ = api
    await start_run(client, "a single question")

    body = (await client.get("/history")).text
    assert "turns" not in body


@pytest.mark.asyncio
async def test_following_up_moves_a_conversation_back_to_the_top(api) -> None:
    """The sidebar orders by the newest turn, not by when the conversation started (D-121).

    Ordering by the head's `created_at` would bury a conversation you are actively working in
    underneath every question asked since it began.
    """
    client, factory = api
    old = await finished_run(client, "the older conversation", factory)
    await finished_run(client, "the newer conversation", factory)
    await follow_up(client, old, "and what about that?")

    body = (await client.get("/history")).text
    assert body.index("the older conversation") < body.index("the newer conversation")


@pytest.mark.asyncio
async def test_opening_a_conversation_shows_every_turn_oldest_first(api) -> None:
    """Opening one entry renders the whole session, in the order it was asked (D-121).

    A conversation that renders only its first turn would hide the answers the follow-ups
    produced -- and those are the only reason to follow up.
    """
    client, factory = api
    first = await finished_run(client, "the first question", factory)
    second = await follow_up(client, first, "the follow-up")
    factory.restart()
    await client.get(f"/runs/{second}/stream")

    view = (await client.get(f"/runs/{first}/view")).text

    assert view.index("the first question") < view.index("the follow-up (resolved)")
    assert view.count("<article") == 2
    assert "Follow-up" in view, "a follow-up turn is marked as one"


@pytest.mark.asyncio
async def test_opening_a_follow_up_shows_the_conversation_it_belongs_to(api) -> None:
    """Any turn's thread_id opens the whole session, not that turn alone (D-121).

    The sidebar only ever links the first turn, but `/runs/{id}/view` is reachable directly,
    and a follow-up rendered without the question it answers reads as a non sequitur.
    """
    client, _ = api
    first = await finished_run(client, "the first question")
    second = await follow_up(client, first, "the follow-up")

    view = (await client.get(f"/runs/{second}/view")).text
    assert "the first question" in view
    assert "the follow-up (resolved)" in view


@pytest.mark.asyncio
async def test_the_composer_is_offered_only_when_there_is_a_review_to_follow_up_on(api) -> None:
    """`data-follow-up` marks a conversation as continuable, and the server decides (D-121).

    POST /runs refuses a follow-up to a run with no review (409). Letting the browser decide
    when to show the composer would mean a button whose only outcome is that 409.
    """
    client, _ = api
    unstarted = await start_run(client, "never streamed")
    assert "data-follow-up" not in (await client.get(f"/runs/{unstarted}/view")).text

    done = await finished_run(client, "actually answered")
    view = (await client.get(f"/runs/{done}/view")).text
    assert f'data-follow-up="{done}"' in view


@pytest.mark.asyncio
async def test_the_composer_points_at_the_last_turn_not_the_first(api) -> None:
    """A second follow-up continues from the latest answer (D-121).

    Pointing at the head would rewrite every follow-up against the first review, so turn 3
    would be blind to what turn 2 found.
    """
    client, factory = api
    first = await finished_run(client, "the first question", factory)
    second = await follow_up(client, first, "the follow-up")
    factory.restart()
    await client.get(f"/runs/{second}/stream")

    view = (await client.get(f"/runs/{first}/view")).text
    assert f'data-follow-up="{second}"' in view


@pytest.mark.asyncio
async def test_the_page_has_a_follow_up_composer(api) -> None:
    """The feature is reachable from the UI, and hidden until it applies (D-121)."""
    client, _ = api
    page = (await client.get("/")).text

    assert 'id="follow-up"' in page
    assert 'id="follow-up-question"' in page
    assert 'id="follow-up"' in page and "hidden" in page


@pytest.mark.asyncio
async def test_the_hidden_attribute_is_not_overridden_by_the_form_rule(api) -> None:
    """`form { display: flex }` beats the UA stylesheet's `[hidden] { display: none }`.

    Every pane on this page is shown and hidden through the `hidden` attribute, and the
    follow-up composer is the first `<form>` to use it -- so without an explicit rule it
    would be visible before there is anything to follow up on. Caught before shipping;
    asserted here because nothing else on the page would fail if the rule were deleted.
    """
    client, _ = api
    css = (await client.get("/static/app.css")).text
    assert "[hidden] { display: none !important; }" in css
