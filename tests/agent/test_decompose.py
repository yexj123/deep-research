"""decompose node tests (D-070) and the subtopic routing guard (D-069).

The node is called directly with a fake model, so these pin the filter rules without running
the graph. The filters are the point: the planner is an LLM and only *proposes*, so the
prompt's explored list is a soft filter (D-022) and these checks are the enforcement.
"""

import pytest
from langgraph.types import Send
from pydantic import ValidationError

from deep_research.agent.context import RunContext
from deep_research.agent.graph import route_subtopics
from deep_research.agent.nodes.decompose import make_decompose
from deep_research.agent.state import ResearchState
from tests.agent.fakes import RecordingFactory, plan_reply


async def run_decompose(state: ResearchState, reply: str) -> dict:
    """Call the decompose node directly with a scripted planner reply."""
    node = make_decompose(RecordingFactory(reply=reply))
    return await node(state, _runtime())


class _Runtime:
    """Minimal stand-in for langgraph.runtime.Runtime: nodes only read `.context`."""

    def __init__(self, context: RunContext) -> None:
        self.context = context


def _runtime(provider: str = "openai") -> _Runtime:
    return _Runtime(RunContext(provider=provider))  # type: ignore[arg-type]


# ---- filtering -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_proposed_subtopics_become_pending() -> None:
    """With nothing explored or failed, every proposed subtopic is dispatched (D-070)."""
    state = ResearchState(question="What is attention?")
    update = await run_decompose(state, plan_reply("Self-attention", "Positional encoding"))
    assert update["pending_subtopics"] == ["Self-attention", "Positional encoding"]


@pytest.mark.asyncio
async def test_an_already_explored_subtopic_is_filtered_out() -> None:
    """A subtopic matching an explored one after casefold+strip is dropped (D-017, D-022).

    The planner gets the explored list in its prompt, but it is an LLM and rephrases or
    repeats. This check is the enforcement, not the prompt.
    """
    state = ResearchState(question="q", explored_subtopics=["Self-attention"])
    update = await run_decompose(state, plan_reply("  SELF-ATTENTION ", "Positional encoding"))
    assert update["pending_subtopics"] == ["Positional encoding"]


@pytest.mark.asyncio
async def test_a_subtopic_that_failed_twice_is_skipped() -> None:
    """Two entries in failed_subtopics means the N=2 retry cap is spent (D-020).

    Each duplicate is one failed worker run, which is why failed_subtopics uses operator.add
    and keeps repeats (D-067).
    """
    state = ResearchState(question="q", failed_subtopics=["Flaky topic", "Flaky topic"])
    update = await run_decompose(state, plan_reply("Flaky topic", "Fresh topic"))
    assert update["pending_subtopics"] == ["Fresh topic"]


@pytest.mark.asyncio
async def test_a_subtopic_that_failed_once_is_retried() -> None:
    """One failure is below the cap, so the subtopic is proposed again (D-018, D-020).

    A temporary failure must not permanently drop a subtopic.
    """
    state = ResearchState(question="q", failed_subtopics=["Flaky topic"])
    update = await run_decompose(state, plan_reply("Flaky topic"))
    assert update["pending_subtopics"] == ["Flaky topic"]


@pytest.mark.asyncio
async def test_pending_subtopics_is_replaced_not_appended() -> None:
    """pending_subtopics has no reducer, so a new round overwrites the old list (D-017, D-067).

    With operator.add, round N would re-dispatch every subtopic from earlier rounds.
    """
    state = ResearchState(question="q", pending_subtopics=["Stale from last round"])
    update = await run_decompose(state, plan_reply("Fresh topic"))
    assert update["pending_subtopics"] == ["Fresh topic"]


@pytest.mark.asyncio
async def test_the_retry_cap_counts_normalized_subtopics() -> None:
    """Casing and whitespace don't reset the N=2 retry cap (D-020, D-073).

    failed_subtopics is written by workers using the planner's exact wording, and the planner
    rephrases. Counting raw strings would let "  FLAKY TOPIC " and "Flaky topic" each get
    their own budget, so a subtopic that always fails would be retried indefinitely -- the
    exact thing D-020's cap exists to prevent.
    """
    state = ResearchState(question="q", failed_subtopics=["Flaky topic", "  FLAKY TOPIC "])
    update = await run_decompose(state, plan_reply("flaky topic", "Fresh topic"))
    assert update["pending_subtopics"] == ["Fresh topic"]


@pytest.mark.asyncio
async def test_subtopic_normalization_uses_casefold_not_lower() -> None:
    """One normalizer everywhere, and it's casefold (D-017, D-020).

    "Straße".lower() != "STRASSE".lower(), but casefold folds both to "strasse". Using
    .lower() in one filter and casefold in another would let a subtopic be simultaneously
    "not yet explored" and "already failed twice".
    """
    state = ResearchState(question="q", failed_subtopics=["Straße", "Straße"])
    update = await run_decompose(state, plan_reply("STRASSE", "Fresh topic"))
    assert update["pending_subtopics"] == ["Fresh topic"]


@pytest.mark.asyncio
async def test_an_unsearchable_subtopic_is_dropped_before_dispatch() -> None:
    """A subtopic of only stopwords never becomes a Send (D-059, D-073).

    build_search_query raises ValueError on it, and ValueError is not on the worker's catch
    list (D-048) -- so dispatching it would crash the whole run. Catching it in the worker
    instead would burn both retry-cap attempts (D-020) on a deterministic failure.
    """
    state = ResearchState(question="q")
    update = await run_decompose(state, plan_reply("What is it?", "Positional encoding"))
    assert update["pending_subtopics"] == ["Positional encoding"]


@pytest.mark.asyncio
async def test_filtering_everything_out_yields_an_empty_list() -> None:
    """When every proposal is already explored, pending is empty -- not an error (D-069).

    This is a legitimate success state: there is nothing left to research. The routing
    function is what keeps it from ending the run silently.
    """
    state = ResearchState(question="q", explored_subtopics=["Self-attention"])
    update = await run_decompose(state, plan_reply("self-attention"))
    assert update["pending_subtopics"] == []


# ---- planner output validation -------------------------------------------------------


@pytest.mark.asyncio
async def test_unparseable_planner_output_raises() -> None:
    """A reply that isn't valid JSON fails the run rather than being swallowed (D-070).

    decompose is not a Send worker, so D-048's catch list does not apply. With one planner
    there is nothing to continue with, so catching would hide the failure (same reasoning
    as D-053).
    """
    with pytest.raises(ValidationError):
        await run_decompose(ResearchState(question="q"), "Here are some ideas: attention!")


@pytest.mark.asyncio
async def test_planner_json_with_the_wrong_shape_raises() -> None:
    """Valid JSON that doesn't match SubtopicPlan is still a ValidationError (D-013, D-070)."""
    with pytest.raises(ValidationError):
        await run_decompose(ResearchState(question="q"), '{"topics": ["attention"]}')


@pytest.mark.asyncio
async def test_the_planner_uses_the_provider_from_run_context() -> None:
    """The model is built for the run's provider, not a module-level default (D-015)."""
    factory = RecordingFactory(reply=plan_reply("Self-attention"))
    node = make_decompose(factory)
    await node(ResearchState(question="q"), _runtime("deepseek"))
    assert factory.providers == ["deepseek"]


# ---- route_subtopics (D-069) ---------------------------------------------------------


def test_route_subtopics_sends_one_worker_per_subtopic() -> None:
    """Each pending subtopic becomes its own Send, carrying seen_paper_ids (D-022, D-069).

    A worker only receives its Send payload, so anything it needs must be in there.
    """
    state = ResearchState(
        question="q",
        pending_subtopics=["Self-attention", "Positional encoding"],
        seen_paper_ids={"2411.18583"},
    )
    sends = route_subtopics(state)

    assert isinstance(sends, list)
    assert [s.node for s in sends] == ["research_worker", "research_worker"]
    assert [s.arg["subtopic"] for s in sends] == ["Self-attention", "Positional encoding"]
    # The payload carries the subtopic and nothing else (D-114). `seen_paper_ids` travelled
    # here for D-022's overlap rule, which was never implemented and which D-094 measured as
    # unable to fire -- 10% round-to-round overlap against a 60% threshold. Asserted as an
    # exact key set rather than "subtopic is present", so anything added back has to be
    # justified rather than accumulating unnoticed in a checkpointed payload.
    assert all(set(s.arg) == {"subtopic"} for s in sends)


def test_route_subtopics_skips_to_synthesize_when_nothing_is_pending() -> None:
    """An empty list must route to synthesize, never to an empty fan-out (D-069).

    Measured 2026-09-20: a conditional edge returning [] produces no error and no downstream
    node -- the graph runs the planner and stops, with no review. Everything-already-explored
    is a success, so it has to reach synthesis.
    """
    assert route_subtopics(ResearchState(question="q", pending_subtopics=[])) == "synthesize"


def test_route_subtopics_returns_sends_not_node_names_when_work_remains() -> None:
    """Pins the mixed return type: node name in one branch, Send objects in the other (D-069)."""
    sends = route_subtopics(ResearchState(question="q", pending_subtopics=["Self-attention"]))
    assert isinstance(sends, list)
    assert isinstance(sends[0], Send)
