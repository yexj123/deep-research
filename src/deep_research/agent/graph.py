"""Graph wiring for milestone 4, with the adaptive exits of D-096.

START -> intake -> decompose -> (Send per subtopic) -> research_worker -> gap_check
              ^                                                              |
              +--------- another round could still change the review --------+
                                                                             |
                                        synthesize <-------------------------+
                                             |
                                        check_citations -> END

The cycle back to `decompose` is the exception rather than the rule: measured over twenty
questions, 19 of 20 runs stop after a single round because that round already filled the
synthesis context (D-096). `agent/exits.py` holds the four conditions and is the only place
they are defined.
"""

import sqlite3

import httpx
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from deep_research.agent.context import RunContext
from deep_research.agent.exits import exit_reason
from deep_research.agent.llm import ModelFactory
from deep_research.agent.nodes.check_citations import check_citations
from deep_research.agent.nodes.decompose import make_decompose
from deep_research.agent.nodes.gap_check import gap_check
from deep_research.agent.nodes.intake import intake
from deep_research.agent.nodes.research_worker import make_research_worker
from deep_research.agent.nodes.synthesize import make_synthesize
from deep_research.agent.sources.rate_limit import ArxivRateLimiter
from deep_research.agent.state import ResearchState


def route_subtopics(state: ResearchState) -> str | list[Send]:
    """One worker per pending subtopic, or straight to synthesis if there are none (D-069).

    Not a node: this is the conditional edge out of decompose. The subtopic count isn't known
    until the planner runs, so a static edge can't express the fan-out.

    Returning an empty list ends the run silently -- measured 2026-09-20: no error, no
    downstream node, no review. Every proposal being already explored or past the retry cap is
    a legitimate success, so it routes to synthesize instead of falling off the end.
    """
    if not state.pending_subtopics:
        return "synthesize"
    # A worker only sees its payload, so seen_paper_ids has to travel with it (D-022).
    # A plain dict (TypedDict at the type level): Send payloads are checkpointed, and a custom
    # class comes back as a dict on resume with only a logged warning (D-071).
    return [
        Send("research_worker", {"subtopic": topic, "seen_paper_ids": state.seen_paper_ids})
        for topic in state.pending_subtopics
    ]


def route_after_gap_check(state: ResearchState) -> str:
    """Another round only if one could still change the review (D-075, D-096).

    Reads the depth `gap_check` just incremented: a conditional edge sees the node's update
    already applied (confirmed 2026-09-21).

    Four exits, and they are different in kind. Any one ending the run is enough, and they
    never disagree, so there is no precedence to document — only an order chosen cheapest
    first.

    1. **The depth ceiling** (D-009, D-026) — the hard backstop. Hitting it means the run was
       cut off, and it is the only rule that guarantees termination rather than relying on a
       heuristic over model-chosen subtopics.
    2. **No new papers at all** (D-075) — correct by construction: a round that retrieved
       nothing unseen cannot change the review. Measured as almost never firing (D-094), but
       it costs nothing and is the only exit that is true by definition rather than by
       measurement.
    3. **The synthesis prompt is already full** (D-096) — `rank_sources` truncates to
       `SYNTHESIS_TOP_N` (D-091), so past that point another round can only reshuffle which
       papers win, never add one the model sees. This is the exit that actually fires:
       measured, one round alone reaches the cap in 19 of 20 questions, and rounds 2-3 bought
       no measurable quality for 2.6x the retrieval (D-094, D-095).
    4. **The round came back empty** (D-096) — see `exits._came_back_empty`.

    The rules themselves live in `agent/exits.py`, not here, because `coverage.py` has to
    report *which* one fired and a second copy of the logic drifted from this one twice
    (D-094, D-096). This function only turns a reason into a node name.

    State accumulates across rounds, so "what did this round add" is not readable from
    totals. `decompose` records `seen_before_round` and `empty_before_round` when the round
    starts; the comparisons here are what make the round's own contribution measurable
    (D-075, D-096).

    Together these make depth *adaptive*: `MAX_DEPTH` stays a ceiling rather than a target,
    and a run that has what it needs stops on its own. That is why D-095's recommendation to
    lower `MAX_DEPTH` was implemented as these rules instead — lowering the constant would
    have bought the same saving while hiding the reason, and would have capped the one
    question in twenty that genuinely needed a second round.
    """
    reason = exit_reason(
        depth=state.depth,
        seen_count=len(state.seen_paper_ids),
        seen_before_round=state.seen_before_round,
        source_count=len(state.sources),
        dispatched=len(state.pending_subtopics),
        empties_this_round=len(state.empty_subtopics) - state.empty_before_round,
    )
    return "synthesize" if reason else "decompose"


def build_graph(
    model_factory: ModelFactory,
    http_client: httpx.AsyncClient,
    limiter: ArxivRateLimiter,
    checkpointer: BaseCheckpointSaver,
    corpus: sqlite3.Connection | None = None,
) -> CompiledStateGraph:
    """Compile the research graph with its dependencies passed in (D-032, D-049, D-064).

    The caller owns all of them. Tests pass a fake factory, an httpx.MockTransport client, a
    zero-delay limiter and a fresh InMemorySaver; the web layer will pass get_chat_model, one
    shared httpx.AsyncClient, one ArxivRateLimiter and AsyncSqliteSaver. Checkpointers are
    built with serde=build_serializer() (D-014).

    `corpus` is optional and defaults to off (O-13, D-101). With a connection, every search
    result is indexed on the way through -- free, since those abstracts were already fetched.
    Without one, behaviour is exactly as before, so the no-corpus arm stays reproducible from
    this codebase rather than from git history (the D-091 pattern). The eval recorder passes
    nothing, which is what keeps the 80 committed recordings comparable.
    """
    builder = StateGraph(ResearchState, context_schema=RunContext)

    builder.add_node("intake", intake)
    builder.add_node("decompose", make_decompose(model_factory))
    builder.add_node("research_worker", make_research_worker(http_client, limiter, corpus))
    builder.add_node("gap_check", gap_check)
    builder.add_node("synthesize", make_synthesize(model_factory))
    builder.add_node("check_citations", check_citations)

    builder.add_edge(START, "intake")
    builder.add_edge("intake", "decompose")
    # The third argument lists every node route_subtopics can reach, so LangGraph can draw
    # and validate the graph -- it can't infer them from the function body.
    builder.add_conditional_edges("decompose", route_subtopics, ["research_worker", "synthesize"])
    builder.add_edge("research_worker", "gap_check")
    # The cycle: back to decompose for another round, or out to synthesize (D-075).
    builder.add_conditional_edges(
        "gap_check", route_after_gap_check, ["decompose", "synthesize"]
    )
    builder.add_edge("synthesize", "check_citations")
    builder.add_edge("check_citations", END)
    return builder.compile(checkpointer=checkpointer)
