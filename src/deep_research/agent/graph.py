"""Graph wiring for milestone 4.

START -> intake -> decompose -> (Send per subtopic) -> research_worker -> gap_check
              ^                                                              |
              +------------------ depth left & new papers -------------------+
                                                                             |
                                        synthesize <-------------------------+
                                             |
                                        check_citations -> END
"""

import httpx
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from deep_research.agent.config import MAX_DEPTH
from deep_research.agent.context import RunContext
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
    """Another round only if depth is left AND the last round found something new (D-075).

    Reads the depth `gap_check` just incremented: a conditional edge sees the node's update
    already applied (confirmed 2026-09-21).

    Both conditions must hold, and they are different in kind. `depth` is the hard ceiling
    (D-009, D-026) — hitting it means the run was cut off. The new-papers check is the
    semantic exit, and it is the one that should normally fire: a round that retrieved
    nothing unseen would spend another paid planner call and another arXiv request to learn
    exactly the same thing.

    State accumulates across rounds, so "what did this round add" is not readable from
    totals. `decompose` records `seen_before_round` when the round starts; the comparison
    here is what makes the contribution measurable (D-075).
    """
    if state.depth > MAX_DEPTH:
        return "synthesize"
    if len(state.seen_paper_ids) == state.seen_before_round:
        return "synthesize"
    return "decompose"


def build_graph(
    model_factory: ModelFactory,
    http_client: httpx.AsyncClient,
    limiter: ArxivRateLimiter,
    checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph:
    """Compile the research graph with its dependencies passed in (D-032, D-049, D-064).

    The caller owns all four. Tests pass a fake factory, an httpx.MockTransport client, a
    zero-delay limiter and a fresh InMemorySaver; the web layer will pass get_chat_model, one
    shared httpx.AsyncClient, one ArxivRateLimiter and AsyncSqliteSaver. Checkpointers are
    built with serde=build_serializer() (D-014).
    """
    builder = StateGraph(ResearchState, context_schema=RunContext)

    builder.add_node("intake", intake)
    builder.add_node("decompose", make_decompose(model_factory))
    builder.add_node("research_worker", make_research_worker(http_client, limiter))
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
