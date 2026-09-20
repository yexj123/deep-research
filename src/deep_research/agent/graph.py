"""Graph wiring for milestone 2: START -> intake -> search -> synthesize -> check_citations -> END."""

import httpx
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from deep_research.agent.context import RunContext
from deep_research.agent.llm import ModelFactory
from deep_research.agent.nodes.check_citations import check_citations
from deep_research.agent.nodes.intake import intake
from deep_research.agent.nodes.search import make_search
from deep_research.agent.nodes.synthesize import make_synthesize
from deep_research.agent.state import ResearchState


def build_graph(
    model_factory: ModelFactory,
    http_client: httpx.AsyncClient,
    checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph:
    """Compile the research graph with its dependencies passed in (D-032, D-049).

    The caller owns all three dependencies. Tests pass a fake factory, an httpx.MockTransport
    client and a fresh InMemorySaver; the web layer will pass get_chat_model, one shared
    httpx.AsyncClient and AsyncSqliteSaver. Checkpointers use serde=build_serializer() (D-014).
    """
    builder = StateGraph(ResearchState, context_schema=RunContext)

    builder.add_node("intake", intake)
    builder.add_node("search", make_search(http_client))
    builder.add_node("synthesize", make_synthesize(model_factory))
    builder.add_node("check_citations", check_citations)

    builder.add_edge(START, "intake")
    builder.add_edge("intake", "search")
    builder.add_edge("search", "synthesize")
    builder.add_edge("synthesize", "check_citations")
    builder.add_edge("check_citations", END)
    return builder.compile(checkpointer=checkpointer)
