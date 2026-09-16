"""Graph wiring for Milestone 1: START -> intake -> synthesize -> END."""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from deep_research.agent.context import RunContext
from deep_research.agent.llm import ModelFactory
from deep_research.agent.nodes.intake import intake
from deep_research.agent.nodes.synthesize import make_synthesize
from deep_research.agent.state import ResearchState


def build_graph(
    model_factory: ModelFactory, checkpointer: BaseCheckpointSaver
) -> CompiledStateGraph:
    """Compile the research graph with its dependencies passed in (D-032).

    The caller owns both dependencies. Tests pass a fake factory and a fresh
    InMemorySaver; the web layer will pass get_chat_model and AsyncSqliteSaver.
    """
    builder = StateGraph(ResearchState, context_schema=RunContext)

    # Node names are part of the streaming contract: they appear as
    # metadata["langgraph_node"] and as keys in "updates" chunks.
    builder.add_node("intake", intake)
    builder.add_node("synthesize", make_synthesize(model_factory))
    builder.add_edge(START, "intake")
    builder.add_edge("intake", "synthesize")
    builder.add_edge("synthesize", END)
    return builder.compile(checkpointer=checkpointer)
