"""synthesize node: turns the question (later: the findings) into the review text."""

from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.runtime import Runtime

from deep_research.agent.context import RunContext
from deep_research.agent.llm import ModelFactory
from deep_research.agent.state import ResearchState

SynthesizeNode = Callable[[ResearchState, Runtime[RunContext]], Awaitable[dict[str, Any]]]

# Milestone 1 only: nothing has been searched yet. Replace this at milestone 2, when
# the node receives real sources and every claim must cite one of them.
SYSTEM_PROMPT = (
    "You are a research assistant writing a short, literature-review-style "
    "overview of the user's research question.\n"
    "\n"
    "No literature search was performed for this answer, and you have no "
    "retrieved papers. Therefore:\n"
    "- Begin with one sentence stating that no sources were consulted and the "
    "content has not been verified against the literature.\n"
    "- Answer only from general knowledge.\n"
    "- Do not cite, quote, or name specific papers, authors, DOIs, arXiv IDs, "
    "or URLs, and never invent references.\n"
    "- Where you are uncertain or the field is unsettled, say so plainly "
    "instead of guessing.\n"
    "\n"
    "Structure the answer with Markdown headings: Overview, Main Approaches, "
    "Open Problems, Summary. Keep it under 500 words."
)


def make_synthesize(model_factory: ModelFactory) -> SynthesizeNode:
    """Build the synthesize node with its model factory captured in a closure (D-032).

    build_graph() calls this once; LangGraph then calls the returned function
    for every run. Tests pass a factory that returns GenericFakeChatModel.
    """

    async def synthesize(
        state: ResearchState, runtime: Runtime[RunContext]
    ) -> dict[str, Any]:
        model = model_factory(runtime.context.provider)

        messages = [
            ("system", SYSTEM_PROMPT),
            ("human", state.question),
        ]
        # No manual streaming: the "messages" stream mode picks up this call's tokens.
        reply = await model.ainvoke(messages)
        return {"review": reply.text}

    return synthesize
