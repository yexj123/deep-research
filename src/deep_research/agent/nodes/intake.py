"""intake node: where user input and run context first enter the graph (D-033)."""

from typing import Any, get_args

from langgraph.runtime import Runtime

from deep_research.agent.context import ProviderType, RunContext
from deep_research.agent.state import ResearchState

# Built from the Literal itself, so adding a provider never needs a second edit here.
VALID_PROVIDERS: frozenset[str] = frozenset(get_args(ProviderType))


def intake(state: ResearchState, runtime: Runtime[RunContext]) -> dict[str, Any]:
    """Validate the run context and the question; return the cleaned question.

    A plain `def` is fine: this node does no I/O, so it has nothing to await.
    """
    if runtime.context is None:
        raise ValueError(
            "intake: no run context. Call the graph with "
            "context=RunContext(provider=...)."
        )

    # Literal hints aren't checked at runtime: RunContext(provider="gemini") constructs fine.
    if runtime.context.provider not in VALID_PROVIDERS:
        raise ValueError(
            f"intake: unknown provider {runtime.context.provider!r}; "
            f"expected one of {sorted(VALID_PROVIDERS)}."
        )
    question = state.question.strip()
    if not question:
        raise ValueError("intake: the research question is empty or only whitespace.")

    # Return only the changed key. Changing `state` in place would be discarded.
    return {"question": question}
