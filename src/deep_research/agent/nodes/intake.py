"""intake node: where user input and run context first enter the graph (D-033)."""

import re
from typing import Any, get_args

from langgraph.runtime import Runtime

from deep_research.agent.config import MODEL_NAME_PATTERN
from deep_research.agent.context import ProviderType, RunContext
from deep_research.agent.state import ResearchState

# Built from the Literal itself, so adding a provider never needs a second edit here.
VALID_PROVIDERS: frozenset[str] = frozenset(get_args(ProviderType))

_MODEL_NAME = re.compile(MODEL_NAME_PATTERN)


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
    # `None` means "use the provider's default" (D-125, D-126), so only a chosen one is
    # checked. Checked here, in the first node, because every later use of it costs money: a
    # malformed name would otherwise surface as a provider error partway through decompose,
    # after the run had already started streaming.
    model = runtime.context.model
    if model and not _MODEL_NAME.fullmatch(model):
        raise ValueError(
            f"intake: {model!r} is not a usable model name. Expected something like "
            "'gpt-4o-mini' or 'deepseek-chat': letters, digits, and . _ - : / only."
        )

    question = state.question.strip()
    if not question:
        raise ValueError("intake: the research question is empty or only whitespace.")

    # Return only the changed key. Changing `state` in place would be discarded.
    return {"question": question}
