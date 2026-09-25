"""decompose node: the planner proposes subtopics; this node decides which to dispatch (D-070)."""

from collections import Counter
from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.runtime import Runtime
from pydantic import BaseModel

from deep_research.agent.config import MAX_SUBTOPICS
from deep_research.agent.context import RunContext
from deep_research.agent.llm import ModelFactory
from deep_research.agent.sources.arxiv import build_search_query
from deep_research.agent.replies import strip_code_fence
from deep_research.agent.state import ResearchState, normalize_subtopic

DecomposeNode = Callable[[ResearchState, Runtime[RunContext]], Awaitable[dict[str, Any]]]

MAX_FAILURES = 2  # the N=2 retry cap (D-020)

SYSTEM_PROMPT = (
    "You are a research planner. Break the user's research question into distinct subtopics "
    "that can each be searched on arXiv independently.\n"
    "\n"
    "Rules:\n"
    f"- Propose at most {MAX_SUBTOPICS} subtopics.\n"
    "- Each subtopic is a short noun phrase of searchable technical terms, not a question.\n"
    "- Subtopics must not overlap with each other.\n"
    "- If an 'Already explored' list is given, propose something genuinely new.\n"
    "\n"
    'Reply with JSON only, in exactly this shape: {"subtopics": ["...", "..."]}'
)


class SubtopicPlan(BaseModel):
    """The planner's reply. External data, so it's validated here in the node (D-013, D-070)."""

    subtopics: list[str]


def _is_searchable(subtopic: str) -> bool:
    """False when build_search_query would raise: only stopwords and punctuation (D-059, D-073).

    Checked here rather than in the worker because ValueError isn't on the worker's catch
    list, so dispatching one would crash the run -- and catching it there would burn both
    retry-cap attempts on a failure that is deterministic.
    """
    try:
        build_search_query(subtopic)
    except ValueError:
        return False
    return True


def _keep_worth_researching(proposed: list[str], state: ResearchState) -> list[str]:
    """The hard filters: the prompt's explored list is only advisory (D-020, D-022, D-073)."""
    explored = {normalize_subtopic(t) for t in state.explored_subtopics}
    # Counter once, not .count() per topic: O(n+m) rather than O(n*m). Normalized, so a
    # rephrasing can't reset the retry cap.
    failures = Counter(normalize_subtopic(t) for t in state.failed_subtopics)

    return [
        topic
        for topic in proposed[:MAX_SUBTOPICS]
        if (key := normalize_subtopic(topic)) not in explored
        and failures[key] < MAX_FAILURES
        and _is_searchable(topic)
    ]


def make_decompose(model_factory: ModelFactory) -> DecomposeNode:
    """Build the decompose node with its model factory captured in a closure (D-032)."""

    async def decompose(state: ResearchState, runtime: Runtime[RunContext]) -> dict[str, Any]:
        model = model_factory(runtime.context.provider)

        explored = "\n".join(f"- {t}" for t in state.explored_subtopics)
        human = f"Research question: {state.question}"
        if explored:
            human += f"\n\nAlready explored (propose something new):\n{explored}"

        reply = await model.ainvoke([("system", SYSTEM_PROMPT), ("human", human)])
        # Catches nothing: with one planner there's nothing to continue with (D-070).
        # The fence is stripped first: D-089 measured 1 run in 10 dying here on an
        # unparseable plan, and a markdown-fenced reply is the most likely cause
        # (D-117). Still no catch -- an unparseable plan after stripping is a real
        # failure and must crash rather than silently research nothing.
        plan = SubtopicPlan.model_validate_json(strip_code_fence(reply.text))

        return {
            "pending_subtopics": _keep_worth_researching(plan.subtopics, state),
            # The baseline gap_check compares against when this round ends, to learn whether
            # the round added anything. Written here because this is where a round starts
            # (D-075).
            "seen_before_round": len(state.seen_paper_ids),
            # The same baseline for zero-result searches, written at the same moment and for
            # the same reason (D-096).
            "empty_before_round": len(state.empty_subtopics),
        }

    return decompose
