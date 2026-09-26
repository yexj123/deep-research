"""Turning a follow-up into a standalone research question (D-121).

**What a follow-up actually needs, and what it does not.** The original design assumed it
needed `sources`, `seen_paper_ids` and `explored_subtopics` carried forward in graph state.
Measured against a 60-paper corpus built by two real runs, **4 of 5 follow-up-shaped subtopics
were already covered locally**, and the fifth correctly fell through to arXiv. The corpus
(D-105) already provides research continuity -- across *every* thread, filtered by relevance
per subtopic -- which is strictly better than dragging a thread's state along:

| | corpus | carried state |
|---|---|---|
| scope | every run ever | one thread |
| selection | BM25 per subtopic | everything, unfiltered |

And carrying `explored_subtopics` forward would actively break a follow-up: the planner filter
would refuse to revisit any subtopic the first question explored, which is precisely what
"tell me more about that" asks for.

**So a follow-up is a fresh run, and the only thing it inherits is language.** "What about
quantization?" is not a research question -- it is a question *about* a previous answer. This
module resolves it into one, and then the ordinary pipeline handles it with no special cases:
no state migration, no `depth` reset, no checkpoint surgery, no second code path to keep
correct.

**Deterministic first, model second.** A question that already stands alone is left untouched,
so the common case costs nothing and cannot be corrupted by a rewrite.
"""

import re
from collections.abc import Awaitable, Callable

from deep_research.agent.context import ProviderType
from deep_research.agent.llm import ModelFactory
from deep_research.agent.replies import strip_code_fence

# Openings and pronouns that only mean something relative to a previous turn. A question
# starting "what about..." or containing a bare "it"/"they" cannot be searched on its own.
NEEDS_CONTEXT: re.Pattern[str] = re.compile(
    r"\b(it|its|they|them|their|that|those|these|this|instead|also|too|compared)\b"
    r"|^\s*(what about|how about|and\b|but\b|why\b|so\b)",
    re.I,
)

# How much of the previous review the rewriter is shown. The first paragraphs carry the
# overview; the rest is detail the rewriter does not need and would pay for.
CONTEXT_CHARS = 1200

SYSTEM_PROMPT = (
    "You rewrite a follow-up question into one that stands on its own.\n"
    "\n"
    "You are given a previous research question, the start of the answer it produced, and a "
    "follow-up. Replace pronouns and implicit references with what they refer to, so the "
    "result can be searched without any of that context.\n"
    "\n"
    "Rules:\n"
    "- Keep the user's intent exactly. Do not broaden, narrow, or answer it.\n"
    "- Add only what is needed to make it self-contained. Do not append topics they did not "
    "ask about.\n"
    "- If the follow-up already stands alone, return it unchanged.\n"
    "- Reply with the rewritten question and nothing else: no quotes, no preamble, no "
    "explanation."
)

Rewriter = Callable[[str, str, str, ProviderType, str | None], Awaitable[str]]


def needs_context(question: str) -> bool:
    """Does this question depend on a previous turn to make sense?

    Deliberately generous: a false positive costs one cheap model call that returns the
    question unchanged, while a false negative sends "what about it?" to arXiv as a literal
    search and produces a confidently irrelevant review.
    """
    return bool(NEEDS_CONTEXT.search(question.strip()))


def make_rewriter(model_factory: ModelFactory) -> Rewriter:
    """Build the rewriter with its model factory captured in a closure (D-032)."""

    async def rewrite(
        question: str,
        parent_question: str,
        parent_review: str,
        provider: ProviderType,
        model: str | None = None,
    ) -> str:
        """The follow-up as a standalone question, or unchanged if it already is one.

        **Never raises, and never returns empty.** A rewrite failure must not cost the user
        their question: the original is a worse search than the rewrite but a far better one
        than nothing, so every failure path returns it. Same reasoning as `check_claims`
        (D-113) -- this is additive, and additive things must not break what they augment.
        """
        if not needs_context(question):
            return question

        human = (
            f"Previous question: {parent_question}\n\n"
            f"Previous answer (beginning):\n{parent_review[:CONTEXT_CHARS]}\n\n"
            f"Follow-up: {question}"
        )
        try:
            reply = await model_factory(provider, model).ainvoke(
                [("system", SYSTEM_PROMPT), ("human", human)]
            )
            rewritten = strip_code_fence(reply.text).strip().strip('"')
        except Exception:  # noqa: BLE001 - see the docstring: never cost the user their question
            return question

        # A rewrite that lost the question, or that returned an essay instead of a question,
        # is worse than no rewrite. Bounded generously: the point is to catch a runaway reply,
        # not to police phrasing.
        if not rewritten or len(rewritten) > 4 * max(len(question), 80):
            return question
        return rewritten

    return rewrite
