"""Follow-up questions become standalone ones (D-121).

**What these pin is a design decision, not just behaviour.** The original plan for multi-turn
sessions was to carry `sources`, `seen_paper_ids` and `explored_subtopics` forward in graph
state. Measured against a 60-paper corpus from two real runs, 4 of 5 follow-up-shaped
subtopics were *already* covered locally -- so the corpus (D-105) supplies research continuity
across every thread, filtered per subtopic, which carried state cannot match.

Carrying `explored_subtopics` would have been worse than redundant: the planner filter would
refuse to revisit anything the first question explored, which is exactly what "tell me more
about that" asks for.

So a follow-up is an ordinary run, and the only thing it inherits is language. That keeps one
code path instead of two, and these tests exist to stop the second one growing back.
"""

import pytest

from deep_research.agent.followup import make_rewriter, needs_context


class _Factory:
    """Records the prompt and returns a scripted reply, or raises."""

    def __init__(self, reply: str = "", boom: bool = False) -> None:
        self.reply = reply
        self.boom = boom
        self.prompts: list[str] = []

    def __call__(self, provider):  # noqa: ANN001 - matches ModelFactory
        outer = self

        class _Model:
            async def ainvoke(self, messages):  # noqa: ANN001
                outer.prompts.append(messages[-1][1])
                if outer.boom:
                    raise RuntimeError("model unavailable")

                class _Reply:
                    text = outer.reply

                return _Reply()

        return _Model()


PARENT_Q = "How does speculative decoding speed up language model inference?"
PARENT_A = "Speculative decoding uses a small draft model to propose tokens..."


# ---- the cheap check that runs first --------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "What about quantization?",
        "How does it compare to pruning?",
        "Why is that the case?",
        "And the memory cost?",
        "Does this work for vision models too?",
    ],
)
def test_a_context_dependent_question_is_detected(question: str) -> None:
    """These cannot be searched alone: sent to arXiv verbatim they return confident nonsense."""
    assert needs_context(question)


@pytest.mark.parametrize(
    "question",
    [
        "What are the tradeoffs of post-training quantization for large language models?",
        "How do mixture-of-experts architectures scale?",
    ],
)
def test_a_standalone_question_is_left_alone(question: str) -> None:
    """The common case must cost nothing and must not be rewritten.

    A rewrite it did not need is a chance to corrupt a question that was already correct.
    """
    assert not needs_context(question)


@pytest.mark.asyncio
async def test_a_standalone_question_makes_no_model_call() -> None:
    """The detector is what keeps the common case free."""
    factory = _Factory(reply="should not be used")
    rewrite = make_rewriter(factory)

    out = await rewrite("How do MoE architectures scale?", PARENT_Q, PARENT_A, "openai")

    assert out == "How do MoE architectures scale?"
    assert factory.prompts == []


# ---- the rewrite ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_follow_up_is_made_standalone() -> None:
    """The whole point: "what about X" becomes something arXiv can be asked."""
    factory = _Factory(
        reply="How does post-training quantization compare to speculative decoding for "
        "language model inference speed?"
    )
    out = await make_rewriter(factory)("What about quantization?", PARENT_Q, PARENT_A, "openai")

    assert "quantization" in out and "speculative decoding" in out
    assert "what about" not in out.lower()


@pytest.mark.asyncio
async def test_the_rewriter_sees_the_previous_question_and_answer() -> None:
    """It cannot resolve "it" without knowing what "it" was.

    Asserted on the prompt actually sent, not on the output -- the output comes from a stub
    and would pass however little context the node supplied.
    """
    factory = _Factory(reply="rewritten")
    await make_rewriter(factory)("What about it?", PARENT_Q, PARENT_A, "openai")

    prompt = factory.prompts[0]
    assert PARENT_Q in prompt
    assert PARENT_A[:40] in prompt
    assert "What about it?" in prompt


@pytest.mark.asyncio
async def test_surrounding_quotes_are_stripped() -> None:
    """Models quote their answers, and a leading quote would reach `build_search_query`."""
    factory = _Factory(reply='"How does quantization affect inference latency?"')
    out = await make_rewriter(factory)("What about it?", PARENT_Q, PARENT_A, "openai")

    assert out == "How does quantization affect inference latency?"


# ---- failure must never cost the question ---------------------------------------------


@pytest.mark.asyncio
async def test_a_model_failure_returns_the_original_question() -> None:
    """Additive things must not break what they augment (the D-113 rule).

    The original is a worse search than the rewrite and a far better one than nothing.
    """
    out = await make_rewriter(_Factory(boom=True))("What about it?", PARENT_Q, PARENT_A, "openai")
    assert out == "What about it?"


@pytest.mark.asyncio
async def test_an_empty_rewrite_returns_the_original_question() -> None:
    """A rewrite that lost the question is worse than no rewrite."""
    out = await make_rewriter(_Factory(reply="   "))("What about it?", PARENT_Q, PARENT_A, "openai")
    assert out == "What about it?"


@pytest.mark.asyncio
async def test_a_runaway_rewrite_returns_the_original_question() -> None:
    """A model that answers instead of rewriting must not have its essay searched.

    The bound is generous on purpose -- it exists to catch a reply that stopped being a
    question, not to police phrasing.
    """
    factory = _Factory(reply="Well, " + "this is a long explanation. " * 40)
    out = await make_rewriter(factory)("What about it?", PARENT_Q, PARENT_A, "openai")

    assert out == "What about it?"
