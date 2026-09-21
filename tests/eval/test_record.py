"""Record real agent runs over the frozen question set (O-11).

**Paid and slow.** Ten questions at roughly 35 seconds each, each making several model calls
and up to nine arXiv requests spaced by the real rate limiter (D-064). Deselected by default;
run deliberately:

    uv run pytest -m record

This is not a test of correctness -- it produces the artifact that `test_review_quality.py`
scores. It fails only if the agent cannot complete a run at all, which is worth knowing
before spending money on judging.
"""

import os

import httpx
import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from deep_research.agent.config import (
    ARXIV_MIN_INTERVAL_SECONDS,
    ARXIV_TIMEOUT_SECONDS,
    MAX_DEPTH,
    MAX_SUBTOPICS,
    MODEL_NAMES,
    RECURSION_LIMIT,
    SYNTHESIS_TOP_N,
)
from deep_research.agent.context import RunContext
from deep_research.agent.coverage import summarize_coverage
from deep_research.agent.graph import build_graph
from deep_research.agent.llm import get_chat_model
from deep_research.agent.nodes.synthesize import format_papers
from deep_research.agent.sources.rate_limit import ArxivRateLimiter
from deep_research.persistence.checkpointer import build_serializer
from tests.eval.recording import Recording, load_questions, save

PROVIDER = "openai"


def _settings() -> dict[str, object]:
    """Everything that makes one recording incomparable to another if it differs."""
    return {
        "provider": PROVIDER,
        "model": MODEL_NAMES[PROVIDER],
        "max_depth": MAX_DEPTH,
        "max_subtopics": MAX_SUBTOPICS,
        "retrieval_unit": "abstract",  # O-13 will produce recordings with "full_text"
        "synthesis_top_n": SYNTHESIS_TOP_N,  # None = the pre-ranking baseline arm (D-091)
    }


@pytest.mark.record
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
@pytest.mark.asyncio
@pytest.mark.parametrize("case", load_questions(), ids=lambda c: c["id"])
async def test_record_a_run(case: dict[str, str]) -> None:
    """Run the agent once for real and save the result for later scoring.

    One test per question rather than one test looping over all ten: a failure then names the
    question that failed, and re-recording a single question is `-m record -k <id>` rather
    than repeating the whole paid set.
    """
    config: RunnableConfig = {
        "configurable": {"thread_id": f"eval-{case['id']}"},
        "recursion_limit": RECURSION_LIMIT,
    }
    limiter = ArxivRateLimiter(ARXIV_MIN_INTERVAL_SECONDS)  # the real one: arXiv's terms apply

    async with httpx.AsyncClient(timeout=ARXIV_TIMEOUT_SECONDS) as http_client:
        graph = build_graph(
            get_chat_model, http_client, limiter, InMemorySaver(serde=build_serializer())
        )
        await graph.ainvoke(
            {"question": case["question"]},
            config,
            context=RunContext(provider=PROVIDER),
            version="v2",
        )
        values = (await graph.aget_state(config)).values

    assert values.get("review"), f"{case['id']}: the run produced no review"

    # Exactly the papers that reached the prompt, which is what faithfulness must judge each
    # claim against (D-091). NOT every retrieved paper: once ranking prunes, those differ, and
    # judging a 20-paper review against 81 papers would score a context the model never saw --
    # making the two arms quietly incomparable while every test still passed.
    shown = set(values.get("synthesized_from", []))
    sources = [s for s in values.get("sources", []) if not shown or s.arxiv_id in shown]
    context = [format_papers([source]) for source in sources]

    path = save(
        Recording(
            id=case["id"],
            question=case["question"],
            review=values["review"],
            retrieval_context=context,
            citation_violations=values.get("citation_violations", []),
            papers_retrieved=len(values.get("sources", [])),
            coverage=vars(summarize_coverage(values)),
            settings=_settings(),
            recorded_at=Recording.now(),
        )
    )
    print(f"\nrecorded {case['id']}: {len(sources)} paper(s) -> {path.name}")
