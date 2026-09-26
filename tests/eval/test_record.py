"""Record real agent runs over the frozen question set (O-11).

**Paid and slow.** Ten questions at roughly 35 seconds each, each making several model calls
and up to nine arXiv requests spaced by the real rate limiter (D-064). Deselected by default;
run deliberately:

    uv run pytest -m record

This is not a test of correctness -- it produces the artifact that `test_review_quality.py`
scores. It fails only if the agent cannot complete a run at all, which is worth knowing
before spending money on judging.
"""

import asyncio
import json
import os
import sqlite3
import time
from datetime import UTC, datetime

import httpx
import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from deep_research.agent.config import (
    ARXIV_MIN_INTERVAL_SECONDS,
    ARXIV_TIMEOUT_SECONDS,
    LOCAL_FIRST,
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
from deep_research.persistence.corpus import SCHEMA as CORPUS_SCHEMA
from deep_research.persistence.corpus import index_sources
from tests.eval.recording import (
    DEFAULT_QUESTION_SET,
    Recording,
    load_questions,
    save,
)

PROVIDER = "openai"

# Spacing has to carry ACROSS tests, but a limiter cannot. Each question gets its own
# ArxivRateLimiter for within-run concurrency; what is shared is only the timestamp of the
# last arXiv request in this process.
#
# A module-level limiter was the obvious fix and is wrong for a reason D-064 already
# documents: `asyncio.Semaphore` binds to the first event loop that touches it, and
# pytest-asyncio gives each test a fresh loop -- so nine of ten questions died with
# RuntimeError on the second loop. Having written that warning and then hit it anyway, the
# timestamp is a plain float precisely because it belongs to no event loop.
#
# The problem is real: a replay script with per-question limiters drew HTTP 429 from arXiv
# (2026-09-21). The app itself is fine -- create_app builds one limiter inside one lifespan.
_LAST_ARXIV_REQUEST = 0.0


async def _space_from_previous_question() -> None:
    """Wait out arXiv's interval since the previous test's last request (D-064)."""
    global _LAST_ARXIV_REQUEST
    wait = ARXIV_MIN_INTERVAL_SECONDS - (time.monotonic() - _LAST_ARXIV_REQUEST)
    if wait > 0:
        await asyncio.sleep(wait)

# Experiment override, e.g. EVAL_MAX_DEPTH=0 to record a single-round arm.
# MAX_DEPTH is a module constant, so `from ..config import MAX_DEPTH` binds a *copy* into each
# importing module's namespace and patching config.py alone changes nothing. Every binding has
# to be patched, which is why _DEPTH_MODULES is a list rather than just graph.py: the first
# version of this patched graph.py only, and coverage.py then reported the d0 and d1 arms as
# having *converged* ("found no papers earlier rounds hadn't seen") when they had in fact hit
# their ceiling -- the run routed correctly, but the artifact described it wrongly. That is the
# D-062/D-069 failure mode again, this time in the measuring instrument.
# Deliberately an env var rather than a src change: the question "does the recursion earn its
# cost?" should be answered before the architecture is reshaped around either answer.
EVAL_MAX_DEPTH = int(os.environ["EVAL_MAX_DEPTH"]) if "EVAL_MAX_DEPTH" in os.environ else MAX_DEPTH

# Which frozen question set to record, e.g. EVAL_QUESTION_SET=narrow for O-14's intersection
# questions. Unlike the depth override this patches nothing -- it selects an input file -- but
# it lands in `settings` for the same reason: the arm name must say which questions were asked,
# or two arms that asked different things look comparable.
EVAL_QUESTION_SET = os.environ.get("EVAL_QUESTION_SET", DEFAULT_QUESTION_SET)

# EVAL_LOCAL_FIRST=1 records the O-13 local-first arm (D-105): a subtopic the corpus covers is
# answered from it and arXiv is not called. Like EVAL_MAX_DEPTH this patches the module that
# imported the constant, not config -- `from ..config import LOCAL_FIRST` binds a copy
# (the D-094 lesson).
# Defaults to config's value (True since D-107). Set EVAL_LOCAL_FIRST=0 to record the
# arXiv-only arm -- both directions stay reproducible from one codebase whichever way the
# default points, which is what keeps the pre-D-107 recordings comparable (D-088, D-091).
EVAL_LOCAL_FIRST = (
    os.environ["EVAL_LOCAL_FIRST"] not in ("0", "false", "")
    if "EVAL_LOCAL_FIRST" in os.environ
    else LOCAL_FIRST
)

# The corpus is rebuilt in memory from the committed recordings rather than kept as a file:
# 1669 papers, indexed in about a second, and reproducible by anyone who has the repo. A
# checked-in 7 MB binary would be neither.
#
# This is the realistic scenario local-first exists for -- someone who has researched these
# topics before -- and it is deliberately generous: the corpus holds papers from earlier runs
# of these exact questions. If local-first does not help here, it will not help anywhere.
_CORPUS: sqlite3.Connection | None = None


# EVAL_FULL_TEXT=1 records against the enriched corpus on disk (D-109) instead of the
# abstracts-only one rebuilt in memory. The corpus *contents* are the variable -- the retrieval
# code is identical, which is what makes the two arms differ by one thing (D-088).
EVAL_FULL_TEXT = os.environ.get("EVAL_FULL_TEXT", "") not in ("", "0", "false")

# EVAL_STALE_CORPUS=1 records O-16's stale arm: retrieval reads the corpus snapshot seeded on
# a known past date (tests/eval/snapshots/), so the run can only cite papers that existed
# then. Again the corpus *contents* are the only variable.
#
# It must reach `retrieval_unit` below, or a stale run would be filed under `abstract-local`
# alongside recordings made from a fresh corpus -- two different experiments under one arm
# name, which is the D-094 mislabelling that `arm_name` exists to make impossible.
EVAL_STALE_CORPUS = os.environ.get("EVAL_STALE_CORPUS", "") not in ("", "0", "false")


def _eval_corpus() -> sqlite3.Connection | None:
    """The seeded corpus, or None when the arm does not use one.

    Abstracts-only is rebuilt in memory from the committed recordings, so it is reproducible
    by anyone with the repo. The full-text corpus is a file because enriching it costs ~18
    minutes of rate-limited downloads; build it with `python -m tests.eval.enrich`.
    """
    global _CORPUS
    if not EVAL_LOCAL_FIRST:
        return None
    if _CORPUS is None:
        from tests.eval.corpus_coverage import all_recorded_papers

        if EVAL_STALE_CORPUS:
            # O-16's stale arm. A file, not an in-memory rebuild, because its value is its
            # `indexed_at` dates -- rebuilding would stamp today and destroy the one property
            # the experiment depends on. `--rebuild` restores it from the committed manifest
            # with the original dates if the file is ever lost.
            from deep_research.persistence.corpus import connect
            from tests.eval.seed_corpus import SNAPSHOT_DB

            if not SNAPSHOT_DB.exists():
                raise RuntimeError(
                    f"{SNAPSHOT_DB} does not exist. Restore it from the committed manifest"
                    " with: uv run python -m tests.eval.seed_corpus --rebuild"
                )
            _CORPUS = connect(str(SNAPSHOT_DB))
        elif EVAL_FULL_TEXT:
            from tests.eval.enrich import CORPUS_PATH
            from deep_research.persistence.corpus import connect

            if not CORPUS_PATH.exists():
                raise RuntimeError(
                    f"{CORPUS_PATH} does not exist. Build it first with:"
                    " uv run python -m tests.eval.enrich"
                )
            _CORPUS = connect(str(CORPUS_PATH))
        else:
            _CORPUS = sqlite3.connect(":memory:", check_same_thread=False)
            _CORPUS.executescript(CORPUS_SCHEMA)
            index_sources(_CORPUS, all_recorded_papers())
    return _CORPUS


def monkeypatch_local_first() -> None:
    """Apply EVAL_LOCAL_FIRST to the module research_worker actually reads.

    Sets it either way rather than only when enabling: once the default flipped to True
    (D-107), an arXiv-only arm needs it turned *off*, and a one-directional patch would have
    recorded that arm under the wrong settings while looking correct.
    """
    import deep_research.agent.nodes.research_worker as worker_module

    worker_module.LOCAL_FIRST = EVAL_LOCAL_FIRST


def _days_since_seed() -> int | None:
    """Whole days between the O-16 corpus seed and now, or None outside the staleness set.

    Derived from the committed manifest and the clock, never passed in: it lands in the arm
    name, and a hand-set label is exactly the mislabelling `arm_name` exists to prevent.
    """
    if EVAL_QUESTION_SET != "staleness":
        return None
    from tests.eval.seed_corpus import MANIFEST_FILE

    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    seeded_at = datetime.fromisoformat(manifest["seeded_at"])
    return (datetime.now(UTC) - seeded_at).days


def _settings() -> dict[str, object]:
    """Everything that makes one recording incomparable to another if it differs."""
    age = _days_since_seed()
    return {
        # Only for the staleness set, so no other arm name changes (O-16).
        **({} if age is None else {"days_since_seed": age}),
        "provider": PROVIDER,
        "model": MODEL_NAMES[PROVIDER],
        "max_depth": EVAL_MAX_DEPTH,
        "max_subtopics": MAX_SUBTOPICS,
        # What is retrieved and from where. "abstract" is arXiv only; "abstract-local" adds
        # D-105's corpus-first path; "full_text" comes later. Reusing this key rather than
        # adding a dimension keeps the 80 committed arm names stable (D-088).
        "retrieval_unit": (
            (
                "abstract-stale"
                if EVAL_STALE_CORPUS
                else "fulltext-local"
                if EVAL_FULL_TEXT
                else "abstract-local"
            )
            if EVAL_LOCAL_FIRST
            else "abstract"
        ),
        "synthesis_top_n": SYNTHESIS_TOP_N,  # None = the pre-ranking baseline arm (D-091)
        "question_set": EVAL_QUESTION_SET,  # which frozen set was asked (O-14)
        # "adaptive" once D-096's exits landed: the run may stop well before max_depth, so
        # the ceiling alone no longer describes how many rounds happened.
        "exit_rule": "adaptive",
    }


# Every module that binds MAX_DEPTH at import time. graph.py decides routing; coverage.py
# decides what the recording *says* about why the run stopped. Missing either one produces a
# recording that is internally inconsistent, so they are patched together, from one list.
_DEPTH_MODULES = ("deep_research.agent.graph", "deep_research.agent.coverage")


def monkeypatch_depth() -> None:
    """Apply the EVAL_MAX_DEPTH override to every module that imported MAX_DEPTH.

    Pins the fix for the mislabelling described above: routing and reporting must agree about
    what the ceiling is, or a ceiling stop gets recorded as a convergence.
    """
    if EVAL_MAX_DEPTH == MAX_DEPTH:
        return
    import importlib

    for name in _DEPTH_MODULES:
        importlib.import_module(name).MAX_DEPTH = EVAL_MAX_DEPTH


@pytest.mark.record
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
@pytest.mark.asyncio
@pytest.mark.parametrize("case", load_questions(EVAL_QUESTION_SET), ids=lambda c: c["id"])
async def test_record_a_run(case: dict[str, str]) -> None:
    """Run the agent once for real and save the result for later scoring.

    One test per question rather than one test looping over all ten: a failure then names the
    question that failed, and re-recording a single question is `-m record -k <id>` rather
    than repeating the whole paid set.
    """
    monkeypatch_depth()
    monkeypatch_local_first()
    await _space_from_previous_question()
    config: RunnableConfig = {
        "configurable": {"thread_id": f"eval-{case['id']}"},
        "recursion_limit": RECURSION_LIMIT,
    }
    async with httpx.AsyncClient(timeout=ARXIV_TIMEOUT_SECONDS) as http_client:
        graph = build_graph(
            get_chat_model,
            http_client,
            ArxivRateLimiter(ARXIV_MIN_INTERVAL_SECONDS),
            InMemorySaver(serde=build_serializer()),
            _eval_corpus(),
        )
        await graph.ainvoke(
            {"question": case["question"]},
            config,
            context=RunContext(provider=PROVIDER),
            version="v2",
        )
        values = (await graph.aget_state(config)).values

    global _LAST_ARXIV_REQUEST
    _LAST_ARXIV_REQUEST = time.monotonic()

    assert values.get("review"), f"{case['id']}: the run produced no review"

    # A run whose every subtopic failed is an OUTAGE, not evidence -- and it passes the check
    # above, because the zero-papers review (D-060) is non-empty prose. Recording it would file
    # an arXiv rate-limit incident as a legitimate zero-result measurement, which is this
    # project's recurring failure shape arriving inside its own evaluation harness.
    #
    # A genuine "nothing published on X" run is different: those subtopics land in
    # empty_subtopics having succeeded (D-021), not in failed_subtopics.
    coverage = summarize_coverage(values)
    assert coverage.explored or not coverage.failed, (
        f"{case['id']}: every subtopic failed ({list(coverage.failed)}) -- this is an outage, "
        "not data. Check arXiv is reachable and not rate-limiting, then re-record."
    )

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
            coverage=vars(coverage),
            settings=_settings(),
            recorded_at=Recording.now(),
        )
    )
    print(f"\nrecorded {case['id']}: {len(sources)} paper(s) -> {path.name}")
