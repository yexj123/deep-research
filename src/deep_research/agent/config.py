"""Agent configuration: API key variables, model names and network limits for each LLM provider and arXiv."""

from deep_research.agent.context import ProviderType

API_KEY_ENV_VARS: dict[ProviderType, str] = {
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}

MODEL_NAMES: dict[ProviderType, str] = {"openai": "gpt-4o", "deepseek": "deepseek-flash"}

# Limits for every LLM call (D-038, D-039). The timeout covers each network wait, not
# the whole call: while streaming, it's the longest silence allowed between pieces of data;
# without streaming, it's the wait for the complete reply.
LLM_TIMEOUT_SECONDS: float = 60.0
LLM_MAX_RETRIES: int = 2

# arXiv settings (D-052). The timeout is applied by whoever creates the httpx.AsyncClient (D-049).
# Values fixed by arXiv's format (URL, XML namespaces, ID patterns) live in sources/arxiv.py.
ARXIV_MAX_RESULTS: int = 10
ARXIV_TIMEOUT_SECONDS: float = 30.0
# arXiv's terms: no more than one request every three seconds. Applied by the
# ArxivRateLimiter the caller passes into build_graph (D-042, D-064).
ARXIV_MIN_INTERVAL_SECONDS: float = 3.0

# How many subtopics the planner proposes per round (D-070). Small on purpose: arXiv access is
# serialized at one request per ARXIV_MIN_INTERVAL_SECONDS (D-064), so subtopics x rounds is
# wall-clock time. At max_depth = 2 this is 3 rounds x 3 subtopics ~ 27 s of rate-limit waiting.
MAX_SUBTOPICS: int = 3

# Recursion control, two layers (D-009).
# 1. The semantic exit: gap_check increments `depth` and stops once it exceeds MAX_DEPTH
#    (D-025, D-076). 0-indexed, so MAX_DEPTH = 2 means 3 search passes (D-026).
MAX_DEPTH: int = 2
# 2. The backstop. Measured 2026-09-21: the real graph needs a minimum of 13 -- intake (1)
#    + 3 rounds x [decompose + workers + gap_check] (9) + synthesize + check_citations (2)
#    = 12 super-steps, and LangGraph needs super-steps + 1. A Send fan-out is one super-step
#    however wide, so this value does NOT depend on MAX_SUBTOPICS. Hitting
#    GraphRecursionError means the semantic exit is broken -- fix that, not this number
#    (D-077).
RECURSION_LIMIT: int = 15

# How many papers reach the synthesis prompt, ranked by BM25 against the question (D-091).
# None means "all", which is the pre-ranking behaviour and the baseline arm -- kept switchable
# so both can be recorded from one codebase rather than from git history.
#
# 20 comes from measurement, not intuition (D-090): across ten runs, 797 papers were supplied
# and 81 cited, and the cited count was 6-12 REGARDLESS of supply. So above ~20 the extra
# context buys nothing measurable, and 20 leaves headroom over the observed maximum of 12
# because BM25 will not perfectly predict which papers the model chooses to cite.
SYNTHESIS_TOP_N: int | None = 20

# What fraction of a round's subtopics must return zero papers before the run stops (D-096).
#
# **This number is a judgement, not a measurement, and it is the only one here that is.**
# Every other constant in this file is backed by a recorded experiment. This one is not, and
# it cannot be with today's data: D-095 observed empty searches at 10.9% on narrow questions,
# but no recorded run ever had enough empty rounds to fit a threshold to. It is named rather
# than written inline as `* 2` so it can be argued with (D-098).
#
# The reasoning for 0.5, at MAX_SUBTOPICS = 3 meaning "2 of 3 empty stops the run":
# - 1.0 (all empty) would almost never fire. D-095 measured one productive subtopic out of
#   three being enough to keep a run drilling into literature that does not exist.
# - Anything at or below 1/3 stops on a single dead end, which is normal and not evidence of
#   anything -- the agent would quit on its first unlucky query.
#
# What would settle it: record an arm with 0.34 and one with 1.0 and compare papers retrieved
# against specificity. Not worth a paid sweep until a run is actually observed stopping here.
EMPTY_ROUND_RATIO: float = 0.5

# How many strongly-matching local papers count as "the corpus covers this subtopic" (O-13),
# and the top-k they are counted within.
#
# **Measured, not guessed** (D-104), against a 1669-paper corpus rebuilt from the 80 committed
# recordings. "Strongly matching" means a paper in the top-k that matches at least half the
# question's terms -- the plain top-k count that O-13 originally specified returns k for every
# query in every corpus and cannot discriminate at all.
#
#   20 in-domain questions    -> 5 to 20 strongly-matching papers
#   4 out-of-domain questions -> 0 to 1
#
# 3 sits in the middle of that gap: above every out-of-domain result, below every in-domain
# one, with margin on both sides. The narrowest in-domain margin is `spec-quant` at 5, an
# intersection question -- which is the right place for the threshold to be tightest, since
# those are the subtopics most likely to be genuinely uncovered.
#
# Re-derive with `uv run python -m tests.eval.corpus_coverage`, which needs no API key.
MIN_LOCAL_PAPERS: int = 3
LOCAL_SEARCH_TOP_K: int = 20