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

# Whether a covered subtopic is answered from the corpus *instead of* searching arXiv (O-13).
#
# **On since D-107, which measured it** over 20 paired questions in two difficulty classes:
#
#   arXiv requests   65 -> 1        wall clock  703s -> 248s  (-65%)
#   specificity      0.78 -> 0.78   faithfulness 0.98 -> 0.99   papers cited 7.95 -> 7.90
#
# Every quality metric inside 2 SE; the two that moved at all moved upward. The mechanism is
# D-105's: two runs of the same question retrieve ~89% different papers yet produce
# indistinguishable reviews, so paper *identity* does not drive quality -- topical relevance
# does, and the corpus has that.
#
# **Safe on a cold corpus by construction.** An empty corpus covers nothing, so a first run
# falls through to arXiv and behaves exactly as before. The feature can only fire where it
# has evidence to fire on.
#
# **The risk it does carry is staleness, and this experiment cannot see it**: the corpus was
# measured at one moment in time. A topic researched repeatedly could stay answered from
# cache indefinitely while arXiv moves on. That is reported in the coverage panel (D-105)
# rather than silently defended against, and the open item says what would settle it.
LOCAL_FIRST: bool = True

# How much full-text excerpt each paper contributes to the synthesis prompt (D-110).
#
# 0 disables excerpts entirely, so the prompt sees abstracts even when the corpus has read the
# paper -- the switchable baseline that keeps the abstracts-only arm reproducible (D-088).
#
# **0 since D-111, which measured it and found it does not pay.** Against the abstracts
# baseline, 2.8x the prompt characters bought:
#
#   specificity      0.78 -> 0.79  (+0.9 SE)   numeric_density  0.51 -> 0.72  (+1.0 SE)
#   faithfulness     0.99 -> 0.97  (-1.9 SE)   papers retrieved 51.8 -> 37.9  (-8.9 SE)
#
# Nothing clears 2 SE in its favour, faithfulness is nominally *down*, and breadth falls 27%
# because one enriched paper occupies several top-k slots. Specificity had measured headroom
# (D-093: 0.95 concrete, 0.22 vague) and did not move, so this is a null from an instrument
# capable of detecting the effect -- not an absent measurement.
#
# 4000 (two chunks) is the value the arm was recorded at, so setting it there reproduces
# D-111 exactly. Raising it spends more of what D-092's pruning saved.
#
# **Coupled to the retrieval tier on purpose**: with this at 0, `_local_answer` restricts
# retrieval to abstracts, because selecting papers on text the model will never read measured
# *worse* than not having full text at all (D-110).
EXCERPT_MAX_CHARS: int = 0