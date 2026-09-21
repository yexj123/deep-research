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