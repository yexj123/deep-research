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