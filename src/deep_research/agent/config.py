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
