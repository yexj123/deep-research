"""Chat model factory: the only place that constructs real LLM clients (D-029, D-034, D-038)."""

import os
from collections.abc import Callable
from typing import assert_never

from langchain_core.language_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI

from deep_research.agent.config import (
    API_KEY_ENV_VARS,
    LLM_MAX_RETRIES,
    LLM_TIMEOUT_SECONDS,
    MODEL_NAMES,
)
from deep_research.agent.context import ProviderType

# Anything with this signature can be passed to build_graph(). Production passes
# get_chat_model; tests pass a factory that returns GenericFakeChatModel.
ModelFactory = Callable[[ProviderType], BaseChatModel]


def get_chat_model(provider: ProviderType) -> BaseChatModel:
    """Return a configured chat model for `provider`, failing loudly if it can't be built."""
    env_var = API_KEY_ENV_VARS[provider]
    api_key = os.environ.get(env_var)
    if not api_key:
        raise ValueError(f"No API key for {provider!r}: set the {env_var} environment variable.")

    model_name = MODEL_NAMES[provider]
    if not model_name:
        raise ValueError(
            f"No model configured for {provider!r}: set MODEL_NAMES[{provider!r}] in agent/config.py."
        )

    # Constructing a client makes no network request; the first call does.
    # Without explicit limits, the underlying OpenAI client has no timeout at all (D-038).
    match provider:
        case "openai":
            return ChatOpenAI(
                model=model_name,
                api_key=api_key,
                timeout=LLM_TIMEOUT_SECONDS,
                max_retries=LLM_MAX_RETRIES,
            )
        case "deepseek":
            return ChatDeepSeek(
                model=model_name,
                api_key=api_key,
                timeout=LLM_TIMEOUT_SECONDS,
                max_retries=LLM_MAX_RETRIES,
            )
        case _:
            # A type checker proves this unreachable. At runtime it catches a
            # provider added to ProviderType without a branch here.
            assert_never(provider)
