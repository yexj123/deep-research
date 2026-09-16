"""Client factory tests (D-029, D-034, D-038). No network: constructing a client sends no request."""

import pytest
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI

from deep_research.agent.config import API_KEY_ENV_VARS, LLM_MAX_RETRIES, LLM_TIMEOUT_SECONDS
from deep_research.agent.context import ProviderType
from deep_research.agent.llm import get_chat_model


def test_missing_openai_key_raises_naming_the_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without OPENAI_API_KEY, the factory fails with a message naming the variable to set."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        get_chat_model("openai")


def test_missing_deepseek_key_raises_naming_the_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without DEEPSEEK_API_KEY, the factory fails with a message naming the variable to set."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        get_chat_model("deepseek")


def test_openai_key_present_returns_chat_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a key set, the factory builds a ChatOpenAI client."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert isinstance(get_chat_model("openai"), ChatOpenAI)


@pytest.mark.parametrize(
    ("provider", "client_class"),
    [("openai", ChatOpenAI), ("deepseek", ChatDeepSeek)],
)
def test_client_has_explicit_timeout_and_retries(
    monkeypatch: pytest.MonkeyPatch,
    provider: ProviderType,
    client_class: type[BaseChatOpenAI],
) -> None:
    """Both clients get the configured limits (D-038, D-039). Without them, the
    underlying OpenAI client has no timeout, and a hung call blocks forever."""
    monkeypatch.setenv(API_KEY_ENV_VARS[provider], "sk-test")
    model = get_chat_model(provider)
    assert isinstance(model, client_class)
    assert model.request_timeout == LLM_TIMEOUT_SECONDS
    assert model.max_retries == LLM_MAX_RETRIES
