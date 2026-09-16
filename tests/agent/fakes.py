"""Test doubles for agent tests."""

from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from deep_research.agent.context import ProviderType

DEFAULT_REPLY = "Attention lets a model weigh every token against every other token."


class RecordingFactory:
    """A ModelFactory for tests: returns a scripted fake model and records each provider it's asked for."""

    def __init__(self, reply: str = DEFAULT_REPLY) -> None:
        self.reply = reply
        self.providers: list[ProviderType] = []

    def __call__(self, provider: ProviderType) -> BaseChatModel:
        self.providers.append(provider)
        # A fresh iterator per call. GenericFakeChatModel consumes one item per
        # invoke, and an exhausted iterator would fail the second call.
        return GenericFakeChatModel(messages=iter([self.reply]))
