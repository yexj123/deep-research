"""The user-typed model (D-125).

The model reaches the nodes through LangGraph's runtime `context`, beside `provider`, and for
the same reason: it is a *per-run* choice, so a module-level global would make two concurrent
runs share one model (D-015). These pin that path end to end, plus the two boundaries where a
bad name must fail before it costs anything.

Nothing here calls a real API. `get_chat_model` is exercised with fake keys, which is enough
to prove which model name it passes on -- whether that name exists is the provider's call, not
this project's.
"""

import pytest
from langgraph.runtime import Runtime

from deep_research.agent.config import MODEL_NAMES
from deep_research.agent.context import RunContext
from deep_research.agent.llm import get_chat_model
from deep_research.agent.nodes.intake import intake
from deep_research.agent.state import ResearchState


def _runtime(context: RunContext) -> Runtime:
    return Runtime(context=context)


# ---- the factory ----------------------------------------------------------------------


def test_an_absent_model_uses_the_providers_default(monkeypatch) -> None:
    """`None` means "whatever MODEL_NAMES says", which is what every run did before D-125.

    This is what makes the override backward compatible: an omitted model needs no second
    code path, and every recording made before it stays reproducible.

    `""` and `"   "` fall back too. They should never reach here -- the API boundary collapses
    them to None (D-126) -- but a blank slipping through is an upstream bug, and falling back
    is the same safe answer as omitting it rather than a second failure mode.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert get_chat_model("openai").model_name == MODEL_NAMES["openai"]
    assert get_chat_model("openai", None).model_name == MODEL_NAMES["openai"]
    assert get_chat_model("openai", "").model_name == MODEL_NAMES["openai"]
    assert get_chat_model("openai", "   ").model_name == MODEL_NAMES["openai"]


def test_every_recommended_model_is_a_usable_name() -> None:
    """The UI's own suggestions must pass the validation the UI submits them to (D-126).

    A recommended model that `intake` rejects would be a form offering an option that cannot
    be chosen -- the kind of contradiction nothing else would catch, since the list and the
    pattern live in the same file and are never compared.
    """
    import re

    from deep_research.agent.config import MODEL_NAME_PATTERN, RECOMMENDED_MODELS

    for provider, models in RECOMMENDED_MODELS.items():
        assert models, f"{provider} has no recommended models"
        for model in models:
            assert re.fullmatch(MODEL_NAME_PATTERN, model), f"{provider}/{model}"


def test_every_provider_default_is_offered_in_the_ui() -> None:
    """The configured default must appear in its own dropdown (D-126).

    Otherwise the form's first option silently differs from what an omitted model actually
    uses, and the two only diverge further as models are added.
    """
    from deep_research.agent.config import RECOMMENDED_MODELS

    for provider, default in MODEL_NAMES.items():
        assert default in RECOMMENDED_MODELS[provider], f"{provider} default {default!r} missing"


def test_a_typed_model_is_passed_through_verbatim(monkeypatch) -> None:
    """No allowlist, deliberately (D-125).

    Providers add and retire models constantly, so a local list would be wrong within weeks
    and would lock the user out of the model they actually want. A typo reaches the API and
    comes back as that provider's own 404, which names the model -- a better error than
    anything this project could produce.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert get_chat_model("openai", "gpt-4o-mini").model_name == "gpt-4o-mini"

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    assert get_chat_model("deepseek", "deepseek-reasoner").model_name == "deepseek-reasoner"


def test_a_missing_key_still_fails_before_the_model_is_considered(monkeypatch) -> None:
    """Key first: a typed model must not change which error a keyless run gets (D-029)."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="No API key"):
        get_chat_model("openai", "gpt-4o-mini")


# ---- the intake boundary --------------------------------------------------------------


@pytest.mark.parametrize("model", [None, ""])
def test_intake_accepts_a_run_with_no_model(model) -> None:
    """The default path must stay untouched by the new validation (D-126)."""
    state = ResearchState(question="  what is attention?  ")
    runtime = _runtime(RunContext(provider="openai", model=model))
    assert intake(state, runtime)["question"] == "what is attention?"


@pytest.mark.parametrize(
    "model",
    ["gpt-4o-mini", "deepseek-chat", "o3-mini", "meta/llama-3.1-70b", "qwen2.5:32b", "gpt-4.1"],
)
def test_intake_accepts_real_model_shapes(model: str) -> None:
    """The shapes real ids actually take, including `org/model` and `name:tag` forms.

    Rejecting any of these would be the allowlist problem wearing a regex.
    """
    state = ResearchState(question="what is attention?")
    intake(state, _runtime(RunContext(provider="openai", model=model)))


@pytest.mark.parametrize(
    "model",
    [
        "gpt 4o",  # whitespace
        "gpt-4o\nmodel",  # newline
        "../../etc/passwd\x00",  # control character
        "-leading-dash",  # must start alphanumeric
        "x" * 101,  # longer than any real id
    ],
)
def test_intake_rejects_a_malformed_model(model: str) -> None:
    """Checked in the FIRST node, because every later use of it costs money (D-125).

    A malformed name would otherwise surface as a provider error partway through `decompose`,
    after the run had started streaming and the user had been billed for the planner call.
    """
    state = ResearchState(question="what is attention?")
    with pytest.raises(ValueError, match="not a usable model name"):
        intake(state, _runtime(RunContext(provider="openai", model=model)))


def test_an_unknown_provider_still_fails_first() -> None:
    """Provider before model: the more fundamental error should be the one reported."""
    state = ResearchState(question="what is attention?")
    with pytest.raises(ValueError, match="unknown provider"):
        intake(state, _runtime(RunContext(provider="gemini", model="gpt 4o")))  # type: ignore[arg-type]
