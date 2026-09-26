from dataclasses import dataclass
from typing import Literal

ProviderType = Literal["openai","deepseek"]

@dataclass
class RunContext:
    provider: ProviderType
    # The exact model to call, chosen or typed by the user (D-125, D-126). `None` means
    # "whatever MODEL_NAMES says for this provider", which is what every run did before this
    # existed -- so an omitted model keeps the old behaviour with no second code path.
    #
    # `None` rather than `""`: "the user did not choose" and "the user chose the empty string"
    # are different facts, and only one of them is possible. An empty string arriving here
    # would be a bug upstream, and a sentinel that cannot be produced by accident is worth
    # more than one that can.
    #
    # It lives here, beside `provider`, because the two are the same kind of thing: a per-run
    # choice that must reach the nodes through LangGraph's runtime context and never through a
    # module-level global (D-015). A global would make two concurrent runs share one model.
    model: str | None = None
