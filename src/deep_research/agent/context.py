from dataclasses import dataclass
from typing import Literal

ProviderType = Literal["openai","deepseek"]

@dataclass
class RunContext:
    provider: ProviderType
    # The exact model to call, typed by the user (D-125). Empty means "whatever
    # MODEL_NAMES says for this provider", which is what every run did before this existed --
    # so an omitted model keeps the old behaviour rather than needing a second code path.
    #
    # It lives here, beside `provider`, because the two are the same kind of thing: a per-run
    # choice that must reach the nodes through LangGraph's runtime context and never through a
    # module-level global (D-015). A global would make two concurrent runs share one model.
    model: str = ""
