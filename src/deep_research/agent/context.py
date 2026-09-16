from dataclasses import dataclass
from typing import Literal

ProviderType = Literal["openai","deepseek"]

@dataclass
class RunContext:
    provider: ProviderType