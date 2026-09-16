from dataclasses import dataclass

@dataclass
class ResearchState:
    question: str 
    review: str = ""
    