from abc import ABC
from typing import Any
from ..core.judging_interfaces import LLMOutput
import dataclasses

class BaseJudge(ABC):
    """Abstract interface for all judges"""
    def evaluate(self, prompt: str, **kwargs: Any) -> LLMOutput:
        ...

@dataclasses.dataclass
class JudgeConfig:
    rubric: str
    include_cot: bool = True
    allow_tie: bool = True