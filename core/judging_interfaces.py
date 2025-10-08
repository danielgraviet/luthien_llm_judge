from abc import ABC, abstractmethod
from .judging_types import LLMOutput

class BaseJudgeInterface(ABC):
    @abstractmethod
    def evaluate(
        self,
        system: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0
    ) -> LLMOutput:
        """Evaluate a candidate response given a prompt"""
        ...
        
class BaseClientInterface(ABC):
    @abstractmethod
    def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0
    ) -> LLMOutput:
        """Send a chat request to a LLM and capture structured result"""
        ...
