import dataclasses
from typing import Optional, Any, TypedDict

@dataclasses.dataclass(frozen=True)
class LLMOutput:
    text: str
    raw_response: Optional[Any] = None
    latency_ms: Optional[float] = None
    total_tokens: Optional[int] = None

@dataclasses.dataclass(frozen=True)  
class LLMClientConfig:
    temperature: float = 0.0
    base_url: str

class PointwiseVerdict(TypedDict):
    score: int
    rationale: str
    compliant: bool 
    
