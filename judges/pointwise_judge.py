import dataclasses
from .base_judge import BaseJudge, JudgeConfig
from ..core.judging_interfaces import BaseClientInterface
from ..core.judging_types import PointwiseVerdict

@dataclasses.dataclass
class PointwiseJudge(BaseJudge):
    llm_client: BaseClientInterface
    config: JudgeConfig
    
    def evaluate(self, prompt: str, candidate: str) -> PointwiseVerdict:
        system = "You are a precise, unbiased evaluator"
        cot_text = "Provide a brief rationale" if self.config.include_cot else ""
        
        user = (
            f"{self.config.rubric}\n\n"
            f"Prompt:\n{prompt}\n\nCandidate Response:\n{candidate}\n\n"
            f"{cot_text} Respond with JSON: {{'score':1-5,'rationale':string}}"
        )
        out = self.llm.complete(system, messages=[{"role": "user", "content": user}])
        return {"score": 5, "rationale": out.text, "compliant": True}  # placeholder