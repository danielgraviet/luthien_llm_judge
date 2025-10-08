from ..core.judging_interfaces import BaseClientInterface
from ..core.judging_types import LLMOutput
from openai import OpenAI, types
import os
from dotenv import load_dotenv
load_dotenv()

class MartianClient(BaseClientInterface):
    def __init__(self):
        base_url = os.environ.get("MARTIAN_BASE_URL")
        self.client = OpenAI(base_url=base_url)
        
    async def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0
    ) -> LLMOutput:
        
        messages = [
            types.chat.ChatCompletionSystemMessageParam(role="system", content="You are a skilled {language} programmer."), # TODO, update for judges
            types.chat.ChatCompletionUserMessageParam(role="user", content=final_prompt),
        ]
        
        response = await self.client.chat.completions.create(
            model="openai/gpt-4.1-nano",
            messages=messages,
            temperature=temperature
        )
        return response