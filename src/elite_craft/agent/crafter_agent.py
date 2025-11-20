from typing import Final

from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware

from elite_craft.model_provider import ModelConfig

LLM_CONFIG: Final = ModelConfig(model='gpt-oss:20b-cloud')
SYSTEM_PROMPT: Final = f""

class CrafterAgent:

    def __init__(self):
        self.llm = LLM_CONFIG.get_llm()

    def main(self):
        agent = create_agent(
            model=self.llm,
            tools=print('retriever'),
            system_prompt=SYSTEM_PROMPT,
            middleware=[TodoListMiddleware()],


        )