from langchain.agents import create_agent
from elite_craft.model_provider import ModelConfig

llm_config = ModelConfig(model='gpt-oss:20b-cloud')




class CrafterAgent:

    def __init__(self):
        self.llm = llm_config.get_llm()

    def main(self):
        agent = create_agent(
            model=self.llm,
            tools=print('retriever'), #todo add retriever here
            system_prompt="prompt",
            middleware="todo",


        )