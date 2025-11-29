from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_groq import ChatGroq
from ai_common.llm import _check_and_pull_ollama_model

from ollama import Client

class ModelConfig:
    def __init__(
        self,
        model:str,
        num_ctx:int = None,
        model_provider_url:str = None,
        reasoning:bool = False,
        temperature:int = 0,
        use_ollama_local:bool = False,
        api_key:str = None,
        use_groq:bool = None,
    ):

        self.model = model
        self.model_provider_url = model_provider_url
        self.num_ctx = num_ctx
        self.reasoning = reasoning
        self.temperature = temperature
        self.use_ollama_local = use_ollama_local
        self.api_key = api_key
        self.use_groq = use_groq


    def get_llm(self):
        """
        Loads the llm.
        """

        if self.use_ollama_local:
            # Use local Ollama
            _check_and_pull_ollama_model(model_name=self.model, ollama_url=self.model_provider_url)
            ollama_client = Client(host=self.model_provider_url)
            ollama_client.generate(model=self.model)

            return ChatOllama( #wrap the model
                model=self.model,
                base_url=self.model_provider_url,
                num_ctx=self.num_ctx,
                reasoning=self.reasoning,
                temperature=self.temperature,
                keep_alive="5m",
            )

        elif self.use_groq:
            # Use Groq Cloud
            return ChatGroq(
                model=self.model,
                api_key=self.api_key,
                temperature=self.temperature,
            )
        else:
            # Use Ollama Cloud
            return ChatOllama(
                model=self.model,
                base_url="https://ollama.com",
                client_kwargs={
                    'headers': {'Authorization': f'Bearer {self.api_key}'}
                },
                num_ctx=self.num_ctx,
                reasoning=self.reasoning,
                temperature=self.temperature,
                keep_alive="5m",
            )

    def get_embedding(self):
        """
        Loads the embedding model from local Ollama (defaults to localhost:11434).
        """
        _check_and_pull_ollama_model(model_name=self.model, ollama_url=self.model_provider_url)
        ollama_client = Client(host=self.model_provider_url)
        ollama_client.embed(model=self.model)

        return OllamaEmbeddings(
            model=self.model,
            base_url=self.model_provider_url,
        )



def main():
    print('main')

if __name__ == '__main__':  # pragma: no cover
    main()
