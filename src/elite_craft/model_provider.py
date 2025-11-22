from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from ai_common.llm import _check_and_pull_ollama_model

from ollama import Client

class ModelConfig:
    def __init__(
        self,
        model:str,  #if we see ":", that means its required, required parameters - must come before optional ones
        num_ctx:int = None, #if you are using local ollama, client will set it 2048 as default, you can set by yourself but if you are using cloud it will take model's actual context limit
        model_provider_url:str = None, #when we see an equal sign that means is optional. We must set a default value
        reasoning:bool = False,
        temperature:int = 0,
        use_ollama_cloud:bool = False,     #Set to True to use Ollama Cloud instead of local
        api_key:str = None,
        use_groq:bool = None,
        use_ollama_embedding: bool = False,
        #num_predict:int = 128,   that causes tool call error, model cant generate tool call because of the limitation.
    ):

        self.model = model
        self.model_provider_url = model_provider_url
        self.num_ctx = num_ctx      #when __init__ invokes it creates dummies when it comes that line they are coming real variables
        self.reasoning = reasoning
        self.temperature = temperature
        self.use_ollama_cloud = use_ollama_cloud
        self.api_key = api_key
        self.use_groq = use_groq
        self.use_ollama_embedding = use_ollama_embedding
        #self.num_predict = num_predict


    def get_llm(self):
        """
        Loads the llm.
        """
        if self.use_ollama_cloud:
            # Use Ollama Cloud
            return ChatOllama(  #wrap the model
                model=self.model,
                base_url="https://ollama.com",
                client_kwargs={
                    'headers': {'Authorization': f'Bearer {self.api_key}'}
                },
                num_ctx=self.num_ctx,
                reasoning=self.reasoning,
                temperature=self.temperature,
                keep_alive="5m",
                #todo research for response_format
            )

        elif self.use_groq:
            # Use Groq Cloud
            return ChatGroq(
                model=self.model,
                api_key=self.api_key,
                temperature=self.temperature,
            )

        else:
            # Use local Ollama
            _check_and_pull_ollama_model(model_name=self.model, ollama_url=self.model_provider_url)
            ollama_client = Client(host=self.model_provider_url)
            ollama_client.generate(model=self.model)

            return ChatOllama(
                model=self.model,
                base_url=self.model_provider_url,
                num_ctx=self.num_ctx,
                reasoning=self.reasoning,
                temperature=self.temperature,
                keep_alive="5m",
                #num_predict=self.num_predict,
            )

    def get_embedding(self):

        if self.use_ollama_embedding:

            # Load embedding model if needed
            _check_and_pull_ollama_model(model_name=self.model, ollama_url=self.model_provider_url)
            ollama_client = Client(host=self.model_provider_url)
            ollama_client.embed(model=self.model)

            return OllamaEmbeddings(
                model=self.model,
                base_url=self.model_provider_url,
            )
        else:
            print('huggingface')


def main():
    print('main')

if __name__ == '__main__':
    main()
