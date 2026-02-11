from typing import Literal

from ai_common.llm import _check_and_pull_ollama_model
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI
from ollama import Client

from elite_craft.enums import Provider


class ModelConfig:
    """
    Configuration for LLM and embedding model providers.

    Supports multiple providers:
    - Local Ollama (provider="ollama_local", requires model_provider_url)
    - Ollama Cloud (provider="ollama_cloud", requires api_key)
    - Groq Cloud (provider="groq", requires api_key)

    Args:
        model: Model name (e.g., 'llama2', 'mixtral-8x7b-32768')
        provider: Which provider to use (must be explicitly provided)
        api_key: API key for cloud providers (required for groq/ollama_cloud)
        num_ctx: Context window size for Ollama models
        model_provider_url: Custom provider URL (for ollama_local)
        reasoning: Enable reasoning mode for supported models
        temperature: Sampling temperature (0 = deterministic)
    """

    def __init__(
        self,
        model: str,
        provider: Provider | Literal["ollama_local", "ollama_cloud", "groq"],
        api_key: str | None = None,
        num_ctx: int | None = None,
        model_provider_url: str | None = None,
        reasoning: bool = False,
        temperature: float = 0,
    ):
        self.model = model
        self.provider = Provider(provider)  # Ensures valid provider
        self.api_key = api_key
        self.num_ctx = num_ctx
        self.model_provider_url = model_provider_url
        self.reasoning = reasoning
        self.temperature = temperature

        # Validate required parameters for each provider
        self._validate_config()

    def _validate_config(self):
        """Fail fast: Validate configuration on initialization."""
        if self.provider == Provider.GROQ and not self.api_key:
            raise ValueError("api_key is required when using Groq provider")

        if self.provider == Provider.OLLAMA_CLOUD and not self.api_key:
            raise ValueError("api_key is required when using Ollama Cloud")

        if self.provider == Provider.OPENAI and not self.api_key:
            raise ValueError("api_key is required when using OpenAI")


    def get_llm(self):
        """
        Load and return the configured LLM instance.

        Returns:
            ChatOllama or ChatGroq instance based on configuration

        Raises:
            ValueError: If provider configuration is invalid
        """
        match self.provider:
            case Provider.OLLAMA_LOCAL:
                # Use local Ollama
                _check_and_pull_ollama_model(
                    model_name=self.model,
                    ollama_url=self.model_provider_url
                )
                ollama_client = Client(host=self.model_provider_url)
                ollama_client.generate(model=self.model)

                # Wrap the model in LangChain interface
                return ChatOllama(
                    model=self.model,
                    base_url=self.model_provider_url,
                    num_ctx=self.num_ctx,
                    reasoning=self.reasoning,
                    temperature=self.temperature,
                    keep_alive="5m",
                )

            case Provider.GROQ:
                # Use Groq Cloud
                return ChatGroq(
                    model=self.model,
                    api_key=self.api_key,
                    temperature=self.temperature,
                )

            case Provider.OLLAMA_CLOUD:
                # Use Ollama Cloud
                return ChatOllama(
                    model=self.model,
                    base_url="https://ollama.com",
                    client_kwargs={
                        'headers': {'Authorization': f'Bearer {self.api_key}'}
                    },
                    temperature=self.temperature,
                )

            case Provider.OPENAI:
                return ChatOpenAI(
                    model=self.model,
                    temperature=self.temperature,
                )

            case _:
                # This shouldn't happen due to enum validation, but be defensive
                raise ValueError(f"Unknown provider: {self.provider}")

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