"""
Elite Craft CLI Development Interface.

Interactive command-line interface for testing the Crafter agent.
Allows users to ask questions and receive answers based on
retrieved documentation from the knowledge base.
"""
import logging
import os

from config import settings
from elite_craft.agent.crafter_agent import Crafter
from elite_craft.enums import Provider


# Configure root logger to control ALL loggers in the application
logging.basicConfig(
    level=settings.LOGGING_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# (Supabase, Anthropic API, etc. use httpx which logs all requests at INFO level)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


if __name__ == "__main__":

    os.environ['LANGSMITH_TRACING'] = getattr(settings, 'LANGSMITH_TRACING','true')
    os.environ['LANGSMITH_ENDPOINT'] = getattr(settings, 'LANGSMITH_ENDPOINT','https://api.smith.langchain.com')
    os.environ['LANGSMITH_API_KEY'] = settings.LANGSMITH_API_KEY.get_secret_value()
    os.environ['LANGSMITH_PROJECT'] = getattr(settings, 'LANGSMITH_PROJECT','elite-craft')

    crafter = Crafter(
        llm_model=settings.LLM_NAME,
        llm_api_key=settings.OLLAMA_API_KEY.get_secret_value(),
        llm_provider=Provider(settings.LLM_PROVIDER),  # Convert string to enum
        ollama_provider_url=settings.OLLAMA_HOST_COLAB.get_secret_value(),
        supabase_url=settings.SUPABASE_URL.get_secret_value(),
        supabase_api_key=settings.SUPABASE_SERVICE_ROLE_SECRET_KEY.get_secret_value(),
        embedding_model=settings.EMBEDDING_MODEL,
        tavily_api_key=settings.TAVILY_API_KEY.get_secret_value(),
    )

    while True:
        query = input(">You: ")
        if query.lower() == 'exit':
            print("Catch You Later")
            break

        # Invoke the agent
        crafter.ask(query=query, print_to_cli=True)

