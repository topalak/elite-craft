"""
Elite Craft CLI Development Interface.

Interactive command-line interface for testing the Crafter agent.
Allows users to ask questions and receive answers based on
retrieved documentation from the knowledge base.
"""
import logging

from config import settings
from elite_craft.agent.crafter_agent import Crafter


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

    crafter = Crafter(
        llm_model=settings.LLM_NAME,
        use_ollama_local=settings.USE_OLLAMA_LOCAL,
        ollama_provider_url=settings.OLLAMA_HOST_LOCAL,
        llm_api_key=settings.OLLAMA_API_KEY,
        embedding_model_name=settings.EMBEDDING_MODEL,
        supabase_url=settings.SUPABASE_URL,
        supabase_api_key=settings.SUPABASE_SERVICE_ROLE_SECRET_KEY
    )

    while True:
        query = input(">You: ")
        if query.lower() == 'exit':
            print("Catch You Later")
            break
        # Get retrieved chunks
        crafter.ask(query=query, print_to_cli=True)

