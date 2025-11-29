import logging

from elite_craft.agent.crafter_agent import Crafter
from config import settings

logger = logging.getLogger(__name__)
logger.setLevel(level=settings.LOGGING_LEVEL)

if __name__ == "__main__":

    crafter = Crafter(llm_model=settings.LLM_NAME,
                      llm_api_key=settings.OLLAMA_API_KEY,
                      embedding_model_name=settings.EMBEDDING_MODEL,
                      supabase_url=settings.SUPABASE_URL,
                      supabase_api_key=settings.SUPABASE_SERVICE_ROLE_SECRET_KEY)

    while True:
        query = input(">You: ")
        if query.lower() == 'exit':
            break
        # Get retrieved chunks
        crafter.ask(query)

