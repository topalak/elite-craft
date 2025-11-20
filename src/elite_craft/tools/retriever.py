import logging

from langchain.tools import tool
from pydantic import BaseModel, Field
from supabase import create_client, Client

from elite_craft.model_provider import ModelConfig
from src.config import settings

logger = logging.getLogger(__name__)
EMBEDDING_MODEL_CONFIG = ModelConfig(model='embeddinggemma', model_provider_url=settings.OLLAMA_HOST_LOCAL)

class RetrieverToolInput(BaseModel):
    """
    That tool retrieves related chunks from the database.
    """

    query: str = Field(description="query that we calculate the similarity score")



class Retriever:
    def __init__(self):
        self.embedding_model = EMBEDDING_MODEL_CONFIG.get_embedding()
        self.SUPABASE_URL = settings.SUPABASE_URL
        self.SUPABASE_KEY = settings.SUPABASE_SERVICE_ROLE_SECRET_KEY
        self.supabase_client: Client = create_client(self.SUPABASE_URL, self.SUPABASE_KEY)

    @tool(args_schema=RetrieverToolInput)
    def retrieve_relevant_chunks(self, query: str, match_count: int = 15, source_filter: str = None) -> list[dict]:
        """
        Retrieve relevant chunks from Supabase using semantic search.

        Args:
            query: Search query string
            match_count: Maximum number of chunks to return (default: 15)
            source_filter: Optional filter by source name (e.g., 'langchain', 'docling')

        Returns:
            List of dictionaries containing chunk content and metadata.
            Each dict contains:
                - chunk_id (int): Unique chunk identifier
                - url (str): Source document URL
                - chunk_number (int): Chunk position in document
                - content (str): Chunk text content
                - source (str): Source framework name
                - crawled_time (str): ISO timestamp of when document was crawled
                - similarity (float): Cosine similarity score (0-1)

        Raises:
            Exception: If database query fails
        """

        try:
            # Generate embedding for the query with EmbeddingGemma format
            formatted_query = f"task: search result | query: {query}"
            query_embedding = self.embedding_model.embed_query(formatted_query)

            # Prepare parameters for match_chunks function
            params = {
                'query_embedding': query_embedding,
                'match_count': match_count,
                'source_filter': source_filter
            }

            result = self.supabase_client.rpc('match_chunks', params).execute()

            if result.data:
                return result.data
            else:
                logger.info("No chunks found for query")
                return []
            
        except Exception as e:
            logger.error(f"Failed to retrieve chunks from database: {e}")
            raise Exception(f"Database query failed: {e}")
