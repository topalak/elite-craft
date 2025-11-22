import logging

from langchain.tools import tool
from pydantic import BaseModel, Field
from supabase import create_client, Client

from elite_craft.model_provider import ModelConfig

logger = logging.getLogger(__name__)

class RetrieverToolInput(BaseModel):
    """
    That tool retrieves related chunks from the database.
    """

    query: str = Field(description="query that we calculate the similarity score")



class Retriever:
    def __init__(self, supabase_url:str, supabase_api_key:str, embedding_model_name:str, model_provider_url:str,use_ollama:bool=True): #todo update use_ollama
        embedding_model_config = ModelConfig(model=embedding_model_name, model_provider_url=model_provider_url, use_ollama_embedding=use_ollama)
        self.embedding_model = embedding_model_config.get_embedding()
        self.supabase_client: Client = create_client(supabase_url, supabase_api_key)

    #@tool(args_schema=RetrieverToolInput)
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
        """

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

def main():
    from src.config import settings
    retriever = Retriever(supabase_url=settings.SUPABASE_URL,
                          supabase_api_key=settings.SUPABASE_SERVICE_ROLE_SECRET_KEY,
                          embedding_model_name="embeddinggemma",
                          model_provider_url=settings.OLLAMA_HOST_LOCAL,
                          use_ollama=True)
    response = retriever.retrieve_relevant_chunks(query='Langchain overview')
    print(response)

if __name__ == "__main__":
    main()
