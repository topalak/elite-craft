import logging

from supabase import Client, create_client

from elite_craft.enums import Provider
from elite_craft.model_provider import ModelConfig


logger = logging.getLogger(__name__)


class Retriever:
    """
    Semantic search retriever for documentation chunks.

    Uses embedding models to perform similarity search against stored
    documentation in Supabase vector database.

    Attributes:
        embedding_model: Configured embedding model for query encoding
        supabase_client: Supabase client for database operations
    """

    def __init__(
        self,
        supabase_url: str,
        supabase_api_key: str,
        embedding_model_name: str
    ):
        embedding_model_config = ModelConfig(
            model=embedding_model_name,
            provider=Provider.OLLAMA_LOCAL
        )
        self.embedding_model = embedding_model_config.get_embedding()
        self.supabase_client: Client = create_client(
            supabase_url,
            supabase_api_key
        )


    def retrieve_relevant_chunks(
        self,
        query: str,
        source_filter: str = None
    ) -> list[dict]:
        """
        Retrieve relevant chunks from Supabase using semantic search.

        Args:
            query: Search query string
            source_filter: Optional filter by source name
                (e.g., 'langchain', 'docling')

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
        query_embedding = self.embedding_model.embed_query(query)

        params = {
            'query_embedding': query_embedding,
            'source_filter': source_filter
        }

        result = self.supabase_client.rpc('match_chunks', params).execute()

        return result.data if result.data else []