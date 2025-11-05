from wsgiref.simple_server import server_version

from pydantic import BaseModel, Field
from ..model_provider import ModelConfig
from config import settings

EMBEDDING_MODEL_CONFIG = ModelConfig(model='embeddinggemma', model_provider_url=settings.OLLAMA_HOST_MY_LOCAL)

#todo we can add more metadata like subheading (validation) and we can filter them in database if query and database corresponds in that subheading we calculate the similarity inside
# of that subset
class RetrieverTool(BaseModel):
    """
    That tool retrieves related chunks from the database.
    """

    source: str = Field(description="Exact framework name, you need to get that information from user's query"
                                    "example: How works LangGraph's streaming? --> source = langgraph") #todo we must convert any generated value into lower case for more robust output
    query: str = Field(description="query that we calculate the similarity score")



class Retriever:
    def __init__(self):
        self.embedding_model = EMBEDDING_MODEL_CONFIG.get_embedding()



    def retrieve_relevant_chunks(self, query: str, match_count: int = 10, section_filter: str = None,
                                 similarity_threshold: float = 0.6) -> list[dict]:
        """
        Retrieve relevant chunks from Supabase using semantic search

        Args:
            query: Search query
            match_count: Number of chunks to retrieve
            section_filter: Optional section filter
            similarity_threshold: Minimum similarity score to include chunk (0.0 to 1.0)

        Returns:
            List of dictionaries containing chunk content and metadata above the similarity threshold
            Each dict has: {'content': str, 'title': str, 'section': str, 'publication_date': str, 'similarity': float}
        """
        try:
            # Generate embedding for the query with EmbeddingGemma format
            formatted_query = f"task: search result | query: {query}"
            query_embedding = self.embedding_model.embed_query(formatted_query)
            print(f"🔍 [DEBUG] Generated query embedding with {len(query_embedding)} dimensions")

            # Call the match_article_chunks function
            params = {
                'query_embedding': query_embedding,
                'match_count': match_count
            }

            if section_filter:
                params['section_filter'] = section_filter

            result = self.supabase.rpc('match_article_chunks', params).execute()
