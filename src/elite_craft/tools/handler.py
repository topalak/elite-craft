from langchain.tools import tool

from elite_craft.tools.retriever import Retriever


class Handler:
    """
    Factory for creating LangChain tools with proper dependency injection.

    Handles initialization of stateful components (like database clients
    and embedding models) and exposes them as stateless tool functions
    that agents can invoke.
    """

    def __init__(
        self,
        supabase_url: str,
        supabase_api_key: str,
        embedding_model: str,
    ):
        """
        Initialize handler with configuration.

        Args:
            supabase_url: Supabase project URL
            supabase_api_key: Supabase API key
            embedding_model: Name of embedding model to use
        """
        self.retriever = Retriever(
            supabase_url=supabase_url,
            supabase_api_key=supabase_api_key,
            embedding_model_name=embedding_model
        )

    def get_retriever_tool(self):
        """
        Create retriever tool for agent use.

        Returns:
            LangChain tool that performs semantic search against
            documentation knowledge base
        """
        @tool("retriever_tool")
        def retriever_tool(query: str, source_filter: str = None) -> list[dict]:
            """
            Retrieve relevant documentation chunks using semantic search.

            The tool searches a vector database
            of documentation and returns the most relevant chunks.

            Args:
                query: Search query describing what information you need
                source_filter: Optional filter by source name (e.g., 'langchain', 'langgraph')

            Returns:
                List of relevant documentation chunks with metadata
            """
            response = self.retriever.retrieve_relevant_chunks(
                query=query,
                source_filter=source_filter
            )
            # TODO: convert list to single str
            return response

        return retriever_tool
