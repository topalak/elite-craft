from langchain.tools import tool
from pydantic import BaseModel, Field

from elite_craft.tools.retriever import Retriever
from elite_craft.tools.web_search import WebSearch


class WebSearchSchema(BaseModel):
    """Schema for web search tool input validation."""

    query: str = Field(description='Generate the most relevant search terms using keywords.')

class RetrieverSchema(BaseModel):
    """Schema for retriever tool input validation."""

    query: str = Field(description='Generate the most relevant search terms using keywords.')

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
        tavily_api_key: str,
    ):
        """
        Initialize handler with configuration.

        Args:
            supabase_url: Supabase project URL
            supabase_api_key: Supabase API key
            embedding_model: Name of embedding model to use
            tavily_api_key: Tavily API key for web search
        """
        self.retriever = Retriever(
            supabase_url=supabase_url,
            supabase_api_key=supabase_api_key,
            embedding_model_name=embedding_model
        )
        self.web_searcher = WebSearch(api_key=tavily_api_key)

    def get_retriever_tool(self):
        """
        Create retriever tool for agent use.

        Returns:
            LangChain tool that performs semantic search against
            documentation knowledge base
        """
        @tool("retriever_tool", args_schema=RetrieverSchema)
        def retriever_tool(query: str) -> list[dict]:
            """
            Retrieve relevant documentation chunks using semantic search.

            The tool searches a vector database
            of documentation and returns the most relevant chunks.

            Args:
                query: Search query describing what information you need

            Returns:
                List of relevant documentation chunks with metadata
            """
            response = self.retriever.retrieve_relevant_chunks(
                query=query,
            )
            return response

        return retriever_tool

    def get_web_search_tool(self):
        """
        Create web search tool for agent use.

        Returns:
            LangChain tool that performs real-time web search
            using Tavily API
        """
        @tool("web_search_tool", args_schema=WebSearchSchema)
        def web_search_tool(query: str) -> dict:
            """
            Search the web for current information using Tavily.

            The tool performs real-time web search and returns relevant
            results with snippets, URLs, and metadata.

            Args:
                query: Search query describing what information to find

            Returns:
                Dictionary containing web search results with URLs, snippets, and metadata
            """
            response = self.web_searcher.web_search(query)

            return response

        return web_search_tool


