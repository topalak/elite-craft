from langchain.tools import tool
from pydantic import BaseModel, Field

from elite_craft.tools.retriever import Retriever
from elite_craft.tools.web_search import WebSearch


class WebSearchSchema(BaseModel):
    """Schema for web search tool input validation."""

    query: str = Field(description='Generate the most relevant search terms using keywords.')

class RetrieverSchema(BaseModel):
    """Schema for retriever tool input validation."""

    query: str = Field(description="Use keywords related to query")

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
        @tool("retriever_tool", #args_schema=RetrieverSchema
        )
        def retriever_tool(query: str) -> list[dict]:
            """
            Retrieve documentation about LangChain, LangGraph, and Docling FRAMEWORKS from the knowledge base.

            This tool contains knowledge about AGENT FRAMEWORK CONCEPTS AND PATTERNS:
            - How to build agents, multi-agent systems, and orchestration
            - How to add tools/functions to agents
            - How to implement streaming, memory, state management
            - How to add human-in-the-loop, approval workflows
            - How to structure graphs, nodes, edges, routing
            - LangChain/LangGraph APIs and architecture patterns
            - Middleware, callbacks, checkpointing
            - RAG chains and retrieval patterns

            This tool does NOT contain:
            - External API integrations (SendGrid, Slack, Stripe, databases, etc.)
            - Real-time framework updates or breaking changes
            - General Python programming unrelated to agent frameworks

            For hybrid queries (e.g., "build an agent that sends emails"), use this tool to learn
            the AGENT PATTERNS (how to structure agents and add tools), then use web_search_tool
            for the DOMAIN KNOWLEDGE (email API integration).

            Args:
                query: This query will use to search for relevant documentation in database by embedding-based semantic search.

            Returns:
                List of documentation chunks with content, URLs, and similarity scores.
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
        @tool("web_search_tool", #args_schema=WebSearchSchema
         )
        def web_search_tool(query: str) -> dict:
            """
            Search the web for real-time information and external API documentation.

            USE THIS TOOL FOR:
            - External API integrations (SendGrid, Slack, Stripe, Twilio, databases, etc.)
            - Real-time framework updates, breaking changes, latest releases
            - Current events, news, up-to-date information
            - Documentation for libraries/services not in the knowledge base
            - When retriever_tool returns insufficient or irrelevant results

            This tool performs live web search and returns ranked results with content snippets,
            source URLs, and relevance metadata.

            Args:
                query: Web search query using relevant keywords for best results.

            Returns:
                Dictionary with search results including URLs, content snippets, and metadata.
            """
            response = self.web_searcher.web_search(query)

            return response

        return web_search_tool


