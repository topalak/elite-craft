from langchain.tools import tool
from pydantic import BaseModel, Field

from elite_craft.tools import WebSearch, CodeExecutor, Retriever

class RetrieverSchema(BaseModel):
    """Schema for retriever tool input validation."""

    query: str = Field(description="Use keywords related to query")

class WebSearchSchema(BaseModel):
    """Schema for web search tool input validation."""

    query: str = Field(description='Generate the most relevant search terms using keywords. Keep query simple as possible.')

class CodeExecutorSchema(BaseModel):
    """Schema for code executor tool input validation."""

    code: str = Field(description="Python code to execute in sandbox")

class Handler:
    """
    Factory for creating LangChain tools with proper dependency injection.

    Handles initialization of stateful components (like API clients)
    and exposes them as stateless tool functions that agents can invoke.
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
        self.code_executor = CodeExecutor()

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
            Search the web for real-time information using Tavily API.

            THIS IS YOUR PRIMARY AND ONLY EXTERNAL KNOWLEDGE SOURCE.
            Use this tool to gather any information not in your training data.

            USE THIS TOOL FOR:
            - Latest Python syntax and code patterns for current Python versions
            - Agent frameworks: LangChain, LangGraph, DeepAgent documentation and examples
            - External API integrations (SendGrid, Slack, Stripe, Twilio, databases, etc.)
            - Real-time framework updates, breaking changes, latest releases
            - Current events, news, up-to-date information
            - Documentation for Python packages, libraries, or external services
            - Any syntax or implementation detail you're uncertain about

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

    def get_code_executor_tool(self):
        """
        Create code executor tool for agent use.

        Returns:
            LangChain tool that executes Python code in isolated
            Docker sandbox with security controls
        """
        @tool("code_executor_tool", args_schema=CodeExecutorSchema)
        def code_executor_tool(code: str) -> dict:
            """
            Execute Python code in a secure Docker sandbox environment.

            MANDATORY USE: You MUST call this tool every time you create code to verify it works.
            You will receive output regardless of success or failure.

            WHEN TO USE:
            - Every time you generate code - test it immediately
            - To validate code functionality before delivering to user
            - To verify syntax and runtime behavior
            - To check if dependencies are available and working

            SECURITY FEATURES:
            - Isolated Docker container (no access to host system)
            - Network disabled (no internet access)
            - Memory limited to 512MB
            - CPU limited to 50% of one core
            - Automatic timeout and cleanup

            PRE-INSTALLED PACKAGES:
            - langchain, langchain-core
            - pydantic
            - numpy, pandas
            - requests

            Args:
                code: Python code to execute (string)

            Returns:
                Dictionary with execution results:
                - stdout: Program output
                - stderr: Error messages
                - exit_code: 0 for success, non-zero for errors
                - timed_out: Whether execution exceeded timeout
                - execution_time: Actual execution duration
            """
            result = self.code_executor.execute(code)
            return result

        return code_executor_tool

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



