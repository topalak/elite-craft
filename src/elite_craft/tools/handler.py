from langchain.tools import tool
from pydantic import BaseModel, Field

from elite_craft.tools import WebSearch, CodeExecutor


class WebSearchSchema(BaseModel):
    """Schema for web search tool input validation."""

    query: str = Field(description='Generate the most relevant search terms using keywords.')

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
        tavily_api_key: str,
    ):
        """
        Initialize handler with configuration.

        Args:
            tavily_api_key: Tavily API key for web search
        """

        self.web_searcher = WebSearch(api_key=tavily_api_key)
        self.code_executor = CodeExecutor()

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

    def get_code_executor_tool(self):
        """
        Create code executor tool for agent use.

        Returns:
            LangChain tool that executes Python code in isolated
            Docker sandbox with security controls
        """
        @tool("code_executor_tool", #args_schema=CodeExecutorSchema
        )
        def code_executor_tool(code: str) -> dict:
            """
            Execute Python code in a secure Docker sandbox.

            USE THIS TOOL TO:
            - Test and validate generated code
            - Run example code snippets
            - Verify code functionality before delivering to user
            - Execute data processing or calculations safely

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


