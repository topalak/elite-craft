from tavily import TavilyClient

domains = [
    "https://docs.langchain.com/oss/python/langchain/install",
    "https://docs.langchain.com/oss/python/langchain/quickstart",
    "https://docs.langchain.com/oss/python/releases/changelog",
    "https://docs.langchain.com/oss/python/langchain/philosophy",
    "https://docs.langchain.com/oss/python/langchain/agents",
    "https://docs.langchain.com/oss/python/langchain/models",
    "https://docs.langchain.com/oss/python/langchain/messages",
    "https://docs.langchain.com/oss/python/langchain/tools",
    "https://docs.langchain.com/oss/python/langchain/short-term-memory",
    "https://docs.langchain.com/oss/python/langchain/streaming/overview",
    "https://docs.langchain.com/oss/python/langchain/streaming/frontend",
    "https://docs.langchain.com/oss/python/langchain/structured-output",
    "https://docs.langchain.com/oss/python/langchain/middleware/overview",
    "https://docs.langchain.com/oss/python/langchain/middleware/built-in",
    "https://docs.langchain.com/oss/python/langchain/middleware/custom",
    "https://docs.langchain.com/oss/python/langchain/guardrails",
    "https://docs.langchain.com/oss/python/langchain/runtime",
    "https://docs.langchain.com/oss/python/langchain/context-engineering",
    "https://docs.langchain.com/oss/python/langchain/mcp",
    "https://docs.langchain.com/oss/python/langchain/human-in-the-loop",
    "https://docs.langchain.com/oss/python/langchain/multi-agent",
    "https://docs.langchain.com/oss/python/langchain/multi-agent/subagents",
    "https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs",
    "https://docs.langchain.com/oss/python/langchain/multi-agent/skills",
    "https://docs.langchain.com/oss/python/langchain/multi-agent/router",
    "https://docs.langchain.com/oss/python/langchain/multi-agent/custom-workflow",
    "https://docs.langchain.com/oss/python/langchain/retrieval",
    "https://docs.langchain.com/oss/python/langchain/long-term-memory"
    ]


class WebSearch:
    """
    Web search interface using Tavily API.

    Provides real-time web search capabilities with image support
    for retrieving current information from the internet.

    Attributes:
        client: TavilyClient instance for performing searches
    """

    def __init__(
            self,
            api_key: str,
    ):
        """
        Initialize web search client.

        Args:
            api_key: Tavily API key for authentication
        """
        self.client = TavilyClient(api_key=api_key)

    def web_search(self, query: str) -> list[dict]:
        """
        Perform web search with image support.

        Searches the web using Tavily API and returns relevant results
        including images and their descriptions.

        Args:
            query: Search query string describing information to find

        Returns:
            response: Dictionary of search results.
        """
        response = self.client.search(
            query=query,
            max_results=10,
            search_depth="advanced",
            chunks_per_source=5,

           # include_domains=domains,
        )
        return response

