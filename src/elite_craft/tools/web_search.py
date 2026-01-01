from tavily import TavilyClient


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
            include_images=False,
            include_image_descriptions=True
        )

        return response

