from crawl4ai import AsyncWebCrawler
from config import settings
from datetime import datetime
from urllib.parse import urlparse


class Crawler:
    """
    Service for asynchronously crawling web pages and extracting content.

    Uses Crawl4AI to fetch and parse web pages, converting HTML content into
    structured markdown format. Designed to extract documentation from framework
    sites (LangChain, LangGraph, etc.) for the knowledge base pipeline.
    """

    # Map documentation domains to source names
    SOURCE_MAPPING = {
        "ai.pydantic.dev": "pydantic",
        "docs.pydantic.dev": "pydantic",
        "docs.crawl4ai.com": "crawl4ai",
        "docs.langchain.com": "langchain",
        "python.langchain.com": "langchain",
        "js.langchain.com": "langchain",
        "docling-project.github.io": "docling",
        # Add more frameworks as needed
    }

    def __init__(self):
        """Initialize the crawler."""
        pass

    @staticmethod
    def _extract_source(url: str) -> str:
        """
        Extract source name from documentation URL.

        Args:
            url: Full documentation URL

        Returns:
            Source name (e.g., 'pydantic', 'langchain', 'docling')

        Raises:
            ValueError: If domain is not in SOURCE_MAPPING
        """
        parsed = urlparse(url)
        domain = parsed.netloc

        if domain in Crawler.SOURCE_MAPPING:
            return Crawler.SOURCE_MAPPING[domain]

        raise ValueError(
            f"Unknown documentation source: {domain}. "
            f"Add to Crawler.SOURCE_MAPPING in crawling.py"
        )

    async def crawl(self, url: str) -> dict:
        """
        Crawl a URL and return structured data for database insertion.

        Args:
            url: URL to crawl

        Returns:
            Dict with keys: body_text, crawled_time, url, source
        """
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)

        # Extract source name from URL
        source = self._extract_source(url)

        # Get current time in configured timezone
        crawled_time = datetime.now(tz=settings.TIME_ZONE)

        return {
            "body_text": result.markdown,
            "crawled_time": crawled_time,
            "url": url,
            "source": source
        }


