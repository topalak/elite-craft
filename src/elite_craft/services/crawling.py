from datetime import datetime
from typing import TypedDict

import logging
from crawl4ai import AsyncWebCrawler
from urllib.parse import urlparse

from src.config import settings

logger = logging.getLogger(__name__)

class CrawledData(TypedDict):
    body_text: str
    crawled_time: str
    url: str
    source: str

class Crawler:
    """
    Service for asynchronously crawling web pages and extracting content.

    Uses Crawl4AI to fetch and parse web pages, converting HTML content into
    structured Markdown format. Designed to extract documentation from framework
    sites (LangChain, LangGraph, etc.) for the knowledge base pipeline.
    """

    # Map documentation domains to source names
    SOURCE_MAPPING = {
        "docs.langchain.com": "langchain",
        "python.langchain.com": "langchain",
        #"docling-project.github.io": "docling",
    }

    @staticmethod
    def _extract_source(url: str) -> str:
        """
        Extract source name from documentation URL.

        Args:
            url: Full documentation URL

        Returns:
            Source name (e.g., 'langchain', 'docling')

        Raises:
            ValueError: If domain is not in SOURCE_MAPPING
        """
        parsed = urlparse(url)
        domain = parsed.netloc

        if domain in Crawler.SOURCE_MAPPING:
            return Crawler.SOURCE_MAPPING[domain]
        else:
            raise ValueError(f"Domain '{domain}' is not in SOURCE_MAPPING")


    async def crawl(self, url: str) -> CrawledData:
        """
        Crawl a URL and return structured data for database insertion.

        Args:
            url: URL to crawl

        Returns:
            Dict with keys:
                - body_text (str): Markdown content
                - crawled_time (str): datetime ISO 8601 format
                - url (str): Source URL
                - source (str): Framework name
        Raises:
            ValueError: If domain is not in SOURCE_MAPPING
            Exception: If crawler fails to fetch URL
        """
        try:
            async with AsyncWebCrawler() as crawler:
                response = await crawler.arun(url=url)
        except ConnectionError as e:
            logger.error(f"Network error while crawling {url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected crawler error for {url}: {e}")
            raise

        # Extract source name from URL
        source = self._extract_source(url)

        # Get current time in configured timezone as ISO format string
        crawled_time = datetime.now(tz=settings.TIME_ZONE).isoformat()

        result: CrawledData = {
            "body_text": str(response.markdown),   # Convert to string to ensure proper Markdown formatting
            "crawled_time": crawled_time,
            "url": url,
            "source": source
        }

        return result

