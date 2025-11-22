from datetime import datetime
import logging
from typing import Final

from crawl4ai import AsyncWebCrawler
from urllib.parse import urlparse

from src.config import settings

logger = logging.getLogger(__name__)


# Map documentation domains to source names
SOURCE_MAPPING: Final = {
    "docs.langchain.com": "langchain",
    "python.langchain.com": "langchain",
    #"docling-project.github.io": "docling",
}

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

    if domain in SOURCE_MAPPING:
        return SOURCE_MAPPING[domain]
    else:
        raise ValueError(f"Domain '{domain}' is not in SOURCE_MAPPING")


async def crawl(url: str) -> dict:
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
    """

    async with AsyncWebCrawler() as crawler:
        response = await crawler.arun(url=url)  #TODO error handling to prevent websites go down

    # Extract source name from URL
    source = _extract_source(url)

    # Get current time in configured timezone as ISO format string
    crawled_time = datetime.now(tz=settings.TIME_ZONE).isoformat()  #TODO make timezone adjustable (add it to signature)

    """
      async def crawl(url: str, timezone: datetime.timezone = None) -> dict:
      if timezone is None:
          from src.config import settings
          timezone = settings.TIME_ZONE

      crawled_time = datetime.now(tz=timezone).isoformat()
    """

    result = {
        "body_text": response.markdown,
        "crawled_time": crawled_time,
        "url": url,
        "source": source
    }

    return result

