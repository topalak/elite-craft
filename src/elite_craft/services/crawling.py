import logging
from datetime import datetime
from typing import Final
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from pydantic import AnyUrl

from config import settings
from elite_craft.services.schemas import CrawledData

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
    parsed = AnyUrl(url)
    #use model json here to extract the domain
    # todo pass url as AnyUrl and parse it inside of the method instead of using urlparser

    parsed = urlparse(url)
    domain = parsed.netloc

    if domain in SOURCE_MAPPING:
        return SOURCE_MAPPING[domain]
    else:
        raise ValueError(f"Domain '{domain}' is not in SOURCE_MAPPING, might be invalid domain")

async def crawl(url: str) -> CrawledData:
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
    browser_config = BrowserConfig()
    run_config = CrawlerRunConfig()

    async with AsyncWebCrawler(config=browser_config) as crawler:
        response = await crawler.arun(
            url=url,
            config=run_config
        )

        logger.info(f"[CRAWL] Response received for: {url}, content length: {len(response.markdown)} chars")

    # Extract source name from URL
    try:
        source = _extract_source(url)
    except ValueError as e:
        logger.warning(f"[CRAWL] No source found for: {e}")
        source = "UNKNOWN"

    # Get current time in configured timezone as ISO format string
    crawled_time = datetime.now(tz=settings.TIME_ZONE).isoformat()


    return CrawledData (
        body_text = response.markdown,
        crawled_time = crawled_time,
        url = AnyUrl(url),
        source =  source
    )


