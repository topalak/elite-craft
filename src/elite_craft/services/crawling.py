import logging
import re
from datetime import datetime

from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from pydantic import AnyUrl

from config import settings
from elite_craft.services.schemas import CrawledData


logger = logging.getLogger(__name__)

# Map documentation domains to source names
SOURCE_MAPPING = {
    "docs.langchain.com": "langchain",
    "python.langchain.com": "langchain",
    "reference.langchain.com": "langchain",
    "docling-project.github.io": "docling",
}


def _clean_markdown(raw_markdown: str) -> str:
    """
    Strip navigation boilerplate from crawled markdown.

    Removes:
        - Top: everything before the first h1 heading (nav, sidebar, breadcrumbs)
        - Bottom: everything from '* * *' separator onwards (footer, edit links, prev/next)

    Args:
        raw_markdown: Raw markdown from crawler

    Returns:
        Cleaned markdown containing only the page content
    """
    # Fix broken headers: merge "## \n[​](url#anchor)\nTitle" into "## Title"
    # Crawled docs split headers across 3 lines with a zero-width-space anchor link
    raw_markdown = re.sub(
        r'^(#{1,6}) \n\[(?:\u200b)?\]\([^)]+\)\n(.+)$',
        r'\1 \2',
        raw_markdown,
        flags=re.MULTILINE
    )

    # Strip top: find first h1 heading (# at start of line)
    h1_match = re.search(r'^# ', raw_markdown, re.MULTILINE)
    if h1_match:
        cleaned = raw_markdown[h1_match.start():]
    else:
        logger.warning("[CLEAN] No h1 heading found, keeping full content")
        cleaned = raw_markdown

    # Strip bottom: remove everything from '* * *' separator onwards
    separator_pos = cleaned.rfind("\n* * *\n")
    if separator_pos != -1:
        cleaned = cleaned[:separator_pos]

    return cleaned.strip()


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
    domain = parsed.host

    if domain in SOURCE_MAPPING:
        return SOURCE_MAPPING[domain]
    else:
        raise ValueError(
            f"Domain '{domain}' is not in SOURCE_MAPPING, "
            f"might be invalid domain"
        )


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

        raw_markdown = str(response.markdown)
        body_text = _clean_markdown(raw_markdown)

        logger.info(
            f"[CRAWL] Response received for: {url}, "
            f"raw length: {len(raw_markdown)} chars, "
            f"cleaned length: {len(body_text)} chars"
        )

    # Extract source name from URL
    try:
        source = _extract_source(url)
    except ValueError as e:
        logger.warning(f"[CRAWL] No source found for: {e}")
        source = "UNKNOWN"

    # Get current time in configured timezone as ISO format string
    crawled_time = datetime.now(tz=settings.TIME_ZONE).isoformat()

    return CrawledData(
        body_text=body_text,
        crawled_time=crawled_time,
        url=AnyUrl(url),
        source=source
    )