from datetime import datetime

from pydantic import AnyUrl, BaseModel


class PipelineResults(BaseModel):
    """
    Result of the process_single_url pipeline operation.

    Attributes:
        url: The URL that was processed
        source: The source identifier for the pipeline
        chunks_uploaded: Number of chunks successfully uploaded to the database
        success: Whether the pipeline operation completed successfully
    """

    url: str
    source: str
    chunks_uploaded: int
    success: bool


class CrawledData(BaseModel):
    """
    Data structure for crawled web content.

    Attributes:
        url: The URL of the crawled page
        body_text: Extracted text content from the page
        crawled_time: Timestamp of when the page was crawled
        source: The source identifier for the crawl operation
    """

    url: AnyUrl
    body_text: str
    crawled_time: datetime | str
    source: str
