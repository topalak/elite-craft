from datetime import datetime

from pydantic import BaseModel, AnyUrl

class PipelineResults(BaseModel):
    """
    External contract: Result of the process_single_url
    """

    url: AnyUrl
    source: str
    chunks_uploaded: int
    success: bool

class CrawledData(BaseModel):

    body_text: str
    crawled_time: datetime | str #control after updating "document" table to understand is datetime type valuable
    url: AnyUrl
    source: str

