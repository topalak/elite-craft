import logging

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from pydantic import AnyUrl

from config import settings

logger = logging.getLogger(__name__)
logger.setLevel(level=settings.LOGGING_LEVEL)

class Chunker:
    """
    Service for chunking documents using Docling's hybrid chunking strategy.
    """

    def __init__(self):
        self.converter = DocumentConverter()
        self.chunker = HybridChunker()

    #@classmethod you don't need to create instance by using this decorator
    def chunk(self, content: str, url:str | AnyUrl) -> list[str]:  #todo I dont want to pass AnyUrl for every time I just defined it crawling process. When I am passing here url via crawled_data.url, the chunk method warns "expected str get AnyUrl. I don't want to define url as str or AnyUrl each time, or should I? I don't know which one is better?
        """
        Convert source content to document and chunk it.

        Args:
            url: URL to convert.
            content: The source content to chunk

        Returns:
            Chunks as list of strings
        """

        # Convert str document to Docling Document
        doc = self.converter.convert_string(
            content=content,
            format=InputFormat.MD,
            name=None
        ).document

        chunk_iter = self.chunker.chunk(dl_doc=doc)

        # Convert Docling Document chunks to string format
        chunks = [chunk.text for chunk in chunk_iter]

        logger.info(msg=f"[CHUNK COMPLETE] Generated chunks for {url}")
        return chunks
