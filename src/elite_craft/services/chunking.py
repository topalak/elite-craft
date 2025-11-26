from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
import logging

logger = logging.getLogger(__name__)

class Chunker:
    """
    Service for chunking documents using Docling's hybrid chunking strategy.
    """

    def __init__(self):
        self.converter = DocumentConverter()
        self.chunker = HybridChunker()

    def chunk(self, content: str, url:str,) -> list[str]:
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

        logger.info(f"[CHUNK COMPLETE] Generated chunks for {url}")
        return chunks
