from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
import logging

logger = logging.getLogger(__name__)

class Chunker:
    """
    Service for chunking documents using Docling's hybrid chunking strategy.

    Initializes converter and chunker once to avoid latency on repeated calls.
    """

    def __init__(self):
        self.converter = DocumentConverter()
        self.chunker = HybridChunker()

    def chunk(self, content: str) -> list[str]:
        """
        Convert source content to document and chunk it.

        Args:
            content: The source content to chunk

        Returns:
            Chunks as list of strings

        Raises:
            Exception: If document conversion or chunking fails
        """

        try:
            doc = self.converter.convert_string(
                content=content,
                format=InputFormat.MD,
                name=None
            ).document
            chunk_iter = self.chunker.chunk(dl_doc=doc)

            # Convert Docling Document chunks to string format
            chunks = [chunk.text for chunk in chunk_iter]

            return chunks

        except Exception as e:
            logger.error(f"Chunking failed!: {e}")
            raise