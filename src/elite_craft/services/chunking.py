import logging

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter

from config import settings


logger = logging.getLogger(__name__)
logger.setLevel(level=settings.LOGGING_LEVEL)


class Chunker:
    """
    Service for chunking documents using Docling's hybrid strategy.

    Converts markdown content into Docling documents and applies hybrid
    chunking to create semantically coherent text segments suitable for
    embedding and retrieval.

    Attributes:
        converter: DocumentConverter instance for markdown processing
        chunker: HybridChunker instance for intelligent text segmentation
    """

    def __init__(self,
        tokenizer_name:str,
        max_token_size_per_chunk: int
    ):
        """Initialize chunker with Docling converter and chunker."""
        self.tokenizer = HuggingFaceTokenizer.from_pretrained(
            model_name=tokenizer_name,
            max_tokens=max_token_size_per_chunk
        )
        self.converter = DocumentConverter()
        self.chunker = HybridChunker(tokenizer=self.tokenizer, merge_peers=True)

    def chunk(self, content: str, url: str) -> list[str]:
        """
        Convert markdown content to document and chunk it.

        Args:
            content: Markdown source content to chunk
            url: Source URL (for logging purposes)

        Returns:
            List of text chunks as strings

        Raises:
            Exception: If document conversion or chunking fails
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

