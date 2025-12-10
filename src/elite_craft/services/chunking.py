import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings


logger = logging.getLogger(__name__)


class Chunker:
    """
    Service for chunking documents using recursive character text splitting.

    Uses RecursiveCharacterTextSplitter with intelligent separator hierarchy
    to create semantically coherent chunks. Splits content based on headers,
    paragraphs, code definitions, and other structural elements while maintaining
    readability and context.

    Attributes:
        splitter: RecursiveCharacterTextSplitter instance for text segmentation
    """

    def __init__(self, chunk_size:int, chunk_overlap:int):
        """Initialize chunker with recursive character text splitter."""
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            keep_separator=True,
            separators=[
                "\n# ",  # H1 - Top-level sections
                "\n## ",  # H2 - Major sections
                "\n### ",  # H3 - Subsections
                "\n#### ",  # H4 - Minor subsections
                "\n\n",  # Paragraph breaks
                "\nclass ",  # Python class definitions
                "\ndef ",  # Python function definitions
                "\n",  # Line breaks
                ". ",  # Sentence boundaries
                " ",  # Word boundaries
                "",  # Character-level (last resort)
                "\n```"
            ]
        )

    def chunk(self, content: str, url: str) -> list[str]:
        """
        Split content into semantically coherent chunks.

        Args:
            content: Source content to chunk
            url: Source URL (for logging purposes)

        Returns:
            List of text chunks as strings

        Raises:
            Exception: If text splitting fails
        """
        chunks = self.splitter.split_text(content)
        logger.info(f"[CHUNK COMPLETE] Generated chunks for {url}")
        return chunks
