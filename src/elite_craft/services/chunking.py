from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker


class Chunker:
    """
    Service for chunking documents using Docling's hybrid chunking strategy.

    Initializes converter and chunker once to avoid latency on repeated calls.
    """
    def __init__(self):
        self.converter = DocumentConverter()
        self.chunker = HybridChunker()

    def chunk(self, source: str):
        """
        Convert source content to document and chunk it.

        Args:
            source: The source content to chunk (URL, file path, or markdown text)

        Returns:
            List of document chunks

        Example:
            >>> chunker = Chunker()
            >>> chunks = chunker.chunk("https://example.com/doc")
        """
        doc = self.converter.convert(source=source).document
        chunk_iter = self.chunker.chunk(dl_doc=doc)

        return list(chunk_iter)