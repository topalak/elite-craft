import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter


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
        chunk_size: Maximum size of each chunk
    """

    def __init__(self, chunk_size: int, chunk_overlap: int):
        """Initialize chunker with recursive character text splitter."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            keep_separator="start",
            separators=[
                "\n## ",  # H2 - Major sections
                "\n### ",  # H3 - Subsections
                "\n#### ",  # H4 - Minor subsections
                "\n```",  # Code blocks
                "\n\n",  # Paragraph breaks
            ]
        )

    def _count_code_fences(self, text: str) -> int:
        """Count number of ``` code fence markers in text."""
        return text.count("```")

    def _has_incomplete_code_block(self, chunk: str) -> bool:
        """Check if chunk has incomplete code block (odd number of ```)."""
        return self._count_code_fences(chunk) % 2 != 0

    def _find_code_block_position(self, chunk: str) -> str:
        """
        Determine if incomplete code block is closer to start or end.

        Returns:
            'start' if closer to beginning, 'end' if closer to end
        """
        first_fence = chunk.find("```")
        last_fence = chunk.rfind("```")
        chunk_midpoint = len(chunk) // 2

        # If first fence is after midpoint, code is at end
        if first_fence > chunk_midpoint:
            return "end"
        # If last fence is before midpoint, code is at start
        elif last_fence < chunk_midpoint:
            return "start"
        # Default: treat as end to preserve code
        else:
            return "end"  #todo this is too rare scenario but lets set it as start to keep above information for code example

    def _fix_incomplete_code_blocks(self, chunks: list[str]) -> list[str]:
        """
        Fix chunks with incomplete code blocks.

        Strategy:
        - If incomplete code is at START: Remove it (will appear in previous chunk)
        - If incomplete code is at END: Extend chunk to include complete code from next chunk

        Args:
            chunks: List of text chunks

        Returns:
            List of fixed chunks with complete code blocks
        """
        if not chunks:
            return chunks

        fixed_chunks = []
        skip_next = False

        for i, chunk in enumerate(chunks):
            if skip_next:
                skip_next = False
                continue

            if not self._has_incomplete_code_block(chunk):
                fixed_chunks.append(chunk)
                continue

            position = self._find_code_block_position(chunk)

            if position == "start":
                # Remove incomplete code at start
                first_fence = chunk.find("```")
                # Find end of incomplete code block line
                newline_after_fence = chunk.find("\n", first_fence)
                if newline_after_fence != -1:  #todo check what is the value of newline_after_fence, why did we define -1
                    cleaned_chunk = chunk[newline_after_fence + 1:].lstrip()
                else:
                    cleaned_chunk = ""

                if cleaned_chunk.strip():
                    fixed_chunks.append(cleaned_chunk)
                else:
                    logger.debug(
                        f"Removed chunk entirely - contained only incomplete code block at start")

            else:  # position == "end"
                # Try to extend into next chunk to complete code block
                if i + 1 < len(chunks):
                    next_chunk = chunks[i + 1]
                    # Find closing fence in next chunk
                    closing_fence = next_chunk.find("```")

                    if closing_fence != -1:
                        # Find end of the closing fence line
                        newline_after_close = next_chunk.find("\n",
                                                              closing_fence + 3)
                        if newline_after_close != -1:
                            code_end = newline_after_close + 1
                        else:
                            code_end = len(next_chunk)

                        # Merge chunks to complete code block
                        extended_chunk = chunk + "\n" + next_chunk[:code_end]
                        fixed_chunks.append(extended_chunk)

                        # Update next chunk to remove merged portion
                        remaining = next_chunk[code_end:].lstrip()
                        if remaining.strip():
                            chunks[i + 1] = remaining
                        else:
                            skip_next = True

                        logger.info(
                            f"Extended chunk by {code_end} chars to complete code block "
                            f"(new size: {len(extended_chunk)} vs limit: {self.chunk_size})"
                        )
                    else:
                        # No closing fence found, keep as is and log warning
                        fixed_chunks.append(chunk)
                        logger.warning(
                            "Incomplete code block at end but no closing fence in next chunk")
                else:
                    # Last chunk with incomplete code - keep as is
                    fixed_chunks.append(chunk)
                    logger.warning(
                        "Last chunk has incomplete code block - keeping as is")

        return fixed_chunks

    def chunk(self, content: str, url: str) -> list[str]:
        """
        Split content into semantically coherent chunks.

        Args:
            content: Source content to chunk
            url: Source URL (for logging purposes)

        Returns:
            List of text chunks as strings with complete code blocks

        Raises:
            Exception: If text splitting fails
        """
        # Step 1: Initial chunking
        chunks = self.splitter.split_text(content)

        # Step 2: Fix incomplete code blocks
        fixed_chunks = self._fix_incomplete_code_blocks(chunks)

        logger.info(
            f"[CHUNK COMPLETE] Generated {len(fixed_chunks)} chunks for {url} "
            f"(original: {len(chunks)})"
        )
        return fixed_chunks
