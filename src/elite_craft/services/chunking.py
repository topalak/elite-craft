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
        chunk_size: Maximum size of each chunk
    """

    def __init__(self, chunk_size: int, chunk_overlap: int):
        """Initialize chunker with recursive character text splitter."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            keep_separator=True,
            separators=[
                "\n# ",
                "\n## ",
                "\n### ",
                "Copy\n```",
                "\n",
                "\n\n",
                "\n```",
            ]
        )

    @staticmethod
    def _has_incomplete_code_block(chunk: str) -> bool:
        """Check if chunk has incomplete code blocks (openings != closings)."""
        opening_number = chunk.count("Copy\n```")
        closing_all = chunk.count("```")
        closing_number = closing_all - opening_number

        return opening_number != closing_number

    @staticmethod
    def _find_code_opening_position(chunk: str) -> str:
        """
        Find if 'Copy```' (code opening) is at start or end of chunk.

        Returns:
            'start' if closer to beginning
            'end' if closer to end
            'none' if no opening found
        """
        first_opening = chunk.find("Copy\n```")
        if first_opening == -1:
            return "none"

        chunk_midpoint = len(chunk) // 2
        return "start" if first_opening < chunk_midpoint else "end"

    def _fix_incomplete_code_blocks(self, chunks: list[str]) -> list[str]:
        """
        Fix chunks with incomplete code blocks.

        Strategy:
        - Code blocks start with 'Copy\n```' and end with '```' (without Copy)
        - If chunk has complete blocks (openings == closings): keep as is
        - If 'Copy```' at START: extend forward until closing ``` (no size limit for code)
        - If 'Copy```' at END: remove and prepend to next chunk
        - If remaining part is too small: prepend to next chunk and reprocess

        Args:
            chunks: List of text chunks

        Returns:
            List of fixed chunks with complete code blocks
        """
        if not chunks:
            return chunks

        minimum_chunk_size = settings.MINIMUM_CHUNK_SIZE
        fixed_chunks = []
        i = 0

        while i < len(chunks):
            chunk = chunks[i]

            # If chunk has complete code blocks, keep it
            if not self._has_incomplete_code_block(chunk):
                fixed_chunks.append(chunk)
                i += 1
                continue

            position = self._find_code_opening_position(chunk)

            if position == "none":
                # No code opening found - might have only closing ``` from previous chunk
                # Should already be handled, but keep chunk as is
                fixed_chunks.append(chunk)
                logger.debug("Chunk has incomplete code but no 'Copy```' opening found")
                i += 1

            elif position == "start":
                # Code block starts here but incomplete - extend until closing ``` (no size limit)
                extended_chunk = chunk
                j = i + 1
                chunks_j_modified = False  # Track if chunks[j] was modified with remaining

                while j < len(chunks):
                    next_chunk = chunks[j]
                    # Look for closing ``` (not part of Copy\n```)
                    closing_pos = next_chunk.find("```")

                    # Make sure it's not part of another Copy```
                    if closing_pos != -1:
                        # Check if it's preceded by Copy\n
                        if closing_pos >= 5 and next_chunk[closing_pos - 5:closing_pos] == "Copy\n":
                            j += 1
                            continue

                        # Found actual closing
                        newline_after_close = next_chunk.find("\n", closing_pos + 3)
                        if newline_after_close != -1:
                            code_end = newline_after_close + 1
                        else:
                            code_end = len(next_chunk)

                        extended_chunk = extended_chunk + "\n" + next_chunk[:code_end]

                        # Handle remaining part
                        remaining = next_chunk[code_end:].lstrip()
                        if remaining.strip():
                            if len(remaining) < minimum_chunk_size:
                                # Too small - prepend to next chunk if exists
                                if j + 1 < len(chunks):
                                    chunks[j + 1] = remaining + "\n" + chunks[j + 1]
                                    logger.info(
                                        f"Prepended small remaining ({len(remaining)} chars) to next chunk")
                                else:
                                    # Last chunk - append to current
                                    extended_chunk = extended_chunk + "\n" + remaining
                                    logger.info(
                                        f"Appended small remaining ({len(remaining)} chars) to completed chunk")
                            else:
                                # Large enough - update chunk for next iteration
                                chunks[j] = remaining
                                chunks_j_modified = True

                        # Found closing - append and exit
                        fixed_chunks.append(extended_chunk)
                        logger.info(
                            f"Extended chunk to complete code block (size: {len(extended_chunk)}, no size limit for code)")
                        i = j if chunks_j_modified else j + 1
                        break
                    else:
                        # No closing found in this chunk - add it and continue searching
                        extended_chunk = extended_chunk + "\n" + next_chunk
                        j += 1
                else:
                    # Loop completed without break - never found closing
                    fixed_chunks.append(chunk)
                    logger.warning("Code opening at start but no closing found in remaining chunks")
                    i += 1

            else:  # position == "end"
                # Code opening at end - remove and prepend to next chunk
                opening_pos = chunk.find("Copy\n```")
                chunk_before_code = chunk[:opening_pos].rstrip()
                code_part = chunk[opening_pos:]

                if chunk_before_code.strip():
                    fixed_chunks.append(chunk_before_code)

                if i + 1 < len(chunks):
                    # Prepend code opening to next chunk
                    chunks[i + 1] = code_part + "\n" + chunks[i + 1]
                    logger.info(
                        f"Moved code opening ({len(code_part)} chars) from end to next chunk")
                else:
                    # Last chunk - keep the incomplete code opening with warning
                    if code_part.strip():
                        fixed_chunks.append(code_part)
                    logger.warning("Last chunk has incomplete code opening at end")

                i += 1

        return fixed_chunks

    @staticmethod
    def _add_tiny_chunks_into_bigger_one(chunks: list[str], min_size: int = settings.MINIMUM_CHUNK_SIZE) -> list[str]:
        """
        Merge chunks smaller than min_size with their smaller neighbor.

        Strategy:
        - First chunk (tiny): merge with next
        - Last chunk (tiny): merge with previous
        - Middle chunk (tiny): merge with whichever neighbor is smaller

        Args:
            chunks: List of text chunks to process
            min_size: Minimum chunk size threshold

        Returns:
            List of chunks with tiny chunks merged into their smaller neighbors

        Raises:
            ValueError: If chunks list is empty
        """
        if not chunks:
            raise ValueError("Cannot process empty chunks list")

        if len(chunks) == 1:
            return chunks  # Single chunk, return as-is

        new_chunks = []
        skip_next = False

        for i, chunk in enumerate(chunks):
            # Skip if previous iteration merged this chunk
            if skip_next:
                skip_next = False
                continue

            # Chunk is large enough, keep as-is
            if len(chunk) >= min_size:
                new_chunks.append(chunk)
                continue

            # Tiny chunk - merge with smaller neighbor
            is_first = (i == 0)
            is_last = (i == len(chunks) - 1)

            if is_last:
                # Last chunk is tiny - merge with previous
                if new_chunks:
                    new_chunks[-1] = new_chunks[-1] + "\n" + chunk
                    logger.debug(f"Merged tiny last chunk ({len(chunk)} chars) with previous")
                else:
                    # Edge case: only chunk in list, keep it
                    new_chunks.append(chunk)

            elif is_first:
                # First chunk is tiny - merge with next
                next_chunk = chunks[i + 1]
                extended_chunk = chunk + "\n" + next_chunk
                new_chunks.append(extended_chunk)
                skip_next = True
                logger.debug(f"Merged tiny first chunk ({len(chunk)} chars) with next")

            else:
                # Middle chunk is tiny - merge with SMALLER neighbor
                next_chunk = chunks[i + 1]
                prev_chunk = new_chunks[-1] if new_chunks else ""

                if len(next_chunk) < len(prev_chunk):
                    # Next is smaller, merge forward
                    extended_chunk = chunk + "\n" + next_chunk
                    new_chunks.append(extended_chunk)
                    skip_next = True
                    logger.debug(
                        f"Merged tiny chunk ({len(chunk)} chars) with smaller next chunk ({len(next_chunk)} chars)"
                    )
                else:
                    # Previous is smaller or equal, merge backward
                    if new_chunks:
                        new_chunks[-1] = new_chunks[-1] + "\n" + chunk
                        logger.debug(
                            f"Merged tiny chunk ({len(chunk)} chars) with smaller previous chunk ({len(prev_chunk)} chars)"
                        )
                    else:
                        new_chunks.append(chunk)

        return new_chunks

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
        logger.info(f"Chunk content length in chars: {len(content)}")

        # Step 2: Fix incomplete code blocks
        fixed_chunks = self._fix_incomplete_code_blocks(chunks)

        extended_chunks = self._add_tiny_chunks_into_bigger_one(fixed_chunks)

        return extended_chunks