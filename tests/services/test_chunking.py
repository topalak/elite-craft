"""
Unit test for Chunker.
"""

from unittest.mock import Mock, patch


class TestChunking:
    """
    Seperated tests.
    """

    def test_list_comprehension(self):
        """
        TEST 1: Basic list comprehension with multiple chunks.

        This tests line 43:
            chunks = [chunk.text for chunk in chunk_iter]

        Run coverage now and see what percentage we get!
        """
        # Arrange: Create fake chunk objects
        fake_chunk_1 = Mock()
        fake_chunk_1.text = "First chunk"

        fake_chunk_2 = Mock()
        fake_chunk_2.text = "Second chunk"

        fake_chunks = [fake_chunk_1, fake_chunk_2]

        # Mock Docling classes BEFORE Chunker creates them
        with patch('elite_craft.services.chunking.DocumentConverter') as MockConverter, \
             patch('elite_craft.services.chunking.HybridChunker') as MockHybridChunker:

            # Setup mock converter behavior
            mock_converter_instance = MockConverter.return_value
            fake_document = Mock()
            fake_convert_result = Mock()
            fake_convert_result.document = fake_document
            mock_converter_instance.convert_string.return_value = fake_convert_result

            # Setup mock chunker behavior
            mock_chunker_instance = MockHybridChunker.return_value
            mock_chunker_instance.chunk.return_value = fake_chunks

            # Create Chunker with mocked dependencies
            from elite_craft.services.chunking import Chunker
            chunker = Chunker()

            # Act: Call chunk method
            result = chunker.chunk(content="test content", url="https://example.com")

        # Assert
        assert result == ["First chunk", "Second chunk"]
        assert len(result) == 2