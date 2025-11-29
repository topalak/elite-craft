"""
Unit test for Chunker.
"""

from unittest.mock import Mock, patch


class TestChunking:
    """
    Seperated tests.
    """

    def test_1_basic_list_comprehension(self):
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


    # def test_2_empty_chunk_list(self):
    #     """
    #     TEST 2: Empty chunk list - edge case.
    #
    #     This tests what happens when chunk_iter is empty.
    #
    #     After uncommenting, run coverage again!
    #     """
    #     # Arrange: Empty chunks
    #     fake_chunks = []

    #     with patch('elite_craft.services.chunking.DocumentConverter') as MockConverter, \
    #          patch('elite_craft.services.chunking.HybridChunker') as MockHybridChunker:

    #         mock_converter_instance = MockConverter.return_value
    #         mock_converter_instance.convert_string.return_value = Mock(document=Mock())

    #         mock_chunker_instance = MockHybridChunker.return_value
    #         mock_chunker_instance.chunk.return_value = fake_chunks

    #         from elite_craft.services.chunking import Chunker
    #         chunker = Chunker()

    #         # Act
    #         result = chunker.chunk(content="", url="https://example.com")

    #     # Assert
    #     assert result == []
    #     assert len(result) == 0


    # def test_3_single_chunk(self):
    #     """
    #     TEST 3: Single chunk - edge case.
    #
    #     Tests list comprehension with only one chunk.
    #
    #     After uncommenting, run coverage again!
    #     """
    #     # Arrange: Single chunk
    #     fake_chunk = Mock()
    #     fake_chunk.text = "Only one chunk"

    #     with patch('elite_craft.services.chunking.DocumentConverter') as MockConverter, \
    #          patch('elite_craft.services.chunking.HybridChunker') as MockHybridChunker:

    #         mock_converter_instance = MockConverter.return_value
    #         mock_converter_instance.convert_string.return_value = Mock(document=Mock())

    #         mock_chunker_instance = MockHybridChunker.return_value
    #         mock_chunker_instance.chunk.return_value = [fake_chunk]

    #         from elite_craft.services.chunking import Chunker
    #         chunker = Chunker()

    #         # Act
    #         result = chunker.chunk(content="short", url="https://example.com")

    #     # Assert
    #     assert result == ["Only one chunk"]
    #     assert len(result) == 1


    # def test_4_many_chunks(self):
    #     """
    #     TEST 4: Many chunks - stress test.
    #
    #     Tests list comprehension with many chunks.
    #
    #     After uncommenting, run coverage again!
    #     """
    #     # Arrange: 10 chunks
    #     fake_chunks = [Mock(text=f"Chunk {i}") for i in range(10)]

    #     with patch('elite_craft.services.chunking.DocumentConverter') as MockConverter, \
    #          patch('elite_craft.services.chunking.HybridChunker') as MockHybridChunker:

    #         mock_converter_instance = MockConverter.return_value
    #         mock_converter_instance.convert_string.return_value = Mock(document=Mock())

    #         mock_chunker_instance = MockHybridChunker.return_value
    #         mock_chunker_instance.chunk.return_value = fake_chunks

    #         from elite_craft.services.chunking import Chunker
    #         chunker = Chunker()

    #         # Act
    #         result = chunker.chunk(content="long content", url="https://example.com")

    #     # Assert
    #     assert len(result) == 10
    #     assert result[0] == "Chunk 0"
    #     assert result[9] == "Chunk 9"