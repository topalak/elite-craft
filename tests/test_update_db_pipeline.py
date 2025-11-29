"""
Unit tests for Update DB Pipeline.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from elite_craft.services.update_db_pipeline import UpdateDBPipeline


class TestProcessSingleUrl:
    """Test the process_single_url method."""

    async def test_process_single_url_complete_pipeline(self):
        """Test that the full pipeline executes all steps correctly."""

        # STEP 1: Patch all service classes to avoid real operations
        with patch('elite_craft.services.update_db_pipeline.Chunker') as MockChunker, \
             patch('elite_craft.services.update_db_pipeline.Embedder') as MockEmbedder, \
             patch('elite_craft.services.update_db_pipeline.SupabaseUploadService') as MockUploader, \
             patch('elite_craft.services.update_db_pipeline.crawl') as mock_crawl:

            # STEP 2: Setup mock instances
            mock_chunker = MockChunker.return_value
            mock_embedder = MockEmbedder.return_value
            mock_uploader = MockUploader.return_value

            # STEP 3: Setup mock return values for each step

            # Mock crawl (Step 1) - must be async!
            mock_crawl.return_value = {
                "body_text": "Sample markdown content",
                "crawled_time": "2025-01-01T00:00:00",
                "url": "https://docs.langchain.com/guide",
                "source": "langchain"
            }

            # Mock insert_document (Step 2) - must be async!
            mock_uploader.insert_document = AsyncMock(return_value=123)

            # Mock chunker.chunk (Step 3)
            mock_chunker.chunk.return_value = ["chunk 1", "chunk 2", "chunk 3"]

            # Mock embedder.embed (Step 4)
            mock_embedder.embed.return_value = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

            # Mock insert_chunks (Step 5) - must be async!
            mock_uploader.insert_chunks = AsyncMock(return_value={
                "total_chunks": 3,
                "success": True
            })

            # STEP 4: Create pipeline
            pipeline = UpdateDBPipeline(
                embedding_model="fake-model",
                supabase_url="https://fake.supabase.co",
                supabase_key="fake-key"
            )

            # STEP 5: Call process_single_url
            result = await pipeline.process_single_url("https://docs.langchain.com/guide")

            # STEP 6: Assert result structure
            assert result["url"] == "https://docs.langchain.com/guide"
            assert result["source"] == "langchain"
            assert result["chunks_uploaded"] == 3
            assert result["success"] is True

            # STEP 7: Verify all steps were called
            mock_crawl.assert_called_once_with(url="https://docs.langchain.com/guide")
            mock_uploader.insert_document.assert_called_once()
            mock_chunker.chunk.assert_called_once()
            mock_embedder.embed.assert_called_once()
            mock_uploader.insert_chunks.assert_called_once()


class TestProcessMultipleUrls:
    """Test the process_multiple_urls method."""

    async def test_process_multiple_urls_success(self):
        """Test processing multiple URLs successfully."""

        with patch('elite_craft.services.update_db_pipeline.Chunker'), \
             patch('elite_craft.services.update_db_pipeline.Embedder'), \
             patch('elite_craft.services.update_db_pipeline.SupabaseUploadService'), \
             patch('elite_craft.services.update_db_pipeline.crawl') as mock_crawl:

            # Setup mocks for successful processing
            mock_crawl.return_value = {
                "body_text": "content",
                "crawled_time": "2025-01-01T00:00:00",
                "url": "https://example.com",
                "source": "langchain"
            }

            # Create pipeline and mock process_single_url
            pipeline = UpdateDBPipeline(
                embedding_model="fake-model",
                supabase_url="https://fake.supabase.co",
                supabase_key="fake-key"
            )

            # Mock process_single_url to return successful results
            async def mock_process(url):
                return {
                    "url": url,
                    "source": "langchain",
                    "chunks_uploaded": 5,
                    "success": True
                }

            pipeline.process_single_url = mock_process

            # Test with 3 URLs
            urls = [
                "https://docs.langchain.com/page1",
                "https://docs.langchain.com/page2",
                "https://docs.langchain.com/page3"
            ]

            results = await pipeline.process_multiple_urls(urls)

            # Assert all succeeded
            assert len(results) == 3
            assert all(r["success"] for r in results)
            assert results[0]["url"] == urls[0]
            assert results[1]["url"] == urls[1]
            assert results[2]["url"] == urls[2]

    async def test_process_multiple_urls_with_failures(self):
        """Test processing multiple URLs with some failures - covers error logging path."""

        with patch('elite_craft.services.update_db_pipeline.Chunker'), \
             patch('elite_craft.services.update_db_pipeline.Embedder'), \
             patch('elite_craft.services.update_db_pipeline.SupabaseUploadService'), \
             patch('elite_craft.services.update_db_pipeline.crawl'), \
             patch('elite_craft.services.update_db_pipeline.logger') as mock_logger:

            # Create pipeline
            pipeline = UpdateDBPipeline(
                embedding_model="fake-model",
                supabase_url="https://fake.supabase.co",
                supabase_key="fake-key"
            )

            # Mock process_single_url to raise exception on second call
            call_count = 0
            async def mock_process_with_errors(url):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    # Second URL raises exception
                    raise ValueError(f"Failed to process {url}")
                return {
                    "url": url,
                    "source": "langchain",
                    "chunks_uploaded": 5,
                    "success": True
                }

            # Replace process_single_url with error-prone version
            pipeline.process_single_url = mock_process_with_errors

            # Test with 3 URLs (use REAL process_multiple_urls method)
            urls = [
                "https://docs.langchain.com/page1",
                "https://docs.langchain.com/page2",  # Will fail
                "https://docs.langchain.com/page3"
            ]

            results = await pipeline.process_multiple_urls(urls)

            # Assert mixed results
            assert len(results) == 3
            assert isinstance(results[0], dict)
            assert isinstance(results[1], ValueError)  # Exception
            assert isinstance(results[2], dict)

            # Verify error logging was called
            assert mock_logger.error.called
            assert any("1 URLs failed" in str(call.args) for call in mock_logger.error.call_args_list)


class TestMainFunction:
    """Test the main() function."""

    async def test_main_function_executes(self):
        """Test that main() function runs the pipeline correctly."""

        with patch('elite_craft.services.update_db_pipeline.UpdateDBPipeline') as MockPipeline, \
             patch('elite_craft.services.update_db_pipeline.logging.basicConfig') as mock_logging, \
             patch('elite_craft.services.update_db_pipeline.logger') as mock_logger:

            # Setup mock pipeline instance
            mock_pipeline_instance = MockPipeline.return_value
            mock_pipeline_instance.process_multiple_urls = AsyncMock(return_value=[
                {"url": "url1", "success": True},
                {"url": "url2", "success": True}
            ])

            # Import and call main
            from elite_craft.services.update_db_pipeline import main
            results = await main()

            # Verify pipeline was created
            MockPipeline.assert_called_once()

            # Verify logging was configured
            mock_logging.assert_called_once()

            # Verify process_multiple_urls was called with URL list
            mock_pipeline_instance.process_multiple_urls.assert_called_once()
            call_args = mock_pipeline_instance.process_multiple_urls.call_args[0][0]
            assert len(call_args) == 3  # 12 URLs in the list
            assert all("langchain.com" in url for url in call_args)

            # Verify final logging
            assert mock_logger.info.called

            # Verify results returned
            assert len(results) == 2