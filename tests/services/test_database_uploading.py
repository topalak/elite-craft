"""
Unit tests for Database Uploading.
"""

import pytest
from unittest.mock import Mock, patch

from elite_craft.services.database_uploading import SupabaseUploadService
from elite_craft.services.schemas import CrawledData

class TestInsertDocument:
    """Test the insert_document method."""

    async def test_insert_document_creates_correct_db_record(self):
        """Test that document metadata is properly formatted and upserted."""

        # STEP 1: Patch create_client to avoid real Supabase connection
        with patch('elite_craft.services.database_uploading.create_client') as mock_create_client:

            # STEP 2: Setup mock client behavior
            mock_client = Mock()
            mock_create_client.return_value = mock_client

            # STEP 3: Setup mock response for upsert chain
            mock_response = Mock()
            mock_response.data = [{"id": 123}]

            # Mock the chain: .table().upsert().execute
            mock_table = Mock()
            mock_upsert = Mock()
            mock_upsert.execute.return_value = mock_response
            mock_table.upsert.return_value = mock_upsert
            mock_client.table.return_value = mock_table

            # STEP 4: Create service instance
            service = SupabaseUploadService(
                supabase_url="https://fake.supabase.co",
                supabase_key="fake-key"
            )

            # STEP 5: Prepare test data
            content: CrawledData = CrawledData(
                url="https://docs.langchain.com/guide",
                source="langchain",
                crawled_time="2025-01-01T00:00:00",
                body_text="A" * 5000  # Long text to test body_preview truncation
            )

            # STEP 6: Call insert_document
            result = await service.insert_document(content)

            # STEP 7: Assert result
            assert result == 123

            # STEP 8: Verify body_text is excluded and body_preview is created (our logic!)
            call_args = mock_table.upsert.call_args[0][0]
            assert "body_text" not in call_args
            assert "body_preview" in call_args
            assert len(call_args["body_preview"]) == 3000  # Truncated to BODY_PREVIEW_END


class TestInsertChunks:
    """Test the insert_chunks method."""

    async def test_insert_chunks_with_no_existing_chunks(self):
        """Test inserting chunks when no existing chunks found."""

        with patch('elite_craft.services.database_uploading.create_client') as mock_create_client:

            # Setup mock client
            mock_client = Mock()
            mock_create_client.return_value = mock_client

            # Mock existing chunks check (no existing chunks - BRANCH A)
            mock_select_response = Mock()
            mock_select_response.data = []  # Empty = no existing chunks

            mock_eq = Mock()
            mock_eq.execute.return_value = mock_select_response
            mock_select_chain = Mock()
            mock_select_chain.eq.return_value = mock_eq

            # Mock insert operation
            mock_insert = Mock()
            mock_insert.execute.return_value = Mock()

            # Setup table mock
            def table_side_effect(table_name):
                table_mock = Mock()
                table_mock.select.return_value = mock_select_chain
                table_mock.insert.return_value = mock_insert
                return table_mock

            mock_client.table.side_effect = table_side_effect

            # Create service
            service = SupabaseUploadService(
                supabase_url="https://fake.supabase.co",
                supabase_key="fake-key"
            )

            # Test data
            chunks = ["chunk 1", "chunk 2"]
            embeddings = [[0.1, 0.2], [0.3, 0.4]]

            # Act
            result = await service.insert_chunks(
                chunks=chunks,
                embeddings=embeddings,
                document_id=123,
                url="https://example.com"
            )

            # Assert
            assert result["total_chunks"] == 2
            assert result["success"] is True

    async def test_insert_chunks_deletes_existing_chunks(self):
        """Test that existing chunks are deleted before inserting new ones."""

        with patch('elite_craft.services.database_uploading.create_client') as mock_create_client:

            # Setup mock client
            mock_client = Mock()
            mock_create_client.return_value = mock_client

            # Mock existing chunks check (HAS existing chunks - BRANCH B)
            mock_select_response = Mock()
            mock_select_response.data = [{"id": 1}, {"id": 2}]  # Has 2 existing chunks

            mock_eq = Mock()
            mock_eq.execute.return_value = mock_select_response
            mock_select_chain = Mock()
            mock_select_chain.eq.return_value = mock_eq

            # Mock delete operation
            mock_delete = Mock()
            mock_delete_eq = Mock()
            mock_delete_eq.execute.return_value = Mock()
            mock_delete.eq.return_value = mock_delete_eq

            # Mock insert operation
            mock_insert = Mock()
            mock_insert.execute.return_value = Mock()

            # Setup table mock
            def table_side_effect(table_name):
                table_mock = Mock()
                table_mock.select.return_value = mock_select_chain
                table_mock.delete.return_value = mock_delete
                table_mock.insert.return_value = mock_insert
                return table_mock

            mock_client.table.side_effect = table_side_effect

            # Create service
            service = SupabaseUploadService(
                supabase_url="https://fake.supabase.co",
                supabase_key="fake-key"
            )

            # Test data
            chunks = ["new chunk"]
            embeddings = [[0.1, 0.2]]

            # Act
            result = await service.insert_chunks(
                chunks=chunks,
                embeddings=embeddings,
                document_id=123,
                url="https://example.com"
            )

            # Assert: Delete was called (testing the if branch!)
            mock_delete.eq.assert_called_once_with('document_id', 123)
            assert result["success"] is True

    async def test_insert_chunks_raises_on_length_mismatch(self):
        """Test that ValueError is raised when chunks and embeddings length mismatch."""

        with patch('elite_craft.services.database_uploading.create_client') as mock_create_client:

            # Setup mock client
            mock_client = Mock()
            mock_create_client.return_value = mock_client

            # Create service
            service = SupabaseUploadService(
                supabase_url="https://fake.supabase.co",
                supabase_key="fake-key"
            )

            # Test data with mismatched lengths
            chunks = ["chunk 1", "chunk 2"]
            embeddings = [[0.1, 0.2]]  # Only 1 embedding for 2 chunks!

            # Act & Assert: Should raise ValueError
            with pytest.raises(ValueError, match="Length mismatch: 2 chunks but 1 embeddings"):
                await service.insert_chunks(
                    chunks=chunks,
                    embeddings=embeddings,
                    document_id=123,
                    url="https://example.com"
                )