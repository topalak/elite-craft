"""
Unit tests for Embedding service.
"""

import pytest
from unittest.mock import Mock, patch

from elite_craft.services.embedding import Embedder


class TestEmbedder:
    """Test Embedder service."""

    def test_embedder_initializes_with_model(self):
        """Test that Embedder initializes with correct model configuration."""
        with patch('elite_craft.services.embedding.ModelConfig') as MockModelConfig:
            mock_config_instance = MockModelConfig.return_value
            mock_embedding_model = Mock()
            mock_config_instance.get_embedding.return_value = mock_embedding_model

            embedder = Embedder(model="test-embedding-model")

            MockModelConfig.assert_called_once_with(model="test-embedding-model")
            mock_config_instance.get_embedding.assert_called_once()
            assert embedder.embedding_model == mock_embedding_model

    def test_embed_returns_embeddings(self):
        """Test that embed calls embed_documents and returns embeddings."""
        with patch('elite_craft.services.embedding.ModelConfig') as MockModelConfig:
            mock_config_instance = MockModelConfig.return_value
            mock_embedding_model = Mock()

            fake_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            mock_embedding_model.embed_documents.return_value = fake_embeddings
            mock_config_instance.get_embedding.return_value = mock_embedding_model

            embedder = Embedder(model="test-model")
            chunks = ["chunk 1", "chunk 2"]
            result = embedder.embed(chunks, url="https://example.com")

            mock_embedding_model.embed_documents.assert_called_once_with(chunks)
            assert result == fake_embeddings
            assert len(result) == 2

    def test_embed_raises_error_on_count_mismatch(self):
        """Test that embed raises ValueError when embedding count doesn't match chunk count."""
        with patch('elite_craft.services.embedding.ModelConfig') as MockModelConfig:
            mock_config_instance = MockModelConfig.return_value
            mock_embedding_model = Mock()

            # Return WRONG number of embeddings (2 embeddings for 3 chunks)
            fake_embeddings = [[0.1, 0.2], [0.3, 0.4]]  # Only 2 embeddings
            mock_embedding_model.embed_documents.return_value = fake_embeddings
            mock_config_instance.get_embedding.return_value = mock_embedding_model

            embedder = Embedder(model="test-model")
            chunks = ["chunk 1", "chunk 2", "chunk 3"]  # 3 chunks!

            # Should raise ValueError due to mismatch
            with pytest.raises(ValueError, match="Embedding count mismatch: 3 chunks produced 2 embeddings"):
                embedder.embed(chunks, url="https://example.com")