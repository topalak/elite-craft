"""
Service for embedding text chunks using Ollama embedding models.

Supports multiple techniques for improved RAG performance including
Contextual Retrieval (using prompt caching) and reranking.
"""
import logging

from elite_craft.model_provider import ModelConfig


logger = logging.getLogger(__name__)


class Embedder:
    """
    Service for generating embeddings from text chunks.

    Uses Ollama's embedding models (default: embeddinggemma) to convert
    text chunks into vector representations for storage in Supabase.
    """

    def __init__(self, model: str):
        """
        Initialize embedder with specified model.
        """
        embedding_model_config = ModelConfig(model=model)
        self.embedding_model = embedding_model_config.get_embedding()

    def embed(self, chunks: list[str], url: str) -> list[list[float]]:
        """
        Generate embeddings for a list of text chunks.

        Args:
            chunks: List of text strings to embed
            url: Source URL (for logging and error messages)

        Returns:
            List of embedding vectors (each vector is a list of floats)

        Raises:
            ValueError: If embedding count doesn't match chunk count
        """
        embeddings = self.embedding_model.embed_documents(chunks)

        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Embedding count mismatch: {len(chunks)} chunks "
                f"produced {len(embeddings)} embeddings for {url}"
            )

        logger.info(
            f"[EMBED COMPLETE] Generated {len(embeddings)} "
            f"embeddings for {url} chunks"
        )
        return embeddings

