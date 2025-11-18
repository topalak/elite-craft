"""
Service for embedding text chunks using Ollama embedding models.

Supports multiple techniques for improved RAG performance including
Contextual Retrieval (using prompt caching) and reranking.
"""

from elite_craft.model_provider import ModelConfig
from config import settings


class Embedder:
    """
    Service for generating embeddings from text chunks.

    Uses Ollama's embedding models (default: embeddinggemma) to convert
    text chunks into vector representations for storage in Qdrant.
    """

    def __init__(
        self,
        model: str = None,
        model_provider_url: str = None,
    ):

        self.model = model
        self.model_provider_url = model_provider_url

        embedding_model_config = ModelConfig(model=self.model, model_provider_url=self.model_provider_url)
        self.embedding_model = embedding_model_config.get_embedding()

    def embed(self, chunks: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of text chunks.

        Args:
            chunks: List of text strings to embed

        Returns:
            List of embedding vectors (each vector is a list of floats)

        """
        # TODO: Implement batch optimizing with async for large datasets
        embeddings = self.embedding_model.embed_documents(chunks)
        return embeddings






