"""
Service for embedding text chunks using Ollama embedding models.

Supports multiple techniques for improved RAG performance including
Contextual Retrieval (using prompt caching) and reranking.
"""
import logging

from pydantic import AnyUrl

from elite_craft.model_provider import ModelConfig

logger = logging.getLogger(__name__)

class Embedder:
    """
    Service for generating embeddings from text chunks.

    Uses Ollama's embedding models (default: embeddinggemma) to convert
    text chunks into vector representations for storage in Supabase
    """

    def __init__(
        self,
        model: str,
    ):
        embedding_model_config = ModelConfig(model=model)
        self.embedding_model = embedding_model_config.get_embedding()

    def embed(self, chunks: list[str], url:str) -> list[list[float]]:
        """
        Generate embeddings for a list of text chunks.

        Args:
            chunks: List of text strings to embed
            url: URL to retrieve embeddings from

        Returns:
            List of embedding vectors (each vector is a list of floats)
        """

        embeddings = self.embedding_model.embed_documents(chunks)

        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Embedding count mismatch: {len(chunks)} chunks produced" f"{len(embeddings)} embeddings for {url}")

        logger.info(f"[EMBED COMPLETE] Generated {len(embeddings)} embeddings for {url} chunks")
        return embeddings
