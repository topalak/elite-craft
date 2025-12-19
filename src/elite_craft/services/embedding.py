"""
Service for embedding text chunks using Ollama embedding models.

Supports multiple techniques for improved RAG performance including
Contextual Retrieval (using prompt caching) and reranking.
"""
import logging

from elite_craft.model_provider import ModelConfig


logger = logging.getLogger(__name__)

# Context length limits for embedding models (in tokens)
# Approximate conversion: 1 token ≈ 4 characters
EMBEDDING_MODEL_CONTEXT_LIMITS = {
    "nomic-embed-text:v1.5": 8192,
    "embeddinggemma": 8192,
}


class Embedder:
    """
    Service for generating embeddings from text chunks.

    Uses Ollama's embedding models (default: embeddinggemma) to convert
    text chunks into vector representations for storage in Supabase.
    Processes chunks in batches to prevent API context length errors.
    """

    def __init__(self, model: str, batch_size: int = 20):
        """
        Initialize embedder with specified model.

        Args:
            model: Embedding model name
            batch_size: Number of chunks to embed per API call (default: 20)
        """
        self.model_name = model
        self.batch_size = batch_size
        embedding_model_config = ModelConfig(model=model)
        self.embedding_model = embedding_model_config.get_embedding()

        # Get context limit for this model
        self.max_tokens = EMBEDDING_MODEL_CONTEXT_LIMITS.get(model, 8192)
        self.max_chars = self.max_tokens * 4  # Approximate: 1 token ≈ 4 chars

        logger.info(
            f"[EMBEDDER INIT] Model: {model}, "
            f"Max context: {self.max_tokens} tokens (~{self.max_chars} chars), "
            f"Batch size: {batch_size}"
        )

    def embed(self, chunks: list[str], url: str) -> list[list[float]]:
        """
        Generate embeddings for a list of text chunks with batching.

        Processes chunks in batches to prevent exceeding API context limits.
        Logs warning for any oversized individual chunks.

        Args:
            chunks: List of text strings to embed
            url: Source URL (for logging and error messages)

        Returns:
            List of embedding vectors (each vector is a list of floats)

        Raises:
            ValueError: If embedding count doesn't match chunk count
        """
        # Check for oversized individual chunks
        oversized = [
            (idx, len(c)) for idx, c in enumerate(chunks)
            if len(c) > self.max_chars
        ]
        if oversized:
            logger.warning(
                f"[EMBED WARNING] {url} has {len(oversized)} chunks "
                f"exceeding {self.max_chars} chars: {oversized[:3]}"
            )

        # Process chunks in batches
        all_embeddings = []
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1

            logger.debug(
                f"[EMBED BATCH {batch_num}/{total_batches}] "
                f"Processing {len(batch)} chunks for {url}"
            )

            # Send batch to embedding API
            batch_embeddings = self.embedding_model.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)

        # Validate result
        if len(all_embeddings) != len(chunks):
            raise ValueError(
                f"Embedding count mismatch: {len(chunks)} chunks "
                f"produced {len(all_embeddings)} embeddings for {url}"
            )

        logger.info(
            f"[EMBED COMPLETE] Generated {len(all_embeddings)} embeddings "
            f"in {total_batches} batches for {url}"
        )
        return all_embeddings

