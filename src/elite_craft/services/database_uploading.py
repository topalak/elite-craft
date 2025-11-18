from supabase import create_client, Client
from config import settings
import asyncio


class SupabaseUploadService:
    """
    Service for uploading crawled data and embeddings to Supabase.

    Handles batch insertion of metadata and chunks with embeddings
    to PostgreSQL with pgvector extension. Uses asyncio.to_thread()
    to wrap sync Supabase client for concurrent operations.
    """

    BATCH_SIZE = 100  # Optimal batch size for PostgreSQL with vector embeddings

    def __init__(self):
        """Initialize Supabase client with service key credentials."""
        self.SUPABASE_URL = settings.SUPABASE_URL
        self.SUPABASE_KEY = settings.SUPABASE_SERVICE_KEY
        self.supabase_client: Client = create_client(self.SUPABASE_URL, self.SUPABASE_KEY)

    async def insert_metadata(self, content_to_insert: dict) -> dict:
        """
        Insert or update document metadata using upsert (handles duplicate URLs).

        Args:
            content_to_insert: Dict with keys: url, source, crawled_time, body_text

        Returns:
            Supabase response with inserted/updated record

        Schema:
            - id: serial primary key
            - url: varchar not null unique
            - source: varchar not null
            - crawled_time: timestamp with time zone not null
            - body_text: text

        Note:
            Uses upsert to update existing records if URL already exists.
            This prevents duplicate key violations on re-crawling.
        """
        return await asyncio.to_thread(
            self.supabase_client.table('metadata')
            .upsert(content_to_insert, on_conflict='url')
            .execute
        )
    """
    old version 
        return await asyncio.to_thread(
        self.supabase_client.table('metadata').insert(content_to_insert).execute
    )
    """

    async def insert_chunks(
            self,
            chunks: list[str],
            embeddings: list[list[float]],
            url: str
    ) -> dict:

        """
        Insert text chunks with embeddings in batches.

        Args:
            chunks: List of text chunks from document
            embeddings: List of embedding vectors (must match chunks length)
            url: URL of source document (foreign key to metadata table)

        Returns:
            Dict with insertion statistics:
                - total_chunks: Number of chunks inserted
                - batches: Number of batches processed
                - success: Boolean indicating success

        Raises:
            ValueError: If chunks and embeddings length mismatch
        """

        # Validate input
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Length mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
            )

        # Prepare batch records
        chunk_records = []
        for chunk_number, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_records.append({
                "url": url,
                "chunk_number": chunk_number,
                "content": str(chunk_text),
                "embedding": embedding
            })

        # Insert in batches
        for i in range(0, len(chunk_records), self.BATCH_SIZE):
            batch = chunk_records[i:i + self.BATCH_SIZE]

            # Wrap sync Supabase call in thread to not block event loop
            await asyncio.to_thread(
                self.supabase_client.table('chunks').insert(batch).execute
            )

            print(f"Inserted batch {i // self.BATCH_SIZE + 1}: {len(batch)} chunks")

        return {
            "total_chunks": len(chunk_records),
            "success": True
        }
