from typing import Final
import asyncio
import logging

from supabase import create_client, Client

BODY_PREVIEW_END: Final = 3000
logger = logging.getLogger(__name__)

class SupabaseUploadService:
    """
    Service for uploading metadata and embeddings to Supabase.

    Handles batch insertion of metadata and chunks with embeddings
    to PostgreSQL with pgvector extension.
    """

    def __init__(self, supabase_url:str, supabase_key:str, batch_size:int=100):
        """Initialize Supabase client with service key credentials."""
        self.supabase_client: Client = create_client(supabase_url, supabase_key)
        self.batch_size: int = batch_size


    async def insert_metadata(self, content_to_insert: dict) -> None:
        """
        Insert or update document metadata using upsert.

        Args:
            content_to_insert: Dict with keys: url, source, crawled_time, body_preview

        Returns:
            None
        """

        url = content_to_insert['url']
        body_text = content_to_insert['body_text']

        db_record = {k: v for k, v in content_to_insert.items() if k != 'body_text'}
        db_record['body_preview'] = body_text[:BODY_PREVIEW_END]


        # Use upsert - updates if exists, inserts if new
        await asyncio.to_thread(
            self.supabase_client.table('metadata')
            .upsert(db_record, on_conflict='url')
            .execute
        )

        logger.info(f"Metadata upserted for: {url}")


    async def insert_chunks(
            self,
            chunks: list[str],
            embeddings: list[list[float]],
            url: str
    ) -> dict:
        """
        Insert text chunks with embeddings in batches.
        Deletes existing chunks for the URL first if they exist.

        Args:
            chunks: List of text chunks from document
            embeddings: List of embedding vectors
            url: URL of source document (foreign key to metadata table)

        Returns:
            Dict with insertion statistics:
                - total_chunks: Number of chunks inserted
                - success: Boolean indicating success

        Raises:
            ValueError: If chunks and embeddings length mismatch
        """

        # Validate input
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Length mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
            )

        # Check if chunks exist for this URL
        existing_chunks = await asyncio.to_thread(
            self.supabase_client.table('chunks').select('id').eq('url', url).execute
        )

        if existing_chunks.data:
            logger.info(f"Existing chunks found in database")
            await asyncio.to_thread(
                self.supabase_client.table('chunks').delete().eq('url', url).execute
            )
            logger.info(f"Deleted {len(existing_chunks.data)} existing chunks for: {url}")

        # Prepare batch records by using list comprehension
        chunk_records = [
            {
                "url": url,
                "chunk_number": chunk_number,
                "content": str(chunk_text),
                "embedding": embedding
            }
            for chunk_number, (chunk_text, embedding) in enumerate(zip(chunks, embeddings))
        ]

        # Insert in batches
        for i in range(0, len(chunk_records), self.batch_size):
            batch = chunk_records[i:i + self.batch_size]

            # Wrap sync Supabase call in thread to not block event loop
            await asyncio.to_thread(
                self.supabase_client.table('chunks').insert(batch).execute
            )

            logger.info(f"Inserted batch {i // self.batch_size + 1}: {len(batch)} chunks")

        logger.info(f"Successfully inserted {len(chunk_records)} chunks for: {url}")

        return {
            "total_chunks": len(chunk_records),
            "success": True
        }
