import asyncio
import logging

from pydantic import AnyUrl
from supabase import Client, create_client

from config import settings
from elite_craft.enums import GeneralEnums
from elite_craft.services.schemas import CrawledData


logger = logging.getLogger(__name__)


class SupabaseUploadService:
    """
    Service for uploading metadata and embeddings to Supabase.

    Handles batch insertion of metadata and chunks with embeddings
    to PostgreSQL with pgvector extension.

    Uses semaphore to limit concurrent chunk uploads to prevent
    HTTP connection pool exhaustion.
    """

    # Class-level semaphore to limit concurrent chunk upload operations
    # Set to 1 because each upload does multiple DB operations
    # (SELECT, DELETE, INSERT batches)
    _upload_semaphore = asyncio.Semaphore(1)

    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        batch_size: int = None
    ):
        # Create client with explicit schema set to 'private'
        self.supabase_client: Client = create_client(
            supabase_url,
            supabase_key,
        )
        self.batch_size = (
            batch_size if batch_size is not None
            else settings.DB_UPLOAD_BATCH_SIZE
        )


    async def insert_document(self, content_to_insert: CrawledData) -> int:
        """
        Insert or update document metadata using upsert.

        Args:
            content_to_insert: Dict with keys: url, source, crawled_time, body_preview

        Returns:
            None
        """

        url = content_to_insert.url
        body_text = content_to_insert.body_text

        db_record = content_to_insert.model_dump(
            mode='json',
            exclude={GeneralEnums.BODY_TEXT}
        )
        db_record[GeneralEnums.BODY_PREVIEW] = (
            body_text[:settings.BODY_PREVIEW_END]
        )

        # Use upsert - updates if exists, inserts if new
        # Wrap sync Supabase call in thread to not block event loop
        response = await asyncio.to_thread(
            self.supabase_client.table(GeneralEnums.DOCUMENTS)
            .upsert(db_record, on_conflict='url')
            .execute
        )

        idx = response.data[0]['id']
        logger.info(
            f"[DB METADATA COMPLETE] Metadata upserted for: {url} "
            f"with {idx}"
        )

        return idx

    async def insert_chunks(
            self,
            chunks: list[str],
            embeddings: list[list[float]],
            document_id: int,
            url: str | AnyUrl,
    ) -> dict:
        """
        Insert text chunks with embeddings in batches.
        Deletes existing chunks for the URL first if they exist.

        Uses class-level semaphore to limit concurrent uploads and prevent
        httpx connection pool exhaustion.

        Args:
            chunks: List of text chunks from document
            embeddings: List of embedding vectors
            document_id: Unique document id
            url: URL to insert chunk for

        Returns:
            Dict with insertion statistics:
                - total_chunks: Number of chunks inserted
                - success: Boolean indicating success

        Raises:
            ValueError: If chunks and embeddings length mismatch
        """
        # Acquire semaphore to limit concurrent uploads
        async with self._upload_semaphore:

            # Validate input
            if len(chunks) != len(embeddings):
                raise ValueError(
                    f"Length mismatch: {len(chunks)} chunks but "
                    f"{len(embeddings)} embeddings"
                )

            # Check if chunks exist for this URL
            existing_chunks = await asyncio.to_thread(
                self.supabase_client.table(GeneralEnums.CHUNKS)
                .select('id')
                .eq('document_id', document_id)
                .execute
            )

            if existing_chunks.data:
                logger.info(
                    f"[DB CHUNKS] Found {len(existing_chunks.data)} "
                    f"existing chunks, deleting for: {url}"
                )
                await asyncio.to_thread(
                    self.supabase_client.table(GeneralEnums.CHUNKS)
                    .delete()
                    .eq('document_id', document_id)
                    .execute
                )
                logger.info(
                    f"[DB CHUNKS] Deleted {len(existing_chunks.data)} "
                    f"existing chunks for: {url}"
                )

            # strict=True raises error if lengths differ
            # We validate before but this makes it more robust
            chunk_records = [
                {
                    "document_id": document_id,
                    "chunk_id_in_document": chunk_id_in_document,
                    "content": str(chunk_text),
                    "embedding": embedding
                }
                for chunk_id_in_document, (chunk_text, embedding)
                in enumerate(zip(chunks, embeddings, strict=True))
            ]

            for i in range(0, len(chunk_records), self.batch_size):
                batch = chunk_records[i:i + self.batch_size]

                # Wrap sync Supabase call in thread to not block event loop
                await asyncio.to_thread(
                    self.supabase_client.table(GeneralEnums.CHUNKS)
                    .insert(batch)
                    .execute
                )

            logger.info(
                f"[DB CHUNKS COMPLETE] Successfully inserted "
                f"{len(chunk_records)} chunks for: {url}"
            )

            return {
                "total_chunks": len(chunk_records),
                "success": True
            }
