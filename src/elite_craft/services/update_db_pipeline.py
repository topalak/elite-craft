import asyncio
import logging

from elite_craft.services.chunking import Chunker
from elite_craft.services.crawling import crawl
from elite_craft.services.database_uploading import SupabaseUploadService
from elite_craft.services.embedding import Embedder
from elite_craft.services.schemas import PipelineResults


logger = logging.getLogger(__name__)


class UpdateDBPipeline:
    """
    End-to-end pipeline for updating documentation database.

    Orchestrates the complete flow: crawling → chunking → embedding →
    database upload. Supports concurrent processing of multiple URLs.
    """

    def __init__(
        self,
        embedding_model: str,
        supabase_url: str,
        supabase_key: str,
        chunk_size: int,
        chunk_overlap: int
    ):
        self.chunker = Chunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.embedder = Embedder(model=embedding_model)
        self.uploader = SupabaseUploadService(
            supabase_url=supabase_url,
            supabase_key=supabase_key
        )


    async def process_single_url(self, url: str) -> PipelineResults:
        """
        Full pipeline: crawl → chunk → embed → upload to database.

        Args:
            url: URL to process

        Returns:
            Dict with processing results
        """

        # Step 1: Crawl and get structured data
        crawled_data = await crawl(url=url)

        # Step 2: Upload metadata to database
        document_id = await self.uploader.insert_document(crawled_data)

        # Step 3: Chunk the document
        # CPU-bound - run in thread to not block event loop
        chunks = await asyncio.to_thread(
            self.chunker.chunk,
            content=crawled_data.body_text,
            url=url,
        )

        # Step 4: Generate embeddings
        # GPU-bound - run in thread to not block event loop
        embeddings = await asyncio.to_thread(
            self.embedder.embed,
            chunks=chunks,
            url=url
        )

        # Step 5: Upload chunks with embeddings
        upload_result = await self.uploader.insert_chunks(
            chunks=chunks,
            embeddings=embeddings,
            document_id=document_id,
            url=url
        )

        result = PipelineResults(
            url=url,
            source=crawled_data.source,
            chunks_uploaded=upload_result["total_chunks"],
            success=True
        )

        logger.info(
            f"Pipeline completed for {result.url}: "
            f"{result.chunks_uploaded} chunks"
        )
        return result

    async def process_multiple_urls(self, urls: list[str]) -> list[dict]:
        """
        Process single or multiple URLs asynchronously.

        Args:
            urls: List of URLs to process

        Returns:
            List of results for each URL (successful and failed)
        """

        # Process all URLs concurrently
        results = await asyncio.gather(
            *[self.process_single_url(url) for url in urls],
            return_exceptions=True  # Continue even if some URLs fail
        )

        # Count results
        successful = [
            r for r in results
            if isinstance(r, PipelineResults) and r.success
        ]
        failed = [r for r in results if isinstance(r, Exception)]

        # Log results
        logger.info(
            f"Processing complete: {len(successful)}/{len(urls)} successful"
        )

        if failed:
            logger.error(f"{len(failed)} URLs failed")
            for error in failed:
                logger.error(f"Error: {error}")

        return results


async def main():
    """Execute update database pipeline asynchronously for testing."""
    from config import settings

    logger.setLevel(level=settings.LOGGING_LEVEL)

    pipeline = UpdateDBPipeline(
        embedding_model=settings.EMBEDDING_MODEL,
        supabase_url=settings.SUPABASE_URL,
        supabase_key=settings.SUPABASE_SERVICE_ROLE_SECRET_KEY,
        chunk_size=settings.CHUNK_SIZE,
    )

    urls = [
        "https://docs.langchain.com/oss/python/langchain/agents",
        "https://docs.langchain.com/oss/python/langchain/messages",
        "https://docs.langchain.com/oss/python/langchain/models",
        "https://docs.langchain.com/oss/python/langchain/tools",
        "https://docs.langchain.com/oss/python/langchain/structured-output",
        "https://docs.langchain.com/oss/python/langchain/middleware/built-in",
        "https://docs.langchain.com/oss/python/langchain/overview",
        "https://docs.langchain.com/oss/python/langchain/streaming",
        "https://docs.langchain.com/oss/python/langchain/guardrails",
        "https://docs.langchain.com/oss/python/langchain/runtime",
        "https://docs.langchain.com/oss/python/langchain/context-engineering",
        "https://docs.langchain.com/oss/python/langchain/human-in-the-loop",

        "https://reference.langchain.com/python/langchain/agents/",
        "https://reference.langchain.com/python/langchain/middleware/",
        "https://reference.langchain.com/python/langchain/models/",
        "https://reference.langchain.com/python/langchain/messages/",
        "https://reference.langchain.com/python/langchain/tools/",
    ]

    # Process all URLs concurrently
    concurrent_results = await pipeline.process_multiple_urls(urls)

    logger.info(f"Total processed: {len(concurrent_results)} URLs")

    return concurrent_results


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())