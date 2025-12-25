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
        chunk_overlap: int,
        embedding_batch_size: int = 20
    ):
        self.chunker = Chunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.embedder = Embedder(
            model=embedding_model,
            batch_size=embedding_batch_size
        )
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

        Raises:
            Exception: Re-raises any exception with URL context added
        """
        try:
            # Step 1: Crawl and get structured data
            logger.debug(f"[PIPELINE] Step 1/5: Crawling {url}")
            crawled_data = await crawl(url=url)

            # Step 2: Upload metadata to database
            logger.debug(f"[PIPELINE] Step 2/5: Uploading metadata for {url}")
            document_id = await self.uploader.insert_document(crawled_data)

            # Step 3: Chunk the document
            # CPU-bound - run in thread to not block event loop
            logger.debug(f"[PIPELINE] Step 3/5: Chunking {url}")
            chunks = await asyncio.to_thread(
                self.chunker.chunk,
                content=crawled_data.body_text,
                url=url,
            )
            logger.info(f"[PIPELINE] Generated {len(chunks)} chunks for {url}")

            # Step 4: Generate embeddings
            # GPU-bound - run in thread to not block event loop
            logger.debug(f"[PIPELINE] Step 4/5: Generating embeddings for {url}")
            embeddings = await asyncio.to_thread(
                self.embedder.embed,
                chunks=chunks,
                url=url
            )

            # Step 5: Upload chunks with embeddings
            logger.debug(f"[PIPELINE] Step 5/5: Uploading chunks to database for {url}")
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
                f"[PIPELINE COMPLETE] {result.url}: "
                f"{result.chunks_uploaded} chunks uploaded successfully"
            )
            return result

        except Exception as e:
            logger.error(
                f"[PIPELINE FAILED] URL: {url} | "
                f"Step: {self._get_current_step(e)} | "
                f"Error: {type(e).__name__}: {str(e)}"
            )
            raise

    @staticmethod
    def _get_current_step(exception: Exception) -> str:
        """Determine which pipeline step failed based on exception context."""
        error_msg = str(exception).lower()
        if "crawl" in error_msg:
            return "Crawling"
        elif "chunk" in error_msg:
            return "Chunking"
        elif "embed" in error_msg or "context length" in error_msg:
            return "Embedding"
        elif "upload" in error_msg or "database" in error_msg:
            return "Database Upload"
        else:
            return "Unknown"

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
        supabase_url=settings.SUPABASE_URL.get_secret_value(),
        supabase_key=settings.SUPABASE_SERVICE_ROLE_SECRET_KEY.get_secret_value(),
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        embedding_batch_size=settings.EMBEDDING_BATCH_SIZE
    )

    urls = [
        "https://docs.langchain.com/oss/python/langchain/install",
        "https://docs.langchain.com/oss/python/langchain/quickstart",
        "https://docs.langchain.com/oss/python/langchain/philosophy",
        "https://docs.langchain.com/oss/python/langchain/agents",
        "https://docs.langchain.com/oss/python/langchain/models",
        "https://docs.langchain.com/oss/python/langchain/messages",
        "https://docs.langchain.com/oss/python/langchain/tools",
        "https://docs.langchain.com/oss/python/langchain/short-term-memory",
        "https://docs.langchain.com/oss/python/langchain/streaming",
        "https://docs.langchain.com/oss/python/langchain/structured-output",
        "https://docs.langchain.com/oss/python/langchain/middleware/overview",
        "https://docs.langchain.com/oss/python/langchain/middleware/built-in",
        "https://docs.langchain.com/oss/python/langchain/middleware/custom",
        "https://docs.langchain.com/oss/python/langchain/guardrails",
        "https://docs.langchain.com/oss/python/langchain/runtime",
        "https://docs.langchain.com/oss/python/langchain/context-engineering",
        "https://docs.langchain.com/oss/python/langchain/mcp",
        "https://docs.langchain.com/oss/python/langchain/human-in-the-loop",
        "https://docs.langchain.com/oss/python/langchain/multi-agent",
        "https://docs.langchain.com/oss/python/langchain/multi-agent/subagents",
        "https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs",
        "https://docs.langchain.com/oss/python/langchain/multi-agent/skills",
        "https://docs.langchain.com/oss/python/langchain/multi-agent/router",
        "https://docs.langchain.com/oss/python/langchain/multi-agent/custom-workflow",
        "https://docs.langchain.com/oss/python/langchain/retrieval",
        "https://docs.langchain.com/oss/python/langchain/long-term-memory",
        "https://docs.langchain.com/oss/python/langchain/studio",
        "https://docs.langchain.com/oss/python/langchain/test",
        "https://docs.langchain.com/oss/python/langchain/ui",
        "https://docs.langchain.com/oss/python/langchain/deploy",
        "https://docs.langchain.com/oss/python/langchain/observability"
    ]

    # Process all URLs concurrently
    concurrent_results = await pipeline.process_multiple_urls(urls)

    logger.info(f"Total processed: {len(concurrent_results)} URLs")

    return concurrent_results


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())