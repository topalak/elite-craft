import asyncio
from typing import TypedDict

import logging

from elite_craft.services.chunking import Chunker
from elite_craft.services.crawling import crawl
from elite_craft.services.database_uploading import SupabaseUploadService
from elite_craft.services.embedding import Embedder

logger = logging.getLogger(__name__)

class PipelineResult(TypedDict):
    """
    Types of single pipeline's result
    """

    url: str
    source: str
    chunks_uploaded: int
    success: bool

class UpdateDBPipeline:
    def __init__(self, embedding_model:str, model_provider_url:str, supabase_url:str, supabase_key:str, batch_size:int = 100):
        self.chunker = Chunker()
        self.embedder = Embedder(model=embedding_model, model_provider_url=model_provider_url)
        self.uploader = SupabaseUploadService(supabase_url=supabase_url, supabase_key=supabase_key, batch_size=batch_size)


    async def pipeline(self, url: str) -> PipelineResult:
        """
        Full pipeline: crawl → chunk → embed → upload to database.

        Args:
            url: URL to process

        Returns:
            Dict with processing results
        """

        # Step 1: Crawl and get structured data
        crawled_data = await crawl(url=url)
        # crawled_data = {
        #     "body_text": str, #content
        #     "crawled_time": datetime,
        #     "url": str,
        #     "source": str
        # }

        # Step 2: Upload metadata to database
        await self.uploader.insert_metadata(crawled_data)

        # Step 3: Chunk the document (CPU-bound - run in thread to not block event loop)
        chunks = await asyncio.to_thread(
            self.chunker.chunk,
            crawled_data["body_text"]
        )

        # Step 4: Generate embeddings (GPU-bound - run in thread to not block event loop)
        embeddings = await asyncio.to_thread(
            self.embedder.embed,
            chunks
        )

        # Step 5: Upload chunks with embeddings
        upload_result = await self.uploader.insert_chunks(
            chunks=chunks,
            embeddings=embeddings,
            url=crawled_data["url"]
        )

        result: PipelineResult =  {
            "url": crawled_data["url"],
            "source": crawled_data["source"],
            "chunks_uploaded": upload_result["total_chunks"],
            "success": True
        }

        logger.info(f"Pipeline completed for {crawled_data['url']}: {result['chunks_uploaded']} chunks")
        return result

    async def process(self, urls: list[str]) -> list[dict]:
        """
        Process single or multiple URLs asynchronously.

        Args:
            urls: List of URLs to process

        Returns:
            List of results for each URL (successful and failed)
        """

        # Process all URLs concurrently
        results = await asyncio.gather(
            *[self.pipeline(url) for url in urls],
            return_exceptions=True  # Continue even if some URLs fail
        )

        # Count results
        successful = [r for r in results if isinstance(r, dict) and r.get("success")]
        failed = [r for r in results if isinstance(r, Exception)]

        # Log results
        logger.info(f"Processing complete: {len(successful)}/{len(urls)} successful")

        if failed:
            logger.error(f"{len(failed)} URLs failed")
            for error in failed:
                logger.error(f"Error: {error}")

        return results


async def main():
    from src.config import settings

    """
        Executes update db pipeline asynchronously.
    """
    pipeline = UpdateDBPipeline(embedding_model='embeddinggemma',
                                model_provider_url=settings.OLLAMA_HOST_LOCAL,
                                supabase_url=settings.SUPABASE_URL,
                                supabase_key=settings.SUPABASE_SECRET_KEY)

    urls = [
        "https://docs.langchain.com/oss/python/langgraph/overview",
        "https://docs.langchain.com/oss/python/langgraph/tutorials/introduction",
        "https://docs.langchain.com/oss/python/langgraph/how-to/tool-calling",
    ]

    # Process all URLs concurrently
    concurrent_results = await pipeline.process(urls)

    logger.info(f"Total processed: {len(concurrent_results)} URLs")

    return concurrent_results


if __name__ == "__main__":
    asyncio.run(main())