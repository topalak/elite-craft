import asyncio
from config import settings

from elite_craft.services.chunking import Chunker
from elite_craft.services.crawling import Crawler
from elite_craft.services.embedding import Embedder
from elite_craft.services.database_uploading import SupabaseUploadService

#TODO, 1- initialized converter and chunker in __init__ to avoid latency
# 2- add try exception block and docstring for each method
# 3



url_to_crawl = "https://docs.langchain.com/oss/python/langgraph/overview"

class UpdateDBPipeline:  #todo init
    def __init__(self):
        self.chunker = Chunker()
        self.crawler = Crawler()
        self.embedder = Embedder(model="embeddinggemma", model_provider_url=settings.OLLAMA_HOST_LOCAL)
        self.uploader = SupabaseUploadService()


    async def process(self, url: str):
        """
        Full pipeline: crawl → chunk → embed → upload to database.

        Steps:
        1. Crawl URL and extract metadata (url, source, crawled_time, body_text)
        2. Upload metadata to Supabase
        3. Chunk the document using HybridChunker
        4. Generate embeddings for chunks
        5. Upload chunks with embeddings to Supabase

        Args:
            url: URL to process

        Returns:
            Dict with processing results
        """

        # Step 1: Crawl and get structured data
        crawled_data = await self.crawler.crawl(url=url)
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
            crawled_data["url"]
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

        return {
            "url": crawled_data["url"],
            "source": crawled_data["source"],
            "chunks_uploaded": upload_result["total_chunks"],
            "success": upload_result["success"]
        }

    async def process_sitemap(self, urls: list[str]) -> list[dict]:
        """
        Process multiple URLs concurrently from a sitemap.

        All URLs are processed in parallel using asyncio.gather().
        This is significantly faster than processing sequentially.

        Args:
            urls: List of URLs to process (from sitemap.xml)

        Returns:
            List of results for each URL (successful and failed)
        """

        # Process all URLs concurrently
        results = await asyncio.gather(
            *[self.process(url) for url in urls],
            return_exceptions=True  # Continue even if some URLs fail
        )
        #EXPLAINED IN OUT FOLDER (async_gather_explanation)


        ############ MONITORING ################
        # Separate successful from failed
        successful = [r for r in results if isinstance(r, dict) and r.get("success")]
        failed = [r for r in results if isinstance(r, Exception)]

        print(f"\n Successful: {len(successful)}/{len(urls)}")
        if failed:
            print(f"Failed: {len(failed)}/{len(urls)}")
            print("\nFailed URLs:")
            for i, error in enumerate(failed, 1):
                print(f"  {i}. {error}")

        return results


async def main():
    """
    Example usage of the pipeline.

    Demonstrates both single URL processing and concurrent sitemap processing.
    """
    pipeline = UpdateDBPipeline()

    # Example 1: Process a single URL
    print("=" * 60)
    print("Example 1: Processing Single URL")
    print("=" * 60)
    result = await pipeline.process(url=url_to_crawl)
    print(f"\n✅ Result: {result}\n")

    """
    # Example 2: Process multiple URLs concurrently
    print("=" * 60)
    print("Example 2: Processing Multiple URLs Concurrently")
    print("=" * 60)

    # Simulated sitemap URLs (replace with actual sitemap parser later)
    example_urls = [
        "https://docs.langchain.com/oss/python/langgraph/overview",
        "https://docs.langchain.com/oss/python/langgraph/tutorials/introduction",
        "https://docs.langchain.com/oss/python/langgraph/how-to/tool-calling",
        # Add more URLs from sitemap.xml parsing
    ]

    # Process all URLs concurrently
    concurrent_results = await pipeline.process_sitemap(example_urls)

    print(f"\n Total processed: {len(concurrent_results)} URLs")
    """

    return result


if __name__ == "__main__":
    asyncio.run(main())