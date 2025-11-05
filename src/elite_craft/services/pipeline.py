from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

import asyncio
from crawl4ai import AsyncWebCrawler#, BrowserConfig

from elite_craft.model_provider import ModelConfig
from config import settings

#TODO, 1- we need to initialize some of them in __init__ to avoid latency  (converter, chunker)
# 2- add try exception block and docstring for each method
# 3

url_to_crawl = "https://docs.langchain.com/oss/python/langgraph/overview"

embedding_model_config = ModelConfig(model="embeddinggemma", model_provider_url=settings.OLLAMA_HOST_MY_LOCAL)
cached_llm_config = ModelConfig(model='gpt-oss:20b-cloud', use_ollama_cloud=True)


class Pipeline:
    def __init__(self):
        self.embedding_model = embedding_model_config.get_embedding()
        self.cached_llm = cached_llm_config.get_llm()

    @staticmethod
    async def crawl(url: str):
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
        return result

    @staticmethod
    def chunking(crawled_result:str):
        doc = DocumentConverter().convert(source=crawled_result).document
        chunker = HybridChunker()
        chunk_iter = chunker.chunk(dl_doc=doc)

        return list(chunk_iter)


    async def contextual_chunking(self, chunks:list[str], document:str):
        prompt = """ <document>
        {document}
        </document>
    
        Here is the chunk we want to situate within the whole document:
        <chunk>
        {chunk}
        </chunk>
    
        Please give a short succinct context to situate this chunk within the overall 
        document for the purposes of improving search retrieval of the chunk. 
        Answer only with the succinct context and nothing else."""


        async def _chunk_enhancer(chunk:str):
            """
            Uses llm to enhance the chunk.

            Args:
                chunk (str): The chunk to enhance

            Returns:
                str: list of enhanced chunks
            """

            formatted_prompt = prompt.format(document=document, chunk=chunk)
            enhanced_chunk = await self.cached_llm.ainvoke([
                {"role":"system", "content":formatted_prompt},
            ])
            return enhanced_chunk

        new_chunks = await asyncio.gather(*[_chunk_enhancer(chunk) for chunk in chunks])  #todo we need to find ollama cloud's per minute request limit and

        return new_chunks

    async def process(self, url: str):
        """
        Full pipeline: crawl → chunk → enhance with context.

        Uses crawl result twice:
        1. For hybrid chunking with Docling (to create chunks)
        2. For contextual chunking (as the full document context)

        Args:
            url: URL to process

        Returns:
            List of enhanced chunks with contextual information
        """
        # Step 1: Crawl the URL
        crawled_result = await self.crawl(url)  #question = Does await waits for corresponded method's inside process right? But we already awaited inside the method, do we need to await again?

        # Step 2: Extract full document text for contextual chunking
        full_document_text = crawled_result.markdown

        # Step 3: Chunk the crawled result using hybrid chunking
        chunks = self.chunking(crawled_result)

        # Step 4: Convert chunks to strings (if they're not already strings)
        chunk_texts = [str(chunk) for chunk in chunks]

        # Step 5: Enhance chunks with contextual information
        enhanced_chunks = await self.contextual_chunking(
            chunks=chunk_texts,
            document=full_document_text
        )

        return enhanced_chunks

#TODO upload to the database, 


async def main():
    """
    Example usage of the pipeline.
    """
    pipeline = Pipeline()
    enhanced_chunks = await pipeline.process(url=url_to_crawl)

    print(f"Successfully processed {len(enhanced_chunks)} enhanced chunks")
    return enhanced_chunks


if __name__ == "__main__":
    asyncio.run(main())