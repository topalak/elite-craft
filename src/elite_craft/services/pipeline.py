from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

import asyncio
from crawl4ai import AsyncWebCrawler#, BrowserConfig

from elite_craft.model_provider import ModelConfig
from config import settings

from crawling import crawl


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
    def chunking(crawled):
        doc = DocumentConverter().convert(source=crawled).document
        chunker = HybridChunker()
        chunk_iter = chunker.chunk(dl_doc=doc)

        return list(chunk_iter)


    def contextual_chunking(self, chunks:list[str], document:str):
        prompt = """ <document>
        {{ $('Extract Document Text').first().json.data }}
        </document>
        Here is the chunk we want to situate within the whole document
        <chunk>
        {{ $json.chunk }}
        </chunk>
        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else. """

        def _chunk_enhancer(chunk:str):
            """
            Uses llm to enhance the chunk.

            Args:
                chunk (str): The chunk to enhance
            """

            formatted_prompt = prompt.format(document=document, chunk=chunk)
            enhanced_chunk = self.cached_llm.invoke([
                {"role":"system", "content":formatted_prompt},
            ])
            return enhanced_chunk

        new_chunks = [_chunk_enhancer(chunk) for chunk in chunks]

        return new_chunks

    def process(self):

        # todo, use crawl method's result for 2 times, first one is for hybrid chunking with docling, second one is for contextual chunking
        asyncio.run(crawl(url="https://www.google.com/search?q=python"))
        print("Processing")


def main():
    run = Pipeline.process()
    print(run)