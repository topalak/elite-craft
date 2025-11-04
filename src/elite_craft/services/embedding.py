"""
This file split the document into chunks and embed them to database
There are multiple techniques for improve RAG performance which are Contextual Retrieval (using prompt caching), reranking
"""

import docling

from elite_craft.model_provider import ModelConfig
from config import settings

from crawling import crawl

url_to_crawl = "https://docs.langchain.com/oss/python/langgraph/overview"

embedding_model_config = ModelConfig(model="embeddinggemma", model_provider_url=settings.OLLAMA_HOST_MY_LOCAL)

class Embedder:
    def __init__(self):
        self.embedding_model = embedding_model_config.get_embedding()


    #def

    def contextual_retrieval(self):
        prompt = """ <document>
        {{ $('Extract Document Text').first().json.data }}
        </document>
        Here is the chunk we want to situate within the whole document
        <chunk>
        {{ $json.chunk }}
        </chunk>
        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else. """

        # todo, handle that by using list comprehension

