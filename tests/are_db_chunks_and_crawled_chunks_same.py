import unittest
from urllib import response

from elite_craft.services.chunking import Chunker
from elite_craft.services.crawling import crawl
import asyncio
from supabase import create_client, Client
from src.config import settings

class ChunkerTest:

    def __init__ (self):
        self.db_client: Client = create_client(supabase_url=settings.SUPABASE_URL.get_secret_value(), supabase_key=settings.SUPABASE_SERVICE_ROLE_SECRET_KEY.get_secret_value())

    def fetch_chunks_from_document_id(self, document_id:int):

        response = (
            self.db_client.table(table_name="chunks")
            .select("*")
            .eq("document_id", document_id)
            .order("chunk_id_in_document", desc=False)
            .execute()
            )

        content_list = [chunk['content'] for chunk in response.data]

        # Format chunks with separators
        #chunks_text = "\n\n---\n\n".join(content_list)

        return content_list

    def list_all_document_ids(self,):

        response = (
            self.db_client.table("documents")
            .select("id")
            .order("id", desc=False)
            .execute()
        )

        document_ids = [item.get('id') for item in response.data]

        return document_ids

    @staticmethod
    async def crawl(url:str,):

        response = await crawl(url=url)
        chunker = Chunker()
        chunks = chunker.chunk(content=response['body_text'], url=url)

        return chunks

async def main():
    """Main async function to test chunking against database."""
    chunker_tester = ChunkerTest()
    document_ids = chunker_tester.list_all_document_ids()

    print(document_ids)

    for document_id in document_ids:
        response = (
            chunker_tester.db_client.table(table_name="documents")
            .select("url")
            .eq("id", document_id)
            .execute()
        )

        url = response.data[0]['url']

        crawled_chunks = await\
            chunker_tester.crawl(url=url)
        db_chunks = chunker_tester.fetch_chunks_from_document_id(document_id=document_id)

        for i, crawled_chunk in enumerate(crawled_chunks):
            database_chunk = db_chunks[i]
            if crawled_chunk != database_chunk:
                print(f'DOCUMENT ID: {document_id}')
                print(f"{60*'='}")
                print(f'CRAWLED CHUNK: {crawled_chunk}:')
                print(f"{60*'='}")
                print(f"DATABASE CHUNK: {database_chunk}")

if __name__ == '__main__':
    # Run the async main function using event loop
    asyncio.run(main())
