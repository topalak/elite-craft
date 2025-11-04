import asyncio
from crawl4ai import AsyncWebCrawler#, BrowserConfig




async def crawl(url:str):
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url,)
        print(result.markdown)


def main():
    asyncio.run(crawl(url="https://www.google.com/search?q=python"))
if __name__ == "__main__":
    main()


