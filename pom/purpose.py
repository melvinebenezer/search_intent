import os
from dotenv import load_dotenv
import dspy
import asyncio
import nest_asyncio
nest_asyncio.apply()

load_dotenv()


lm = dspy.LM(model="anthropic/claude-3-5-sonnet-20241022")
dspy.configure(lm=lm)

def get_purpose(html_content):
    chat = dspy.ChainOfThought('question->response:dict')

    prompt = f"""
    You are a helpful assistant that can help me understand the purpose of a website.
    I will give you the HTML content of the website and you will need to understand the purpose of the website.
    Please provide a short description of the purpose of the website.
    there can be multiple purposes, please list them all.
    these will be referred to as activities. 
    e,g the the website sells a beauty products, provides consultations, and offers training. etc... 
    the activities should be in the form of a list along with their starting url. 
    the output should be a dictionary with the following keys:
    - activities: list[dict[str, str]]
    
    for the following html content:
    {html_content}
    """
    result = chat(question=prompt)

    return result

import asyncio
from playwright.async_api import async_playwright

from crawl4ai import AsyncWebCrawler, CacheMode, BrowserConfig, CrawlerRunConfig, CacheMode

async def simple_crawl():
    crawler_run_config = CrawlerRunConfig( cache_mode=CacheMode.BYPASS)
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            args=[
                '--disable-http2',
                '--no-sandbox',
                '--disable-dev-shm-usage'
            ]
        )
        page = await browser.new_page()
        try:
            await page.goto('https://www.redbus.in/', timeout=30000)
            print(f'Title: {await page.title()}')
        except Exception as e:
            print(f"Error accessing the website: {str(e)}")
        finally:
            await browser.close()

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            # url="https://www.kidocode.com/degrees/technology",
            url="https://www.redbus.in/",
            config=crawler_run_config
        )
        # print(result.links)
        purpose = get_purpose(result.links)
        print(purpose)
        return purpose.response['activities']
        # print(result.markdown_v2.raw_markdown[:500].replace("\n", " -- "))  # Print the first 500 characters


async def test():
    activities = await simple_crawl()
    print(activities)

if __name__ == "__main__":
    asyncio.run(test())




