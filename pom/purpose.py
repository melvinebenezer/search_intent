import os
from dotenv import load_dotenv
import dspy
import asyncio
import nest_asyncio
from bs4 import BeautifulSoup
import re
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

    IMPORTANT: Respond with ONLY a JSON object in the following format:
    
        "activities": [
            "activity": "activity name", "url": "activity url",
            "activity": "activity name", "url": "activity url"
        ]
      

    HTML content:
    {html_content}
    """
    
    try:
        result = chat(question=prompt)
        # Ensure we have the expected structure
        if isinstance(result.response, dict) and 'activities' in result.response:
            return result
        else:
            print("Unexpected response format:", result.response)
            return None
    except Exception as e:
        print(f"Error in get_purpose: {str(e)}")
        return None

import asyncio
from playwright.async_api import async_playwright

from crawl4ai import AsyncWebCrawler, CacheMode, BrowserConfig, CrawlerRunConfig, CacheMode

async def simple_crawl(url):
    crawler_run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    async with AsyncWebCrawler() as crawler:
        try:
            result = await crawler.arun(
                url=url,
                config=crawler_run_config
            )
            cleaned_html = remove_linksinhtml(result.html)
            purpose = get_purpose(cleaned_html)
            if purpose and hasattr(purpose, 'response') and isinstance(purpose.response, dict):
                activities = purpose.response.get('activities')
                if activities:
                    print("Initial crawl purpose:", activities)
                    return activities
            print("No valid activities found in response")
            return None
        except Exception as e:
            print(f"Error in simple_crawl: {str(e)}")
            return None

def remove_linksinhtml(html_content):
    # Remove https:// links and common URL patterns
    cleaned = html_content.replace("https://", "")
    cleaned = cleaned.replace("http://", "")
    # Remove common URL parameters
    cleaned = re.sub(r'\?[^)\s]*', '', cleaned)
    return cleaned

def get_input_elements(html_content, activity):
    pom = dspy.ChainOfThought('question->response:dict')
    prompt = f"""
    You are a helpful assistant that can help me understand the purpose of a website.
    I will give you the HTML content of a url and you will extract the page object model.
    that is the input elements like textboxes, dropdowns, buttons, etc.
    but related to the activity {activity}
    return the input elements in the form of a list of dictionaries with the following keys:
    - input_elements: list[dict[str, str]]
    each element should have the following keys: id, class, type, name, placeholder, value, label, description and any other relevant information that is available in the html.
    the input elements should be in the sequence of the purpose of the activity {activity}
    
    HTML content:
    {html_content}
    """
    
    result = pom(question=prompt)
    return result.response

async def get_activity_model(activity):
    crawler_run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    activities_map = {}
    activity_title = activity["activity"]
    url = activity["url"]
    print(f"Processing activity: {activity_title} at {url}")
    
    async with AsyncWebCrawler() as crawler:
        try:
            activity_result = await crawler.arun(
                url=url,
                config=crawler_run_config
            )
            print(f"Successfully crawled {activity_title}")
            cleaned_html = remove_linksinhtml(activity_result.html)
            input_elements = get_input_elements(cleaned_html, activity_title)
            # print(input_elements)
            activities_map[activity_title] = input_elements
        except Exception as e:
            print(f"Error crawling {activity_title}: {str(e)}")
    return activities_map

async def test():
    activities = await simple_crawl('https://www.redbus.in/')
    print(activities)
    for activity in activities:
        activities_map = await get_activity_model(activity)
        print(activities_map)
        break

if __name__ == "__main__":
    asyncio.run(test())




