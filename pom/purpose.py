import os
from dotenv import load_dotenv
import dspy
import asyncio
import nest_asyncio
import re
import json
import hashlib
from datetime import datetime
nest_asyncio.apply()

load_dotenv()


lm = dspy.LM(model="anthropic/claude-3-5-sonnet-20241022")
dspy.configure(lm=lm)

# Add these constants at the top of the file
CACHE_DIR = "cache"
LLM_CACHE_FILE = os.path.join(CACHE_DIR, "llm_cache.json")

def ensure_cache_dir():
    """Ensure cache directory exists"""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

def generate_cache_key(prompt_type, content):
    """Generate a unique cache key based on prompt type and content"""
    content_hash = hashlib.md5(content.encode()).hexdigest()
    return f"{prompt_type}_{content_hash}"

def load_llm_cache():
    """Load the LLM cache from file"""
    ensure_cache_dir()
    try:
        with open(LLM_CACHE_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_llm_cache(cache):
    """Save the LLM cache to file"""
    ensure_cache_dir()
    with open(LLM_CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

def cached_llm_call(prompt_type, content, llm_func, url):
    """
    Wrapper for LLM calls that implements caching
    prompt_type: string identifying the type of prompt (e.g., 'purpose', 'pom')
    content: the content being analyzed
    llm_func: the function that makes the actual LLM call
    """
    cache = load_llm_cache()
    cache_key = generate_cache_key(prompt_type, url)
    
    # Check if we have a cached response
    if cache_key in cache:
        print(f"Cache hit for {prompt_type}")
        return cache[cache_key]['response']
    
    # If not in cache, make the LLM call
    print(f"Cache miss for {prompt_type}, making LLM call...")
    response = llm_func()
    
    # Store in cache with timestamp
    cache[cache_key] = {
        'timestamp': datetime.now().isoformat(),
        'response': response
    }
    save_llm_cache(cache)
    
    return response

def get_purpose(html_content, url):
    chat = dspy.ChainOfThought('question->response:dict')

    def llm_call():
        prompt = f"""
        You are a helpful assistant that can help me understand the purpose of a website.
        I will give you the HTML content of the website and you will need to understand the purpose of the website.
        Please provide a short description of the purpose of the website.
        there can be multiple purposes, please list them all.
        these will be referred to as activities. 
        e,g the the website sells a beauty products, provides consultations, and offers training. etc... 
        the activities should be in the form of a list along with their starting url. 

        IMPORTANT: Respond with ONLY a JSON object in the following format:
        {{
            "activities": [
                {{"activity": "activity name", "url": "activity url"}},
                {{"activity": "activity name", "url": "activity url"}}
            ]
        }}
        
        HTML content:
        {html_content}
        """
        
        result = chat(question=prompt)
        
        # Convert the Prediction object to a dictionary
        if hasattr(result, 'response') and isinstance(result.response, dict):
            return {
                'activities': result.response.get('activities', [])
            }
        return None

    try:
        result = cached_llm_call('purpose', html_content, llm_call, url)
        if result and 'activities' in result:
            return result
        print("No valid activities found in response")
        return None
    except Exception as e:
        print(f"Error in get_purpose: {str(e)}")
        return None

import asyncio
from playwright.async_api import async_playwright

from crawl4ai import AsyncWebCrawler, CacheMode, BrowserConfig, CrawlerRunConfig, CacheMode
import random
import string

async def simple_crawl(url):
    crawler_run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    async with AsyncWebCrawler() as crawler:
        try:
            result = await crawler.arun(
                url=url,
                config=crawler_run_config
            )
            cleaned_html = remove_linksinhtml(result.html)
            purpose = get_purpose(cleaned_html, url)
            if purpose:
                activities = purpose.get('activities', None)
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

def get_input_elements(html_content, activity, url):
    pom = dspy.ChainOfThought('question->response:dict')
    
    def llm_call():
        prompt = f"""
        You are a helpful assistant that can help me understand the steps involved in {activity} on a website.
        I will give you the HTML content of a url and you will extract the page object model.
        pom or page object model is the input elements in a sequence. 
        
        For each step, identify:
        1. The input elements (textboxes, dropdowns, buttons, etc.)
        2. Whether user input is required before proceeding (step_lock)
        3. The URL or path for this step
        4. A description of what this step accomplishes
        5. A sample input for the input elements
        Return the steps in the following format:
        {{
            "pom": 
                {{
                    "description": "Description of what this step does",
                    "url": "URL or path for this step",
                    "step_lock": true/false (python bool) (whether user input is required),
                    "input_elements": [
                        {{
                            "id": "...",
                            "class": "...",
                            "type": "...",
                            "name": "...",
                            "placeholder": "...",
                            "value": "...",
                            "label": "...",
                            "description": "...",
                            "sample_input": "..."
                        }}
                    ]
                }}
            
        }}
        
        HTML content:
        {html_content}
        """
        
        result = pom(question=prompt)
        return result.response

    return cached_llm_call('pom', f"{activity}_{html_content}", llm_call, url)

async def explore_steps(activity_name, initial_step):
    """
    Automatically explores steps by filling in forms and following redirects to discover the complete flow
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # Set to True in production
        context = await browser.new_context()
        page = await context.new_page()
        
        all_steps = [initial_step]  # Start with the initial step
        current_step = 0
        
        while current_step < len(all_steps):
            step = all_steps[current_step]
            url = f"https://{step['url']}"  # Ensure URL has protocol
            
            try:
                # Navigate to the step's URL
                await page.goto(url, wait_until="networkidle")
                
                # Fill in the forms if step requires input
                if step['step_lock'] and step.get('input_elements'):
                    for element in step['input_elements']:
                        selector = None
                        
                        # Try different selectors in order of preference
                        if element.get('id'):
                            selector = f"#{element['id']}"
                        elif element.get('name'):
                            selector = f"[name='{element['name']}']"
                        elif element.get('class'):
                            selector = f".{element['class'].replace(' ', '.')}"
                            
                        if selector:
                            try:
                                # Skip buttons - they'll be handled later
                                if element['type'].lower() == 'button' or element['type'].lower() == 'submit':
                                    continue
                                    
                                # Handle different input types
                                elif element['type'] == 'select':
                                    await page.click(selector)
                                    await page.keyboard.press('ArrowDown')
                                    await page.keyboard.press('Enter')
                                elif element['type'] == 'date':
                                    value = element['sample_input']
                                    await page.fill(selector, value)
                                    await page.keyboard.press('Enter')
                                else:
                                    value = element['sample_input']
                                    await page.fill(selector, value)
                            except Exception as e:
                                print(f"Error filling element {selector}: {str(e)}")
                
                # Look for submit button in input_elements first
                submit_button = None
                for element in step.get('input_elements', []):
                    if element['type'].lower() in ['button', 'submit']:
                        selector = None
                        if element.get('id'):
                            selector = f"#{element['id']}"
                        elif element.get('name'):
                            selector = f"[name='{element['name']}']"
                        elif element.get('class'):
                            selector = f".{element['class'].replace(' ', '.')}"
                        
                        if selector:
                            try:
                                submit_button = await page.wait_for_selector(selector)
                                break
                            except Exception:
                                continue

                # If no submit button found in input_elements, try generic submit button search
                if not submit_button:
                    submit_button = await find_submit_button(page)

                if submit_button:
                    # Get current URL before clicking
                    previous_url = page.url
                    
                    # Click and wait for navigation
                    await submit_button.click()
                    await page.wait_for_load_state("networkidle")
                    
                    # If URL changed, we've found a new step
                    if page.url != previous_url:
                        # Extract POM for the new page
                        html_content = await page.content()
                        new_step_data = get_input_elements(html_content, activity_name)
                        
                        if new_step_data and isinstance(new_step_data, dict):
                            new_step = new_step_data
                            new_step['url'] = page.url.replace('https://', '')
                            new_step['step_number'] = len(all_steps) + 1
                            all_steps.append(new_step)
                
                current_step += 1
                
            except Exception as e:
                print(f"Error exploring step {current_step + 1}: {str(e)}")
                break
        
        await browser.close()
        return all_steps

def generate_test_input(element):
    """Generate appropriate test input based on element type"""
    element_type = element['type'].lower()
    
    if element_type == 'text':
        if 'email' in element.get('name', '').lower():
            return 'test@example.com'
        elif 'name' in element.get('name', '').lower():
            return 'Test User'
        else:
            return 'Test Input'
    
    elif element_type == 'number':
        if 'age' in element.get('name', '').lower():
            return '25'
        elif 'phone' in element.get('name', '').lower():
            return '1234567890'
        else:
            return '42'
    
    elif element_type == 'email':
        return 'test@example.com'
    
    elif element_type == 'date':
        from datetime import datetime, timedelta
        future_date = datetime.now() + timedelta(days=7)
        return future_date.strftime('%Y-%m-%d')
    
    return 'test'

async def find_submit_button(page):
    """Find the most likely submit/next button on the page"""
    button_selectors = [
        'button[type="submit"]',
        'input[type="submit"]',
        'button:has-text("Continue")',
        'button:has-text("Next")',
        'button:has-text("Submit")',
        'button:has-text("Search")',
        '[role="button"]:has-text("Continue")',
    ]
    
    for selector in button_selectors:
        try:
            button = await page.wait_for_selector(selector, timeout=2000)
            if button:
                return button
        except:
            continue
    
    return None

# Update the process_single_activity function to use explore_steps
async def process_single_activity(activity):
    crawler_run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    activities_map = {}
    activity_title = activity["activity"]
    url = activity["url"]
    print(f"Processing activity: {activity_title} at {url}")

    if not url.startswith('https://'):
        url = f"https://{url}"
    
    async with AsyncWebCrawler() as crawler:
        try:
            # Get initial page content and POM
            activity_result = await crawler.arun(
                url=url,
                config=crawler_run_config
            )
            cleaned_html = remove_linksinhtml(activity_result.html)
            initial_step_data = get_input_elements(cleaned_html[:-20000], activity_title, url)
            
            if initial_step_data and isinstance(initial_step_data, dict):
                # Explore subsequent steps starting with the initial POM
                all_steps = await explore_steps(activity_title, initial_step_data['pom'])
                activities_map[activity_title] = {'steps': all_steps}
            else:
                activities_map[activity_title] = {
                    'steps': [],
                    'message': f"Could not extract initial steps for {activity_title}"
                }
                
        except Exception as e:
            print(f"Error crawling {activity_title}: {str(e)}")
            activities_map[activity_title] = {
                'steps': [],
                'message': f"Error during crawl: {str(e)}"
            }
    
    return activities_map

async def test():
    activities = await simple_crawl('https://www.redbus.in/')
    print(activities)
    for activity in activities:
        activities_map = await process_single_activity(activity)
        print(activities_map)
        break

def save_activities_map(activities_map):
    with open('activities_map.json', 'w') as f:
        json.dump(activities_map, f)

def load_activities_map():
    with open('activities_map.json', 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    asyncio.run(test())




