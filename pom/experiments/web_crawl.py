import asyncio
from crawl4ai import *
from page_analyzer import PageAnalyzer
import re
import os
from crawl4ai.extraction_strategy import LLMExtractionStrategy

def remove_linksINMarkdown(markdown):
    # Remove https:// links and common URL patterns
    cleaned = markdown.replace("https://", "")
    cleaned = cleaned.replace("http://", "")
    # Remove common URL parameters
    cleaned = re.sub(r'\?[^)\s]*', '', cleaned)
    return cleaned

async def main():
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(
                url="https://www.redbus.in",
                # url="https://www.lovesaranghae.com/",
                max_depth=1  # Limit crawl depth to avoid too much data
            )
            
            # OpenAI
            # openai_strategy = LLMExtractionStrategy(provider="openai/gpt-4o", api_token=os.getenv("OPENAI_API_KEY"))
            # # openai_result = await openai_strategy.extract("https://www.redbus.in")
            # openai_result = await openai_strategy.extract("https://lovesaranghae.com/")
            # openai_result.save_to_file("openai_result.md")
            
            # Clean the markdown content
            cleaned_markdown = remove_linksINMarkdown(result.markdown)
            
            # Save both original and cleaned content
            with open("page_content.md", "w", encoding='utf-8') as f:
                # f.write("## Original Content\n\n")
                # f.write(result.markdown)
                f.write("\n\n## Cleaned Content\n\n")
                f.write(cleaned_markdown)
            
            # Analyze the cleaned content
            analyzer = PageAnalyzer()

            purpose = analyzer.analyse_page_purpose(cleaned_markdown)
            
            print("\n=== Analysis Results ===")
            print(f"Page Purpose: {purpose}")
            print(f"\nContent saved to: page_content.md")
            
    except Exception as e:
        print(f"Error during crawling: {e}")

if __name__ == "__main__":
    asyncio.run(main())