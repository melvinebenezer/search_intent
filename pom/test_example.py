from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from example_page import ExamplePage

def setup_driver():
    chrome_options = Options()
    # Add options as needed
    # chrome_options.add_argument('--headless')
    service = Service('path_to_chromedriver')  # Update path as needed
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def test_example_page():
    driver = setup_driver()
    try:
        # Initialize the page object
        example_page = ExamplePage(driver)
        
        # Navigate to the website
        example_page.navigate_to()
        
        # Perform some actions
        example_page.search_for_item("test product")
        
        # Scroll through products
        example_page.scroll_through_products()
        
        # Get all products
        products = example_page.get_all_products()
        print("Found products:", products)
        
    finally:
        driver.quit()

if __name__ == "__main__":
    test_example_page() 