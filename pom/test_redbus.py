from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from redbus_page import RedBusPage

def test_redbus_search():
    # Setup driver
    chrome_options = Options()
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    try:
        # Initialize RedBus page
        redbus_page = RedBusPage(driver)
        
        # Navigate to RedBus
        driver.get("https://www.redbus.in")
        
        # Wait for page elements
        redbus_page.wait_for_elements()
        
        # Perform search
        redbus_page.search_buses(
            from_city="Mumbai",
            to_city="Pune",
            date="2024-01-02"
        )
        
        # Add verification steps here
        
    finally:
        driver.quit()

if __name__ == "__main__":
    test_redbus_search() 