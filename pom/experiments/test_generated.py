from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from generated_page import RedbusPage

def test_redbus_booking():
    # Setup driver
    chrome_options = Options()
    # chrome_options.add_argument('--headless')  # Uncomment to run headless
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    try:
        # Initialize RedBus page
        redbus = RedbusPage(driver)
        
        # Perform search
        search_url = redbus.search_buses(
            from_city="Mumbai",
            to_city="Pune",
            date="2-Jan-2025"
        )
        
        print(f"Final URL: {search_url}")
        
        # Keep browser open for inspection
        input("Press Enter to close the browser...")
        
    finally:
        driver.quit()

if __name__ == "__main__":
    test_redbus_booking() 