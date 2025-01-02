import os
import sys

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from pom.base_page import BasePage

def test_auto_pom_generation():
    # Setup driver
    chrome_options = Options()
    # Add options for headless mode if needed
    # chrome_options.add_argument('--headless')
    
    # Use ChromeDriverManager to handle driver installation
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    try:
        # Initialize base page
        base_page = BasePage(driver)
        
        # Navigate to target website
        driver.get("https://www.redbus.in")
        
        # Generate POM
        output_file = "generated_page.py"
        generated_code = base_page.generate_page_object(output_file)
        
        print("Generated Page Object Model:")
        print(generated_code)
        
        print(f"\nPOM has been saved to {output_file}")
        
    finally:
        driver.quit()

if __name__ == "__main__":
    test_auto_pom_generation() 