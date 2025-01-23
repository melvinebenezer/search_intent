from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from generated_page import RedbusPage  # Adjust class name as needed

# Setup driver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# Use the generated page object
page = RedbusPage(driver)
driver.get("https://www.redbus.in")
page.wait_for_elements()

# Perform the search
page.search_buses("Mumbai", "Pune", "2024-01-02")

# Keep browser open for inspection
input("Press Enter to close...")
driver.quit() 