from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from base_page import BasePage

class BuyCosmeticsProducts&BeautyProductsOnlineinIndiaatBestPrice|NykaaPage(BasePage):
    # Locators
    # Main search input field
    SEARCH_BOX = (By.ID, 'search-box')
    # Submit button
    SUBMIT_BUTTON = (By.CSS_SELECTOR, 'button[type='submit']')

    def __init__(self, driver):
        super().__init__(driver)
        self.url = 'https://www.nykaa.com/'

    def wait_for_elements(self):
        """Wait for critical elements to be ready"""
        self.wait.until(EC.visibility_of_element_located(self.SEARCH_BOX), timeout=10)

    def search_for_item(self, search_term):
        """Search for an item on the page"""
        self.input_text(*self.SEARCH_BOX, search_term)
        self.click(*self.SUBMIT_BUTTON)