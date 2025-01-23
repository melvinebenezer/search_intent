from selenium.webdriver.common.by import By
from base_page import BasePage

class ExamplePage(BasePage):
    # Locators
    SEARCH_INPUT = (By.ID, "search-input")
    SUBMIT_BUTTON = (By.CSS_SELECTOR, "button[type='submit']")
    NAVIGATION_MENU = (By.CLASS_NAME, "nav-menu")
    PRODUCT_ITEMS = (By.CLASS_NAME, "product-item")
    
    def __init__(self, driver):
        super().__init__(driver)
        self.url = "https://example.com"
    
    def navigate_to(self):
        self.driver.get(self.url)
    
    def search_for_item(self, search_term):
        self.input_text(*self.SEARCH_INPUT, search_term)
        self.click(*self.SUBMIT_BUTTON)
    
    def get_all_products(self):
        products = self.find_elements(*self.PRODUCT_ITEMS)
        return [product.text for product in products]
    
    def scroll_through_products(self):
        products = self.find_elements(*self.PRODUCT_ITEMS)
        for product in products:
            self.scroll_to_element(product) 