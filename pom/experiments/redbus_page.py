from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from base_page import BasePage

class RedBusPage(BasePage):
    # Locators
    SOURCE_INPUT = (By.ID, "src")
    DESTINATION_INPUT = (By.ID, "dest")
    DATE_PICKER = (By.ID, "onwardCal")
    SEARCH_BUTTON = (By.ID, "search_button")
    
    def __init__(self, driver):
        super().__init__(driver)
        self.url = "https://www.redbus.in"
    
    def set_source_city(self, city):
        """Set the source city in the From field"""
        self.input_text(*self.SOURCE_INPUT, city)
        # Wait for and select first suggestion if available
        try:
            suggestion = self.wait.until(EC.presence_of_element_located(
                (By.CSS_SELECTOR, "ul.autoFill li:first-child")
            ))
            suggestion.click()
        except:
            pass
    
    def set_destination_city(self, city):
        """Set the destination city in the To field"""
        self.input_text(*self.DESTINATION_INPUT, city)
        # Wait for and select first suggestion if available
        try:
            suggestion = self.wait.until(EC.presence_of_element_located(
                (By.CSS_SELECTOR, "ul.autoFill li:first-child")
            ))
            suggestion.click()
        except:
            pass
    
    def set_date(self, date):
        """Set the journey date"""
        # Click to open date picker
        self.click(*self.DATE_PICKER)
        # Select the date (implementation depends on the date picker structure)
        # This is a simplified version
        date_element = self.wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, f"td[data-date='{date}']")
        ))
        date_element.click()
    
    def click_search(self):
        """Click the search button"""
        self.click(*self.SEARCH_BUTTON)
    
    def search_buses(self, from_city, to_city, date):
        """Complete flow to search for buses"""
        self.set_source_city(from_city)
        self.set_destination_city(to_city)
        self.set_date(date)
        self.click_search()
        
    def wait_for_elements(self):
        """Wait for all critical elements to be ready"""
        self.wait.until(EC.presence_of_element_located(self.SOURCE_INPUT))
        self.wait.until(EC.presence_of_element_located(self.DESTINATION_INPUT))
        self.wait.until(EC.presence_of_element_located(self.DATE_PICKER))
        self.wait.until(EC.element_to_be_clickable(self.SEARCH_BUTTON)) 