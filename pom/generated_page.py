from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from base_page import BasePage

class BusTicketBookingOnlinemadeEasy,SecurewithTopBusOperators-redBusPage(BasePage):
    # Locators
    # Source city input field
    SOURCE_INPUT = (By.ID, 'src')

    def __init__(self, driver):
        super().__init__(driver)
        self.url = 'https://www.redbus.in/'

    def wait_for_elements(self):
        """Wait for critical elements to be ready"""
        self.wait.until(EC.presence_of_element_located(self.SOURCE_INPUT), timeout=10)

    def search_buses(['self', 'from_city', 'to_city', 'date']):
        """Search for buses between cities on a specific date"""
        self.set_source_city(from_city)
        self.set_destination_city(to_city)
        self.set_date(date)
        self.click_search()