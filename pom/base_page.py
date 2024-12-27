from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from page_analyzer import PageAnalyzer

class BasePage:
    def __init__(self, driver):
        self.driver = driver
        self.wait = WebDriverWait(driver, 10)
        self.page_analyzer = PageAnalyzer()
    
    def find_element(self, by, value):
        return self.wait.until(EC.presence_of_element_located((by, value)))
    
    def find_elements(self, by, value):
        return self.wait.until(EC.presence_of_all_elements_located((by, value)))
    
    def click(self, by, value):
        element = self.find_element(by, value)
        element.click()
    
    def input_text(self, by, value, text):
        element = self.find_element(by, value)
        element.clear()
        element.send_keys(text)
    
    def scroll_to_element(self, element):
        self.driver.execute_script("arguments[0].scrollIntoView(true);", element)
    
    def scroll_to_bottom(self):
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    
    def is_element_visible(self, by, value, timeout=10):
        try:
            WebDriverWait(self.driver, timeout).until(
                EC.visibility_of_element_located((by, value))
            )
            return True
        except TimeoutException:
            return False 
    
    def analyze_current_page(self):
        """Analyzes the current page and returns suggested POM structure"""
        page_source = self.driver.page_source
        analysis = self.page_analyzer.analyze_page(page_source)
        return analysis
    
    def generate_page_object(self, output_file=None):
        """Generates a new page object class based on the current page"""
        analysis = self.analyze_current_page()
        
        # Generate Python code for the new page object
        code = self._generate_page_object_code(analysis)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(code)
        
        return code
    
    def _generate_page_object_code(self, analysis):
        """Generates Python code for a new page object based on analysis"""
        code = [
            "from selenium.webdriver.support.ui import WebDriverWait",
            "from selenium.webdriver.support import expected_conditions as EC",
            "from selenium.webdriver.common.by import By",
            "from base_page import BasePage\n",
            f"class {self.driver.title.replace(' ', '')}Page(BasePage):",
            "    # Locators"
        ]
        
        # Add locators with descriptions
        for name, locator_type, value, description in analysis.get('locators', []):
            code.append(f"    # {description}")
            code.append(f"    {name} = (By.{locator_type}, '{value}')")
        
        code.append("\n    def __init__(self, driver):")
        code.append("        super().__init__(driver)")
        code.append(f"        self.url = '{self.driver.current_url}'")
        
        # Add wait conditions
        code.append("\n    def wait_for_elements(self):")
        code.append("        \"\"\"Wait for critical elements to be ready\"\"\"")
        for wait in analysis.get('wait_conditions', []):
            element = wait['element']
            wait_type = wait['wait_type']
            timeout = wait.get('timeout', 10)
            
            if wait_type == 'presence':
                condition = 'presence_of_element_located'
            elif wait_type == 'visibility':
                condition = 'visibility_of_element_located'
            elif wait_type == 'clickable':
                condition = 'element_to_be_clickable'
                
            code.append(f"        self.wait.until(EC.{condition}(self.{element}), timeout={timeout})")
        
        # Add methods
        for method in analysis.get('methods', []):
            method_code = self._generate_method_code(method)
            code.extend(method_code)
        
        return "\n".join(code)
    
    def _generate_method_code(self, method):
        """Generates code for a single method"""
        code = [
            f"\n    def {method['name']}({method.get('parameters', 'self')}):",
            f"        \"\"\"{method['description']}\"\"\"",
        ]
        
        # Add method implementation if provided
        if 'implementation' in method:
            code.extend([f"        {line}" for line in method['implementation'].split('\n')])
        else:
            code.append("        pass")
        
        return code 