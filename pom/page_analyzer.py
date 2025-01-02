import requests
import json
from bs4 import BeautifulSoup
import re

class PageAnalyzer:
    def __init__(self, model_name="llama2"):
        self.model_name = model_name
        self.ollama_endpoint = "http://localhost:11434/api/generate"
        
    def analyze_page(self, html_content):
        # Parse HTML and extract important interactive elements
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract key interactive elements
        interactive_elements = self._extract_interactive_elements(soup)
        
        with open("formatted_html.html", "w") as file:
            file.write(str(interactive_elements))
            
        # Generate prompt for LLM with focused content
        prompt = self._create_prompt(interactive_elements)
        print(f"Prompt: {prompt}")
        
        # Get LLM analysis
        response = self._query_ollama(prompt)
        return self._parse_llm_response(response)
    
    def _extract_interactive_elements(self, soup):
        """Extract RedBus specific interactive elements"""
        elements = {
            'inputs': [],
            'buttons': []
        }
        
        # Find source/destination inputs
        src_input = soup.find('input', {'id': 'src'})
        if src_input:
            elements['inputs'].append({
                'type': 'text',
                'id': 'src',
                'placeholder': 'From',
                'container_class': 'sc-VigVT ishpWr',
                'label': 'From'
            })
        
        dest_input = soup.find('input', {'id': 'dest'})
        if dest_input:
            elements['inputs'].append({
                'type': 'text',
                'id': 'dest',
                'placeholder': 'To',
                'container_class': 'sc-VigVT ishpWr',
                'label': 'To'
            })
        
        # Find date picker
        date_picker = soup.find('div', {'id': 'onwardCal'})
        if date_picker:
            elements['inputs'].append({
                'type': 'date',
                'id': 'onwardCal',
                'container_class': 'sc-fjdhpX elUAqf',
                'label': 'Date'
            })
        
        # Find search button
        search_button = soup.find('button', {'id': 'search_button'})
        if search_button:
            elements['buttons'].append({
                'id': 'search_button',
                'text': 'SEARCH BUSES',
                'class': 'sc-cvbbAY gDXYez'
            })
        
        return elements
    
    def _create_prompt(self, interactive_elements):
        return f"""Create a RedBus Page Object Model (POM) for the search functionality.
        You must respond ONLY with a JSON object in the following format, no other text:
        
        {{
            "locators": [
                {{
                    "name": "SOURCE_INPUT",
                    "type": "ID",
                    "value": "src",
                    "description": "Source city input field"
                }}
            ],
            "methods": [
                {{
                    "name": "search_buses",
                    "description": "Search for buses between cities on a specific date",
                    "parameters": "self, from_city, to_city, date",
                    "implementation": "self.set_source_city(from_city)\\nself.set_destination_city(to_city)\\nself.set_date(date)\\nself.click_search()"
                }}
            ],
            "wait_conditions": [
                {{
                    "element": "SOURCE_INPUT",
                    "wait_type": "presence",
                    "timeout": 10
                }}
            ]
        }}

        Interactive Elements:
        {json.dumps(interactive_elements, indent=2)}
        
        Remember:
        1. Response must be ONLY valid JSON
        2. Include methods for:
           - Setting source city
           - Setting destination city
           - Setting date
           - Performing search
        3. Use reliable selectors (ID > data-attributes > CSS)
        4. Include appropriate wait conditions for each element
        """
    
    def _query_ollama(self, prompt):
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.1,  # Lower temperature for more consistent JSON
            "max_tokens": 2000
        }
        
        try:
            response = requests.post(self.ollama_endpoint, json=payload)
            response.raise_for_status()
            return response.json()['response']
        except requests.exceptions.RequestException as e:
            print(f"Error querying Ollama: {e}")
            return self._get_default_json()
    
    def _parse_llm_response(self, llm_response):
        try:
            print(f"LLM response: {llm_response}")
            # Try to extract JSON from the response if it's embedded in text
            json_match = re.search(r'\{[\s\S]*\}', llm_response)
            if json_match:
                json_str = json_match.group(0)
                pom_structure = json.loads(json_str)
            else:
                pom_structure = json.loads(llm_response)
            
            # Validate the structure
            if not self._validate_pom_structure(pom_structure):
                print("Invalid POM structure, using default")
                return self._get_default_json()
            
            print(f"Parsed POM structure: {pom_structure}")
            return pom_structure
            
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM response: {e}")
            print("Raw response:", llm_response)
            return self._get_default_json()
    
    def _validate_pom_structure(self, structure):
        """Validate that the POM structure has the required fields"""
        required_fields = ['locators', 'methods', 'wait_conditions']
        return all(field in structure for field in required_fields)
    
    def _get_default_json(self):
        """Return a default POM structure if parsing fails"""
        return {
            "locators": [
                {
                    "name": "SOURCE_INPUT",
                    "type": "ID",
                    "value": "src",
                    "description": "Source city input field"
                },
                {
                    "name": "DESTINATION_INPUT",
                    "type": "ID",
                    "value": "dest",
                    "description": "Destination city input field"
                },
                {
                    "name": "DATE_PICKER",
                    "type": "ID",
                    "value": "onwardCal",
                    "description": "Date picker element"
                },
                {
                    "name": "SEARCH_BUTTON",
                    "type": "ID",
                    "value": "search_button",
                    "description": "Search buses button"
                }
            ],
            "methods": [
                {
                    "name": "search_buses",
                    "description": "Search for buses between cities on a specific date",
                    "parameters": "self, from_city, to_city, date",
                    "implementation": "self.set_source_city(from_city)\nself.set_destination_city(to_city)\nself.set_date(date)\nself.click_search()"
                }
            ],
            "wait_conditions": [
                {
                    "element": "SOURCE_INPUT",
                    "wait_type": "visibility",
                    "timeout": 10
                }
            ]
        } 