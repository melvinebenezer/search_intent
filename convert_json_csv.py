import json
import csv
from datetime import datetime

def convert_json_to_csv(input_file, output_file):
    # Read JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Define CSV headers
    headers = [
        'Keyword', 'Difficulty', 'Position', 'Previous position', 
        'Position Serp Features', 'Search vol.', 'Search intent',
        'SERP features', 'Competition', 'CPC', 'URL', 'Traffic',
        'Traffic share', 'Traffic cost'
    ]
    
    # Create CSV file and write headers
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
        # Process each keyword
        for item in data['keywords']:
            # Create row with default values
            row = {
                'Keyword': item['kw'],
                'Difficulty': '',  # Empty as not provided in input
                'Position': '',    # Empty as not provided in input
                'Previous position': '',
                'Position Serp Features': '',
                'Search vol.': '',
                'Search intent': ','.join(item['intent']),  # Join intent array with commas
                'SERP features': '',
                'Competition': '',
                'CPC': '',
                'URL': '',
                'Traffic': '',
                'Traffic share': '',
                'Traffic cost': ''
            }
            
            writer.writerow(row)

# Example usage
if __name__ == "__main__":
    input_file = "data/kws.json"  # Your input JSON file
    output_file = f"data/keywords_export_{datetime.now().strftime('%Y%m%d')}.csv"
    
    try:
        convert_json_to_csv(input_file, output_file)
        print(f"Conversion completed successfully. Output saved to {output_file}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")