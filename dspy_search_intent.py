import dspy
from typing import List, Literal
from dataclasses import dataclass
import logging
import json
import os
import pandas as pd
from dotenv import load_dotenv

# Enable experimental features
dspy.settings.experimental = True

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('intent_classifier.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define the possible intent classes
INTENT_CLASSES = ['L', 'C', 'T', 'N']

@dataclass
class KeywordIntent:
    keyword: str
    intent: str
    confidence: float

class IntentClassifier(dspy.Signature):
    """Classify search intent for keywords."""
    keyword: str = dspy.InputField()
    intent: Literal['L', 'C', 'T', 'N'] = dspy.OutputField(desc="Intent classification (L/C/T/N)")
    explanation: str = dspy.OutputField()
    confidence: float = dspy.OutputField()

class BulkIntentClassifierManager:
    def __init__(self, model="gpt-4", api_key=None):
        # Initialize DSPy with OpenAI
        self.lm = dspy.LM(model=f"openai/{model}", api_key=api_key)
        self.lm = dspy.LM('ollama/llama3.2')
        dspy.configure(lm=self.lm)
        
        # Create the basic classifier with proper signature format
        class IntentSignature(dspy.Signature):
            """Signature for intent classification"""
            text = dspy.InputField()
            intents = dspy.OutputField(desc="List of intents (L/C/T/N)")
            explanations = dspy.OutputField()
            confidences = dspy.OutputField()
        
        self.predictor = dspy.Predict(IntentSignature)
        
        # Set up the prompt template directly
        self.predictor.prompt = """
        Classify each search query into one of these categories:
        - C: Commercial/Shopping intent (e.g., "buy shoes", "dresses for women", "mens shirts")
           * Product searches
           * Shopping queries
           * Brand searches with purchase intent
        
        - I: Informational intent (e.g., "how to style lehenga", "shoe size chart")
           * How-to queries
           * Product information
           * General information
        
        - N: Navigational intent (e.g., "myntra login", "myntra app download")
           * Website/app specific queries
           * Login/account related
           * Specific page navigation
        
        Important rules:
        1. Most product searches (clothing, shoes, accessories) are Commercial (C)
        2. Brand names in shopping context are Commercial (C)
        3. Login/account queries are Navigational (N)
        4. Product information queries are Informational (I)

        Examples:
        Query: "lehenga for wedding"
        Intent: C (Commercial - shopping for wedding lehenga)
        
        Query: "myntra customer care"
        Intent: N (Navigational - seeking specific service page)
        
        Query: "how to measure shoe size"
        Intent: I (Informational - seeking product information)
        
        Query: "mens formal shirts"
        Intent: C (Commercial - shopping for clothing)
        
        Query: "nike shoes"
        Intent: C (Commercial - brand-specific product search)

        Queries to classify:
        {{text}}

        Return your response as a JSON object with three lists:
        - intents: list of intent classifications (C/I/N only)
        - confidences: list of confidence scores (0-1)
        - explanations: list of brief explanations
        """
    
    def classify_bulk(self, keywords: List[str], batch_size: int = 10) -> List[KeywordIntent]:
        """Classify multiple keywords in batches."""
        results = []
        
        # Process keywords in batches
        for i in range(0, len(keywords), batch_size):
            batch = keywords[i:i + batch_size]
            try:
                logger.info(f"Processing batch of {len(batch)} keywords")
                
                # Format the input text with numbered queries
                input_text = "\n".join(f"{idx+1}. {keyword}" 
                                     for idx, keyword in enumerate(batch))
                
                # Process entire batch in one API call
                result = self.predictor(text=input_text)
                
                logger.info(f"Received result: {result}")
                
                # Parse comma-separated values
                try:
                    intents = [i.strip() for i in result.intents.split(',')]
                    confidences = [float(c.strip()) for c in result.confidences.split(',')]
                    
                    # Process each result
                    for keyword, intent, confidence in zip(batch, intents, confidences):
                        results.append(KeywordIntent(
                            keyword=keyword,
                            intent=intent,
                            confidence=confidence
                        ))
                except Exception as e:
                    logger.error(f"Error parsing result: {str(e)}")
                    # Fallback for single result
                    confidence = 0.9 if str(result.confidences).lower() == 'high' else 0.5
                    results.append(KeywordIntent(
                        keyword=batch[0],
                        intent=result.intents,
                        confidence=confidence
                    ))
                    
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                # Handle failed batch by marking all keywords as errors
                for keyword in batch:
                    results.append(KeywordIntent(
                        keyword=keyword,
                        intent="ERROR",
                        confidence=0.0
                    ))
        
        return results

def main():
    # Read keywords from CSV
    csv_path = "data/test/export_research_in_domain_history_usd_2024-12_myntra.com_with_predictions.csv"
    df = pd.read_csv(csv_path)
    
    # Print column names to debug
    print("Available columns:", df.columns.tolist())
    
    # The column might have whitespace or different capitalization
    keyword_column = next(col for col in df.columns if col.lower().strip() == 'keyword')
    search_intent_column = next(col for col in df.columns if col.lower().strip() == 'search intent')
    
    # Get first 50 keywords from the keyword column
    keywords = df[keyword_column].head(50).fillna('').tolist()
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    # Initialize classifier
    classifier = BulkIntentClassifierManager(model="gpt-4o-mini", api_key=api_key)
    
    # Get classifications
    results = classifier.classify_bulk(keywords, batch_size=10)
    
    # Create results DataFrame with only the fields we want
    output_df = pd.DataFrame([
        {
            'Keyword': r.keyword,
            'Search Intent': df[df[keyword_column] == r.keyword][search_intent_column].iloc[0],
            'Predicted Intent': r.intent
        } for r in results
    ])
    
    # Save to CSV
    output_path = csv_path.replace('.csv', '_classified_simple.csv')
    output_df.to_csv(output_path, index=False)
    
    # Print results
    print("\nClassification Results:")
    print("-" * 50)
    for _, row in output_df.iterrows():
        print(f"Keyword: {row['Keyword']}")
        print(f"Search Intent: {row['Search Intent']}")
        print(f"Predicted Intent: {row['Predicted Intent']}")
        print("-" * 50)
    
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()