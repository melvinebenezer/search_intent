import random
from typing import Literal
from dspy.datasets import DataLoader
from datasets import load_dataset
import dspy
import pandas as pd
import pickle
import os

# Load the Banking77 dataset.
CLASSES = load_dataset("PolyAI/banking77", split="train").features['label'].names
kwargs = dict(fields=("text", "label"), input_keys=("text",), split="train")

# Load the first 2000 examples from the dataset, and assign a hint to each *training* example.
trainset = [
    dspy.Example(x, hint=CLASSES[x.label], label=CLASSES[x.label]).with_inputs("text", "hint")
    for x in DataLoader().from_huggingface(dataset_name="PolyAI/banking77", **kwargs)[:2000]
]
random.Random(0).shuffle(trainset)
print("Banking77 dataset size:", len(trainset))
trainset = trainset[:100]

# New code for Myntra dataset
INTENT_CLASSES = ['L', 'C', 'N', 'T']  # Your classification classes

# Load the Myntra dataset and clean column names
myntra_df = pd.read_csv('data/test/export_research_in_domain_history_usd_2024-12_myntra.com_with_predictions.csv')
myntra_df.columns = myntra_df.columns.str.strip()  # Remove whitespace from column names

# Create DSPy examples from the Myntra dataset
myntra_trainset = [
    dspy.Example(
        text=row['Keyword'].strip(),
        hint=row['Search intent'].strip(),
        label=row['Search intent'].strip()
    ).with_inputs("text", "hint")
    for _, row in myntra_df.iterrows()
]

myntra_trainset = myntra_trainset[:100]
random.Random(0).shuffle(myntra_trainset)
print("Myntra dataset size:", len(myntra_trainset))

dspy.settings.experimental = True
# dspy.configure(lm=dspy.LM('gpt-4o-mini-2024-07-18'))
dspy.configure(lm=dspy.LM('ollama/llama3.2'))

# Define the DSPy module for classification
signature = dspy.Signature("text -> label").with_updated_fields('label', type_=Literal[tuple(CLASSES)])
classify = dspy.ChainOfThoughtWithHint(signature)

MODEL_PATH = 'myntra_optimized_classifier.pkl'

if os.path.exists(MODEL_PATH):
    # Load the existing optimized classifier
    print("Loading existing optimized classifier...")
    with open(MODEL_PATH, 'rb') as f:
        optimized_classifier = pickle.load(f)
else:
    # Train a new classifier
    print("Training new Myntra classifier...")
    optimizer = dspy.BootstrapFinetune(metric=(lambda x, y, trace=None: x.label == y.label), num_threads=24)
    optimized_classifier = optimizer.compile(classify, trainset=myntra_trainset)
    
    # Save the optimized classifier
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(optimized_classifier, f)

# Test the classifier
# result = optimized_classifier(text="What does a pending cash withdrawal mean?")
# print(result)

# Test Myntra dataset
""
result = optimized_classifier(text="shoes for women")
print(result)
