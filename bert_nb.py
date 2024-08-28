import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel
import torch
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import tqdm
import wandb

use_wandb = False
if use_wandb:
    wandb.init(project="bert-naive-bayes-intent-classification")

# Load and preprocess data
data = pd.read_csv('data/export_research_in_domain_history_usd_2024-08_nykaa.com.csv')

label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['Search intent'])

print(data.head())

# Initialize BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Get BERT embeddings
def get_bert_embeddings(text_list):
    embeddings = []
    for text in tqdm.tqdm(text_list):
        inputs = tokenizer(text, return_tensors='pt', max_length=512, padding='max_length', truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(embedding)
    return np.array(embeddings)

# Extract BERT embeddings
X = get_bert_embeddings(data['Keyword'].tolist())
y = data['label'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Gaussian Naive Bayes
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Predict and evaluate
y_pred = nb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

if use_wandb:
    wandb.log({"accuracy": accuracy})
    wandb.sklearn.plot_classifier(nb_classifier, X_train, X_test, y_train, y_test, y_pred, label_encoder.classes_, model_name="Gaussian Naive Bayes")

    # Finish the WandB run
    wandb.finish()

# # Optional: Save the model
# import joblib
# joblib.dump(nb_classifier, 'naive_bayes_model.joblib')