import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from transformers import BertTokenizer, BertModel
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from tqdm import tqdm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

def get_bert_embeddings(text_list, tokenizer, model):
    """Generate BERT embeddings for a list of texts."""
    embeddings = []
    for text in tqdm(text_list, desc="Generating BERT embeddings"):
        inputs = tokenizer(text, return_tensors='pt', max_length=512, padding='max_length', truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(embedding)
    return np.array(embeddings)

def train_and_evaluate_models(X_train, X_test, y_train, y_test, class_names):
    """Train and evaluate multiple models."""
    models = [
        ("Logistic Regression", OneVsRestClassifier(LogisticRegression(solver='liblinear'))),
        ("Random Forest", OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))),
        ("SVM", OneVsRestClassifier(SVC(kernel='linear', probability=True))),
        ("XGBoost", OneVsRestClassifier(XGBClassifier(use_label_encoder=False, eval_metric='logloss'))),
        ("KNN", OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5))),
    ]

    results = []
    for name, model in models:
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        print(f"Evaluating {name}...")
        y_pred_probs = model.predict_proba(X_test)
        threshold = 0.5
        y_pred = (y_pred_probs > threshold).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": report['weighted avg']['precision'],
            "Recall": report['weighted avg']['recall'],
            "F1-Score": report['weighted avg']['f1-score']
        })
    
    return results

def main(data_path):
    """Main function to run the classification pipeline."""
    # Load data
    print("Loading data...")
    data = pd.read_csv(data_path)

    # Preprocessing
    print("Preprocessing data...")
    data['labels'] = data['Search intent'].apply(lambda x: [label.strip() for label in x.split(',')])
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(data['labels'])
    class_names = mlb.classes_

    # Initialize BERT
    print("Initializing BERT...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Get BERT embeddings
    print("Generating embeddings...")
    X = get_bert_embeddings(data['Keyword'].tolist(), tokenizer, model)

    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test, class_names)

    # Display results
    print("\nModel Results:")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    return results_df

if __name__ == "__main__":
    # Specify your data path here
    data_path = "data/keywords_export_20241126.csv"
    results = main(data_path)

    # Save results to an excel file
    results.to_excel("model_results.xlsx", index=False)

