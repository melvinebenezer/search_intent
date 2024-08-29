import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel
import torch
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import tqdm
import os
import streamlit as st
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
import torch.nn as nn
import torch.optim as optim

# ... keep your existing model and preprocessing code ...

# Streamlit UI
st.title("BERT-based Multi-Label Classification")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data_path = uploaded_file.name
    st.write(f"Selected file: {data_path}")

    if st.button("Run Models"):
        data = pd.read_csv(uploaded_file)

        # Preprocessing
        label_encoder = LabelEncoder()
        data['labels'] = data['Search intent'].apply(lambda x: [label.strip() for label in x.split(',')])
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(data['labels'])
        class_names = mlb.classes_

        # Initialize BERT
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        # Get BERT embeddings
        @st.cache(allow_output_mutation=True)
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

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define models
        models = [
            ("Logistic Regression", OneVsRestClassifier(LogisticRegression(solver='liblinear'))),
            ("Random Forest", OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))),
            ("SVM", OneVsRestClassifier(SVC(kernel='linear', probability=True))),
            ("XGBoost", OneVsRestClassifier(XGBClassifier(use_label_encoder=False, eval_metric='logloss'))),
            ("KNN", OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5))),
        ]

        results = []
        for name, model in models:
            model.fit(X_train, y_train)
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

        # Display results
        st.write("## Results")
        st.table(pd.DataFrame(results))

if __name__ == "__main__":
    # The Streamlit app is already running, so we don't need to call any run() method
    pass