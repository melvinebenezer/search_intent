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
import wandb
import os

force_recompute = False
use_wandb = False
if use_wandb:
    wandb.init(project="bert-naive-bayes-intent-classification")

# Load and preprocess data
data_path = 'data/export_research_in_domain_history_usd_2024-08_nykaa.com.csv'
data = pd.read_csv('data/export_research_in_domain_history_usd_2024-08_nykaa.com.csv')

# Count the occurrences of each intent in the entire dataset
intent_counts = data['Search intent'].apply(lambda x: [label.strip() for label in x.split(',')]).explode().value_counts()

print("Intent counts in the entire dataset:")
print(intent_counts)

label_encoder = LabelEncoder()
data['labels'] = data['Search intent'].apply(lambda x: [label.strip() for label in x.split(',')])
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(data['labels'])
class_names = mlb.classes_
print(f"y shape: {y.shape}")
print(f"Classes: {class_names}")
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

# Store the embeddings in a file
embeddings_path = 'data/' + os.path.split(data_path)[-1] + '_embeddings.npy'

if not os.path.exists(embeddings_path) or force_recompute:
    # Extract BERT embeddings
    X = get_bert_embeddings(data['Keyword'].tolist())
    np.save(embeddings_path, X)
else:
    X = np.load(embeddings_path)

# X shape = [1000, 768]
# Y shape = [1000, 5]

# Split data
# Xtrain = [800, 768]
# Xtest = [ 200, 786]
X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(X, y,  data.index, test_size=0.2, random_state=42)
train_intents = data.iloc[train_idx]['Search intent'].apply(lambda x: [label.strip() for label in x.split(',')])
train_intent_counts = train_intents.explode().value_counts()

print("\n Intent counts in the training Dataset:")
print(train_intent_counts)


from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Train the OneVsRestClassifier
ovr_classifier = OneVsRestClassifier(LogisticRegression(solver='liblinear'))
ovr_classifier.fit(X_train, y_train)

# Predict and evaluate
threshold = 0.5
y_pred_probs = ovr_classifier.predict_proba(X_test)
y_pred = (y_pred_probs > threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))


from sklearn.ensemble import RandomForestClassifier

# Train a Random Forest Classifier
rf_classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
rf_classifier.fit(X_train, y_train)

# Predict and evaluate
threshold = 0.5
y_pred_probs = rf_classifier.predict_proba(X_test)
y_pred = (y_pred_probs > threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))


from sklearn.svm import SVC

# Train an SVM with linear kernel
svm_classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True))
svm_classifier.fit(X_train, y_train)

# Predict and evaluate
threshold = 0.5
y_pred_probs = svm_classifier.predict_proba(X_test)
y_pred = (y_pred_probs > threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))


from xgboost import XGBClassifier

# Train an XGBoost Classifier
xgb_classifier = OneVsRestClassifier(XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
xgb_classifier.fit(X_train, y_train)

# Predict and evaluate
y_pred_probs = xgb_classifier.predict_proba(X_test)
threshold = 0.5
y_pred = (y_pred_probs > threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))


from sklearn.neighbors import KNeighborsClassifier

# Train a KNN Classifier
knn_classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5))
knn_classifier.fit(X_train, y_train)

# Predict and evaluate
y_pred_probs = knn_classifier.predict_proba(X_test)
threshold = 0.5
y_pred = (y_pred_probs > threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))


import torch.nn as nn
import torch.optim as optim

class MultiLabelNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MultiLabelNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

model = MultiLabelNN(input_size=X_train.shape[1], num_classes=y_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    outputs = model(torch.tensor(X_train).float())
    loss = criterion(outputs, torch.tensor(y_train).float())
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch} Loss: {loss.item()}')

model.eval()
with torch.no_grad():
    y_pred_probs = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
    y_pred = (y_pred_probs > threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))