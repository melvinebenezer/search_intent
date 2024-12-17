import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer, BertModel
import torch
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import f1_score, classification_report, make_scorer
import numpy as np
from tqdm import tqdm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
import os
import hashlib
import pickle

def get_cache_path(text_list, cache_dir="embeddings_cache"):
    """Generate a unique cache path based on the input text."""
    # Create cache directory if it doesn't exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # Create a hash of the text content to use as filename
    text_concat = "".join(text_list)
    filename = hashlib.md5(text_concat.encode()).hexdigest() + ".pkl"
    return os.path.join(cache_dir, filename)

def get_bert_embeddings(text_list, tokenizer, model, cache_dir="embeddings_cache"):
    """Generate BERT embeddings for a list of texts with caching."""
    cache_path = get_cache_path(text_list, cache_dir)
    
    # Try to load from cache first
    if os.path.exists(cache_path):
        print("Loading embeddings from cache...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    print("Generating new embeddings...")
    embeddings = []
    for text in tqdm(text_list, desc="Generating BERT embeddings"):
        inputs = tokenizer(text, return_tensors='pt', max_length=512, padding='max_length', truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(embedding)
    
    embeddings = np.array(embeddings)
    
    # Save to cache
    print("Saving embeddings to cache...")
    with open(cache_path, 'wb') as f:
        pickle.dump(embeddings, f)
    
    return embeddings

def train_with_cv_and_evaluate(X_train, X_test, y_train, y_test, class_names):
    """Train with cross-validation using F1 score and evaluate models."""
    # Create F1 scorer
    f1_scorer = make_scorer(f1_score, average='weighted')
    
    models = [
        ("Logistic Regression", OneVsRestClassifier(LogisticRegression(solver='liblinear', class_weight='balanced'))),
        # ("Random Forest", OneVsRestClassifier(RandomForestClassifier(n_estimators=100, 
        #                                                             random_state=42, 
        #                                                             class_weight='balanced'))),
        # ("SVM", OneVsRestClassifier(SVC(kernel='linear', 
        #                               probability=True, 
        #                               class_weight='balanced'))),
        # ("XGBoost", OneVsRestClassifier(XGBClassifier(eval_metric='logloss',
        #                                              objective='binary:logistic',
        #                                              scale_pos_weight=1))),
        ("KNN", OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5,
                                                        weights='distance')))
    ]

    results = []
    detailed_cv_results = {}
    
    for name, model in models:
        print(f"\nTraining {name} with cross-validation...")
        
        # Perform cross-validation
        cv_scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=5,
            scoring={
                'f1': f1_scorer,
                'precision': 'precision_weighted',
                'recall': 'recall_weighted'
            },
            return_train_score=True
        )
        
        detailed_cv_results[name] = {
            'CV F1 (mean)': cv_scores['test_f1'].mean(),
            'CV F1 (std)': cv_scores['test_f1'].std(),
            'CV Precision (mean)': cv_scores['test_precision'].mean(),
            'CV Recall (mean)': cv_scores['test_recall'].mean()
        }
        
        # Train on full training set
        print(f"Training final {name} model...")
        model.fit(X_train, y_train)
        
        # Save the trained model
        model_path = f"models/{name.lower().replace(' ', '_')}.pkl"
        os.makedirs('models', exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Predict with probability threshold optimization
        print(f"Evaluating {name}...")
        y_pred_probs = model.predict_proba(X_test)
        
        # Find optimal threshold using F1 score
        best_threshold = 0.5
        best_f1 = 0
        for threshold in np.arange(0.3, 0.7, 0.05):
            y_pred = (y_pred_probs > threshold).astype(int)
            current_f1 = f1_score(y_test, y_pred, average='weighted')
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = threshold
        
        # Final prediction with optimal threshold
        y_pred = (y_pred_probs > best_threshold).astype(int)
        
        # Calculate metrics
        test_f1 = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        results.append({
            "Model": name,
            "Best Threshold": best_threshold,
            "Test F1-Score": test_f1,
            "CV F1-Score": detailed_cv_results[name]['CV F1 (mean)'],
            "CV F1 Std": detailed_cv_results[name]['CV F1 (std)'],
            "Test Precision": report['weighted avg']['precision'],
            "Test Recall": report['weighted avg']['recall']
        })
    
    return results, detailed_cv_results

def main(data_path, cache_dir="embeddings_cache"):
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

    # Save the MultiLabelBinarizer for future use
    os.makedirs('models', exist_ok=True)
    with open('models/label_binarizer.pkl', 'wb') as f:
        pickle.dump(mlb, f)

    # Initialize BERT
    print("Initializing BERT...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Get BERT embeddings with caching
    X = get_bert_embeddings(data['Keyword'].tolist(), tokenizer, model, cache_dir)

    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y.sum(axis=1))

    # Train and evaluate models
    results, cv_results = train_with_cv_and_evaluate(X_train, X_test, y_train, y_test, class_names)

    # Display results
    print("\nModel Results:")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Display detailed CV results
    print("\nDetailed Cross-Validation Results:")
    cv_results_df = pd.DataFrame(cv_results).T
    print(cv_results_df)
    
    return results_df, cv_results_df

def flexible_evaluate(y_true, y_pred, mlb):
    """
    Evaluate predictions with flexible matching criteria including precision and recall.
    If true label is 'L,C', accepts predictions containing either L or C or T.
    """
    def is_match(true_labels, pred_labels):
        # If true labels contain both L and C
        if 'L' in true_labels and 'C' in true_labels:
            # Check if prediction contains any of L, C, or T
            return any(label in pred_labels for label in ['L', 'C', 'T'])
        # For other cases, use exact matching
        return set(true_labels) == set(pred_labels)

    # Convert binary matrices back to label sets
    true_labels = [set(mlb.inverse_transform(y_true)[i]) for i in range(len(y_true))]
    pred_labels = [set(mlb.inverse_transform(y_pred)[i]) for i in range(len(y_pred))]
    
    # Calculate metrics
    total = len(true_labels)
    matches = 0
    true_lc_count = 0  # Count of true L,C cases
    pred_lct_count = 0  # Count of predicted L/C/T cases
    correct_lc_matches = 0  # Correct predictions for L,C cases
    
    for t, p in zip(true_labels, pred_labels):
        # Count total matches
        if is_match(t, p):
            matches += 1
        
        # Count L,C specific cases
        if 'L' in t and 'C' in t:
            true_lc_count += 1
            if any(label in p for label in ['L', 'C', 'T']):
                correct_lc_matches += 1
        
        if any(label in p for label in ['L', 'C', 'T']):
            pred_lct_count += 1
    
    accuracy = matches / total
    
    # Calculate precision and recall for L,C cases
    precision = correct_lc_matches / pred_lct_count if pred_lct_count > 0 else 0
    recall = correct_lc_matches / true_lc_count if true_lc_count > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Create detailed report
    report = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score,
        'Total Samples': total,
        'Correct Predictions': matches,
        'True L,C Cases': true_lc_count,
        'Predicted L/C/T Cases': pred_lct_count,
        'Correct L,C Matches': correct_lc_matches,
        'Flexible Matching Used': "Yes - L,C matches with L/C/T"
    }
    
    return report

def load_and_evaluate(test_data_path):
    """
    Load saved models and evaluate on test data with flexible matching.
    """
    # Load the label binarizer
    with open('models/label_binarizer.pkl', 'rb') as f:
        mlb = pickle.load(f)
    
    # Load test data
    data = pd.read_csv(test_data_path)
    data['labels'] = data['Search intent'].apply(lambda x: [label.strip() for label in x.split(',')])
    y_true = mlb.transform(data['labels'])
    
    # Initialize BERT for embeddings
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    X_test = get_bert_embeddings(data['Keyword'].tolist(), tokenizer, model)
    
    results = []
    
    # Load and evaluate each saved model
    for model_file in os.listdir('models'):
        if model_file.endswith('.pkl') and model_file != 'label_binarizer.pkl':
            model_name = model_file.replace('.pkl', '').replace('_', ' ').title()
            
            # Load the model
            with open(f'models/{model_file}', 'rb') as f:
                clf = pickle.load(f)
            
            # Get predictions
            y_pred = clf.predict(X_test)
            
            # Evaluate with flexible matching
            eval_report = flexible_evaluate(y_true, y_pred, mlb)
            eval_report['Model'] = model_name
            results.append(eval_report)
    
    # Create DataFrame with results
    results_df = pd.DataFrame(results)
    return results_df

if __name__ == "__main__":
    data_path = "data/keywords_export_20241126.csv"
    
    # Train models if not already trained
    if not os.path.exists('models'):
        print("Training models...")
        results_df, cv_results_df = main(data_path)
    
    # Evaluate with flexible matching
    print("\nEvaluating models with flexible matching criteria...")
    flexible_results = load_and_evaluate(data_path)
    print("\nFlexible Evaluation Results:")
    print(flexible_results.to_string(index=False))
    
    # Save results
    os.makedirs('results', exist_ok=True)
    flexible_results.to_excel("results/flexible_evaluation_results.xlsx", index=False)