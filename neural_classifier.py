import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
import numpy as np
from tqdm import tqdm
import os
import pickle
from transformers import BertTokenizer, BertModel
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import hashlib

class SearchIntentDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return x + self.block(x)  # Residual connection

class IntentClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3, num_residual_blocks=2):
        super(IntentClassifier, self).__init__()
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) 
            for _ in range(num_residual_blocks)
        ])
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Initial projection
        x = self.input_projection(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Output
        return self.output_layers(x)

def get_bert_embeddings(text_list, data_source=None, cache_dir="embeddings_cache"):
    """Generate BERT embeddings for a list of texts with caching."""
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache filename from data source
    if data_source:
        # Extract directory/file name and clean it for use as filename
        cache_name = os.path.basename(data_source.rstrip('/'))
        cache_filename = f"embeddings_{cache_name}.pkl"
    else:
        # Fallback to hash if no data source provided
        text_concat = "".join(text_list)
        cache_filename = hashlib.md5(text_concat.encode()).hexdigest() + ".pkl"
    
    cache_path = os.path.join(cache_dir, cache_filename)
    
    # Try to load from cache first
    if os.path.exists(cache_path):
        print(f"Loading embeddings from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    print(f"Generating new embeddings for {cache_filename}...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    embeddings = []
    for text in tqdm(text_list, desc="Generating BERT embeddings"):
        inputs = tokenizer(text, return_tensors='pt', max_length=512, padding='max_length', truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(embedding)
    
    embeddings = np.array(embeddings)
    
    # Save to cache
    print(f"Saving embeddings to cache: {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump(embeddings, f)
    
    return embeddings

class F1Loss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape
        
        y_pred = torch.sigmoid(y_pred)  # In case predictions are not sigmoid activated
        
        tp = (y_true * y_pred).sum(dim=0)
        fp = ((1 - y_true) * y_pred).sum(dim=0)
        fn = (y_true * (1 - y_pred)).sum(dim=0)
        
        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)
        
        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        return 1 - f1.mean()  # Return 1 - F1 since we're minimizing

class NeuralClassifier:
    def __init__(self, input_dim, hidden_dim=256, dropout=0.3, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.scaler = StandardScaler()
        self.model = None
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        print(f"Using device: {self.device}")

    def train(self, X_train, X_test, y_train, y_test, class_names, 
              batch_size=32, n_epochs=10, learning_rate=0.001, continue_training=False):
        """Train the neural network classifier."""
        # Print dataset sizes
        print("\nDataset Statistics:")
        print(f"Training samples: {len(X_train):,}")
        print(f"Validation samples: {len(X_test):,}")
        print(f"Number of classes: {len(class_names)}")
        print(f"Input dimension: {X_train.shape[1]}")
        print("-" * 50)
        
        # Initialize output dimension
        output_dim = len(class_names)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create data loaders
        train_dataset = SearchIntentDataset(X_train_scaled, y_train)
        test_dataset = SearchIntentDataset(X_test_scaled, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Initialize model
        self.model = IntentClassifier(
            self.input_dim, 
            self.hidden_dim, 
            output_dim, 
            self.dropout
        ).to(self.device)
        
        # Handle model loading based on continue_training flag
        model_path = 'models/neural_network.pt'
        if continue_training and os.path.exists(model_path):
            try:
                print("Loading existing neural network model for continued training...")
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            except Exception as e:
                print(f"Error loading existing model: {e}")
                print("Starting fresh training instead...")
        else:
            print("Starting fresh training...")
        
        # Training setup
        bce_criterion = nn.BCELoss()
        f1_criterion = F1Loss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, verbose=True
        )
        
        best_loss = float('inf')
        
        # Training loop
        for epoch in range(n_epochs):
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}')
            
            for batch_X, batch_y in progress_bar:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                # Combine BCE and F1 losses
                bce_loss = bce_criterion(outputs, batch_y)
                f1_loss = f1_criterion(outputs, batch_y)
                loss = bce_loss + f1_loss  # You can adjust weights if needed, e.g.: 0.7 * bce_loss + 0.3 * f1_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({
                    'loss': total_loss/len(train_loader),
                    'bce': bce_loss.item(),
                    'f1': f1_loss.item()
                })
            
            # Validation
            val_loss, val_f1 = self._validate(test_loader, bce_criterion, f1_criterion)
            scheduler.step(val_loss)
            
            print(f'Epoch {epoch+1}: Val Loss = {val_loss:.4f}, F1 Score = {val_f1:.4f}')
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                self.save_model()
    
    def _validate(self, test_loader, bce_criterion, f1_criterion):
        """Validate the model."""
        self.model.eval()
        val_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                outputs = self.model(batch_X)
                
                # Calculate both losses
                bce_loss = bce_criterion(outputs, batch_y)
                f1_loss = f1_criterion(outputs, batch_y)
                loss = bce_loss + f1_loss
                
                val_loss += loss.item()
                
                predictions.extend((outputs > 0.5).cpu().numpy())
                true_labels.extend(batch_y.cpu().numpy())
        
        val_loss /= len(test_loader)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        
        return val_loss, f1
    
    def predict(self, X):
        """Predict labels for new data."""
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        # Create dataset with dummy labels of correct output dimension
        output_dim = self.model.output_layers[-2].out_features
        dataset = SearchIntentDataset(X_scaled, np.zeros((len(X), output_dim)))
        loader = DataLoader(dataset, batch_size=32)
        
        predictions = []
        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                predictions.extend((outputs > 0.5).cpu().numpy())
        
        return np.array(predictions)
    
    def save_model(self, model_dir='models'):
        """Save the model and scaler."""
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(model_dir, 'neural_network.pt'))
        with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load_model(self, model_dir='models', output_dim=None):
        """Load the model and scaler."""
        if output_dim is not None and self.model is None:
            self.model = IntentClassifier(
                self.input_dim, 
                self.hidden_dim, 
                output_dim, 
                self.dropout
            ).to(self.device)
        
        self.model.load_state_dict(torch.load(
            os.path.join(model_dir, 'neural_network.pt'),
            map_location=self.device
        ))
        with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)

def evaluate_model(model, X_test, y_test, class_names):
    """Evaluate the model and return detailed metrics."""
    predictions = model.predict(X_test)
    
    # Calculate metrics with zero_division handling
    report = classification_report(
        y_test,
        predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # Calculate accuracy
    accuracy = (y_test == predictions).mean()
    
    # Prepare detailed metrics
    metrics = {
        "Accuracy": accuracy,
        "Macro F1-Score": report['macro avg']['f1-score'],
        "Weighted F1-Score": report['weighted avg']['f1-score'],
        "Macro Precision": report['macro avg']['precision'],
        "Weighted Precision": report['weighted avg']['precision'],
        "Macro Recall": report['macro avg']['recall'],
        "Weighted Recall": report['weighted avg']['recall'],
        "Per-Class Metrics": {
            class_name: {
                "F1-Score": report[class_name]['f1-score'],
                "Precision": report[class_name]['precision'],
                "Recall": report[class_name]['recall'],
                "Support": report[class_name]['support']
            }
            for class_name in class_names
        },
        "Detailed Report": report
    }
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neural Network-based text classification')
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--continue_training', action='store_true', help='Continue training existing model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden layer dimension')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--data_dir', type=str, default='data/kw_csvs', help='Directory containing training CSV files')
    parser.add_argument('--test_file', type=str, default='data/test/test.csv', help='Path to test CSV file')
    args = parser.parse_args()

    if args.train or args.continue_training:
        # Get all CSV files from data directory
        csv_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.csv')]
        if not csv_files:
            print(f"No CSV files found in {args.data_dir}")
            exit(1)
            
        print(f"Training on {len(csv_files)} CSV files...")
        
        # Load and preprocess data
        all_data = pd.DataFrame()
        print(len(csv_files))
        for file in tqdm(csv_files, desc="Loading data"):
            data = pd.read_csv(file)
            all_data = pd.concat([all_data, data], ignore_index=True)
        
        # Prepare data
        X = get_bert_embeddings(
            all_data['Keyword'].tolist(),
            data_source=args.data_dir
        )
        all_data['labels'] = all_data['Search intent'].apply(lambda x: [label.strip() for label in x.split(',')])
        
        # Initialize or load label binarizer
        if args.continue_training and os.path.exists('models/label_binarizer.pkl'):
            with open('models/label_binarizer.pkl', 'rb') as f:
                mlb = pickle.load(f)
                y = mlb.transform(all_data['labels'])
        else:
            mlb = MultiLabelBinarizer()
            y = mlb.fit_transform(all_data['labels'])
            os.makedirs('models', exist_ok=True)
            with open('models/label_binarizer.pkl', 'wb') as f:
                pickle.dump(mlb, f)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y.sum(axis=1)
        )
        
        # Initialize and train classifier
        classifier = NeuralClassifier(
            input_dim=X_train.shape[1],
            hidden_dim=args.hidden_dim,
            dropout=args.dropout
        )
        
        classifier.train(
            X_train, X_test, y_train, y_test,
            class_names=mlb.classes_,
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            learning_rate=args.learning_rate,
            continue_training=args.continue_training
        )
        
        # Evaluate and print detailed results
        metrics = evaluate_model(classifier, X_test, y_test, mlb.classes_)
        print("\nFinal Training Results:")
        print(f"Accuracy: {metrics['Accuracy']:.4f}")
        print(f"Weighted F1-Score: {metrics['Weighted F1-Score']:.4f}")
        print(f"Weighted Precision: {metrics['Weighted Precision']:.4f}")
        print(f"Weighted Recall: {metrics['Weighted Recall']:.4f}")
        print("\nMacro Averages:")
        print(f"Macro F1-Score: {metrics['Macro F1-Score']:.4f}")
        print(f"Macro Precision: {metrics['Macro Precision']:.4f}")
        print(f"Macro Recall: {metrics['Macro Recall']:.4f}")
        
        print("\nPer-Class Performance:")
        for class_name, class_metrics in metrics['Per-Class Metrics'].items():
            print(f"\n{class_name}:")
            print(f"  F1-Score: {class_metrics['F1-Score']:.4f}")
            print(f"  Precision: {class_metrics['Precision']:.4f}")
            print(f"  Recall: {class_metrics['Recall']:.4f}")
            print(f"  Support: {class_metrics['Support']}")
        
        # Save detailed metrics to file
        results_file = "results/training_metrics.txt"
        os.makedirs('results', exist_ok=True)
        with open(results_file, 'w') as f:
            f.write("Training Metrics Summary\n")
            f.write("=======================\n\n")
            f.write(f"Accuracy: {metrics['Accuracy']:.4f}\n")
            f.write(f"Weighted F1-Score: {metrics['Weighted F1-Score']:.4f}\n")
            f.write(f"Weighted Precision: {metrics['Weighted Precision']:.4f}\n")
            f.write(f"Weighted Recall: {metrics['Weighted Recall']:.4f}\n\n")
            f.write("Macro Averages\n")
            f.write("--------------\n")
            f.write(f"Macro F1-Score: {metrics['Macro F1-Score']:.4f}\n")
            f.write(f"Macro Precision: {metrics['Macro Precision']:.4f}\n")
            f.write(f"Macro Recall: {metrics['Macro Recall']:.4f}\n\n")
            f.write("Per-Class Performance\n")
            f.write("--------------------\n")
            for class_name, class_metrics in metrics['Per-Class Metrics'].items():
                f.write(f"\n{class_name}:\n")
                f.write(f"  F1-Score: {class_metrics['F1-Score']:.4f}\n")
                f.write(f"  Precision: {class_metrics['Precision']:.4f}\n")
                f.write(f"  Recall: {class_metrics['Recall']:.4f}\n")
                f.write(f"  Support: {class_metrics['Support']}\n")
        
        print(f"\nDetailed metrics saved to: {results_file}")

    if args.test:
        if not os.path.exists(args.test_file):
            print(f"Test file not found: {args.test_file}")
            exit(1)
            
        # Load test data
        test_data = pd.read_csv(args.test_file)
        X_test = get_bert_embeddings(
            test_data['Keyword'].tolist(),
            data_source=args.test_file
        )
        test_data['labels'] = test_data['Search intent'].apply(lambda x: [label.strip() for label in x.split(',')])
        
        # Load label binarizer and transform labels
        with open('models/label_binarizer.pkl', 'rb') as f:
            mlb = pickle.load(f)
        y_test = mlb.transform(test_data['labels'])
        
        # Initialize and load trained classifier
        classifier = NeuralClassifier(
            input_dim=X_test.shape[1],
            hidden_dim=args.hidden_dim,
            dropout=args.dropout
        )
        classifier.load_model(output_dim=len(mlb.classes_))
        
        # Print and save test results
        metrics = evaluate_model(classifier, X_test, y_test, mlb.classes_)
        print("\nTest Results:")
        print(f"Accuracy: {metrics['Accuracy']:.4f}")
        print(f"Weighted F1-Score: {metrics['Weighted F1-Score']:.4f}")
        print(f"Weighted Precision: {metrics['Weighted Precision']:.4f}")
        print(f"Weighted Recall: {metrics['Weighted Recall']:.4f}")
        
        # Save test metrics
        test_results_file = "results/test_metrics.txt"
        with open(test_results_file, 'w') as f:
            f.write("Test Metrics Summary\n")
            f.write("===================\n\n")
            f.write(f"Accuracy: {metrics['Accuracy']:.4f}\n")
            f.write(f"Weighted F1-Score: {metrics['Weighted F1-Score']:.4f}\n")
            f.write(f"Weighted Precision: {metrics['Weighted Precision']:.4f}\n")
            f.write(f"Weighted Recall: {metrics['Weighted Recall']:.4f}\n")
        
        print(f"\nDetailed test metrics saved to: {test_results_file}")

    if not args.train and not args.test:
        parser.print_help()