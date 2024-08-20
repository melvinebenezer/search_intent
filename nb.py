import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class GaussianNaiveBayes(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GaussianNaiveBayes, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.epsilon = 1e-6

        # Initialize parameters
        self.class_priors = nn.Parameter(torch.ones(num_classes) / num_classes)
        self.means = nn.Parameter(torch.randn(num_classes, num_features))
        self.log_variances = nn.Parameter(torch.zeros(num_classes, num_features))

    def forward(self, x):
        log_probs = torch.zeros(x.shape[0], self.num_classes, device=x.device)

        for c in range(self.num_classes):
            class_log_prior = torch.log(self.class_priors[c] + self.epsilon)
            variances = torch.exp(self.log_variances[c]) + self.epsilon
            log_det = torch.sum(torch.log(variances))
            diff = x - self.means[c]
            mahalanobis = torch.sum((diff ** 2) / variances, dim=1)
            log_probs[:, c] = class_log_prior - 0.5 * (log_det + mahalanobis + self.num_features * np.log(2 * np.pi))

        return log_probs

    def fit(self, X, y, lr=0.01, epochs=100):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.NLLLoss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            log_probs = self(X)
            loss = criterion(log_probs, y)
            
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch+1}. Stopping training.")
                break
            
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                # Ensure priors sum to 1
                self.class_priors.data = torch.softmax(self.class_priors, dim=0)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    def evaluate(self, X, y):
        with torch.no_grad():
            log_probs = self(X)
            predictions = torch.argmax(log_probs, dim=1)
            accuracy = (predictions == y).float().mean()
        return accuracy.item()

# Usage remains the same
if __name__ == '__main__':
    X = torch.randn(100, 2)
    y = torch.randint(0, 3, (100,))

    model = GaussianNaiveBayes(num_features=2, num_classes=3)
    model.fit(X, y)

    accuracy = model.evaluate(X, y)
    print(f'Accuracy: {accuracy}')

    # Save the model
    torch.save(model.state_dict(), 'nb_model.pth')