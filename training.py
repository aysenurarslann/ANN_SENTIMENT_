import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from model import DeepSentimentClassifier, FocalLoss
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args



# Model Eğitimi
def train_model(model, X_train, y_train, epochs=10, lr=0.001):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = FocalLoss()  # Focal Loss kullanımı
    model.train()
    train_losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
    
    return model

# Model Değerlendirme
def evaluate_model(model, X_test, y_test):
    model.eval()
    outputs = model(X_test).detach().cpu().numpy()
    preds = outputs.argmax(axis=1)
    acc = accuracy_score(y_test, preds)
    print("Classification Report:")
    print(classification_report(y_test, preds))
    print("Confusion Matrix:")
    
    return acc
