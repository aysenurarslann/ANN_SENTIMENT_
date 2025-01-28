import numpy as np
from sklearn.model_selection import KFold
import torch
from training import train_model, evaluate_model
from model import DeepSentimentClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from itertools import product
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class ComplexSentimentClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.4):
        super(ComplexSentimentClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc4 = nn.Linear(hidden_dim // 4, output_dim)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

def ten_fold_cv(X_tensor, y_tensor, input_dim, hidden_dim, output_dim, dropout_rate):
    print("\n--- 10-Fold Cross Validation ---")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_accuracies = []

    for train_idx, test_idx in kf.split(X_tensor):
        X_train, X_test = X_tensor[train_idx], X_tensor[test_idx]
        y_train, y_test = y_tensor[train_idx], y_tensor[test_idx]
        
        model = ComplexSentimentClassifier(input_dim, hidden_dim, output_dim, dropout_rate)
        model = train_model(model, X_train, y_train, epochs=15, lr=0.0005)
        acc = evaluate_model(model, X_test, y_test)
        fold_accuracies.append(acc)

    print(f"10-Fold Cross Validation Ortalama Doğruluk: {np.mean(fold_accuracies):.4f}")


def ten_fold_cv_hyperparameter_search(X_tensor, y_tensor, input_dim, output_dim):
    print("\n--- 10-Fold Cross Validation Hiperparametre Araması ---")
    # Hiperparametre değerleri
    hidden_dim_list = [64, 128, 256]  # Gizli katman boyutları
    dropout_rate_list = [0.3, 0.4, 0.5]  # Dropout oranları
    lr_list = [0.001, 0.0005, 0.0001]  # Öğrenme oranları

    # Kombinasyonları oluştur
    param_combinations = list(product(hidden_dim_list, dropout_rate_list, lr_list))

    best_acc = 0
    best_params = None
    best_model = None

    for hidden_dim, dropout_rate, lr in param_combinations:
        print(f"\nDeneme: hidden_dim={hidden_dim}, dropout_rate={dropout_rate}, lr={lr}")
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        fold_accuracies = []

        for train_idx, test_idx in kf.split(X_tensor):
            X_train, X_test = X_tensor[train_idx], X_tensor[test_idx]
            y_train, y_test = y_tensor[train_idx], y_tensor[test_idx]

            model = ComplexSentimentClassifier(input_dim, hidden_dim, output_dim, dropout_rate)
            model = train_model(model, X_train, y_train, epochs=15, lr=lr)
            acc = evaluate_model(model, X_test, y_test)
            fold_accuracies.append(acc)

        mean_acc = np.mean(fold_accuracies)
        print(f"Ortalama Doğruluk (10-Fold): {mean_acc:.4f}")

        # En iyi parametrelerin kaydedilmesi
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_params = (hidden_dim, dropout_rate, lr)
            best_model = model

    # En iyi doğruluk ve parametreler ile en son modelin doğruluğunu yazdırıyoruz
    print(f"\nEn iyi doğruluk: {best_acc:.4f} --- Parametreler: hidden_dim={best_params[0]}, dropout_rate={best_params[1]}, lr={best_params[2]}")

    # En iyi modelin doğruluğunu tekrar hesaplayıp yazdırıyoruz ve konfizyon matrisi çiziyoruz
    print("\n--- En İyi Modelin Sonuçları ---")
    y_pred = best_model(torch.tensor(X_tensor, dtype=torch.float32)).argmax(dim=1).detach().numpy()
    y_true = y_tensor.detach().numpy()

    cm = confusion_matrix(y_true, y_pred)
    print("Konfüzyon Matrisi:\n", cm)

    # Konfüzyon Matrisi Görselleştirme
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='viridis')
    plt.title("En İyi Model - Konfüzyon Matrisi")
    plt.show()

    return best_model, best_params