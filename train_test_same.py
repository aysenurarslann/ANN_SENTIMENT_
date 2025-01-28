import torch
from training import train_model, evaluate_model
from model import DeepSentimentClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F



class SimpleSentimentClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.3):
        super(SimpleSentimentClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



def train_and_test_same(X_tensor, y_tensor, input_dim, hidden_dim, output_dim, dropout_rate):
    print("\n--- Eğitim Setini Aynı Zamanda Test Verisi Olarak Kullanma ---")
    model = SimpleSentimentClassifier(input_dim, hidden_dim, output_dim, dropout_rate)
    model = train_model(model, X_tensor, y_tensor, epochs=15, lr=0.0005)
    acc = evaluate_model(model, X_tensor, y_tensor)
    print(f"Eğitim Seti Doğruluğu: {acc:.4f}")
from itertools import product

def train_and_test_hyperparameter_search(X_tensor, y_tensor, input_dim, output_dim):
    print("\n--- Hiperparametre Araması Başlıyor ---")
    # Hiperparametrelerin değerlerini belirleyin
    hidden_dim_list = [32, 64, 128]  # Gizli katman boyutları
    dropout_rate_list = [0.2, 0.3, 0.5]  # Dropout oranları
    lr_list = [0.001, 0.0005, 0.0001]  # Öğrenme oranları

    # Kombinasyonları oluştur
    param_combinations = list(product(hidden_dim_list, dropout_rate_list, lr_list))

    best_acc = 0
    best_params = None
    best_model = None

    for hidden_dim, dropout_rate, lr in param_combinations:
        print(f"\nDeneme: hidden_dim={hidden_dim}, dropout_rate={dropout_rate}, lr={lr}")
        model = SimpleSentimentClassifier(input_dim, hidden_dim, output_dim, dropout_rate)
        
        # Modeli eğit
        model = train_model(model, X_tensor, y_tensor, epochs=15, lr=lr)
        
        # Modeli değerlendir
        acc = evaluate_model(model, X_tensor, y_tensor)
        print(f"Doğruluk: {acc:.4f}")
        
        # En iyi modeli kaydet
        if acc > best_acc:
            best_acc = acc
            best_params = (hidden_dim, dropout_rate, lr)
            best_model = model

     # En iyi doğruluk ve parametreler ile en son modelin doğruluğunu yazdırıyoruz
    print(f"\nEn iyi doğruluk: {best_acc:.4f} --- Parametreler: hidden_dim={best_params[0]}, dropout_rate={best_params[1]}, lr={best_params[2]}")

    # En iyi modelin doğruluğunu tekrar hesaplayıp yazdırıyoruz
    final_acc = evaluate_model(best_model, X_tensor, y_tensor)
    print(f"En iyi modelin doğruluğu: {final_acc:.4f}")

    return best_model, best_params



