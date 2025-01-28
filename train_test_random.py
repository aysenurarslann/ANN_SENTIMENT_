import numpy as np
from sklearn.model_selection import train_test_split
import torch
from training import train_model, evaluate_model
import torch.nn as nn
import torch.optim as optim
from itertools import product
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class RandomSplitClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.3):
        super(RandomSplitClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x

def random_split_evaluation(X_tensor, y_tensor, input_dim, hidden_dim, output_dim, dropout_rate, lr):
    print("\n--- En İyi Model ile %66-%34 Değerlendirme ---")
    accuracies = []

    for split in range(5):  # 5 farklı rastgele ayırma
        X_train, X_test, y_train, y_test = train_test_split(
            X_tensor, y_tensor, test_size=0.34, random_state=42 + split
        )

        model = RandomSplitClassifier(input_dim, hidden_dim, output_dim, dropout_rate)
        model = train_model(model, X_train, y_train, epochs=15, lr=lr)
        acc = evaluate_model(model, X_test, y_test)
        accuracies.append(acc)

    mean_acc = np.mean(accuracies)
    print(f"En İyi Model Rastgele Eğitim/Test Ayırma Ortalama Doğruluk (%66-%34): {mean_acc:.4f}")
    return mean_acc


def random_split_hyperparameter_search(X_tensor, y_tensor, input_dim, output_dim):
    print("\n--- Rastgele Eğitim/Test Ayırma Hiperparametre Araması ---")
    # Hiperparametre kombinasyonları
    hidden_dim_list = [64, 128, 256]
    dropout_rate_list = [0.3, 0.4, 0.5]
    lr_list = [0.001, 0.0005, 0.0001]

    param_combinations = list(product(hidden_dim_list, dropout_rate_list, lr_list))
    best_acc = 0
    best_params = None
    best_model = None

    for hidden_dim, dropout_rate, lr in param_combinations:
        print(f"\nDeneme: hidden_dim={hidden_dim}, dropout_rate={dropout_rate}, lr={lr}")
        accuracies = []

        for split in range(5):  # 5 farklı rastgele ayırma
            X_train, X_test, y_train, y_test = train_test_split(
                X_tensor, y_tensor, test_size=0.34, random_state=42 + split
            )

            model = RandomSplitClassifier(input_dim, hidden_dim, output_dim, dropout_rate)
            model = train_model(model, X_train, y_train, epochs=15, lr=lr)
            acc = evaluate_model(model, X_test, y_test)
            accuracies.append(acc)

        mean_acc = np.mean(accuracies)
        print(f"Ortalama Doğruluk (%66-%34): {mean_acc:.4f}")

        if mean_acc > best_acc:
            best_acc = mean_acc
            best_params = (hidden_dim, dropout_rate, lr)
            best_model = model
            
    print(f"\nEn iyi doğruluk: {best_acc:.4f} --- Parametreler: hidden_dim={best_params[0]}, dropout_rate={best_params[1]}, lr={best_params[2]}")
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

def retrain_best_model(X_tensor, y_tensor, input_dim, output_dim, best_params):
    print("\n--- En İyi Parametrelerle Son Eğitim ve Test ---")
    hidden_dim, dropout_rate, lr = best_params
    
    # Eğitim ve Test için yeni bir ayırma
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=0.34, random_state=42
    )
    
    # Yeni model
    final_model = RandomSplitClassifier(input_dim, hidden_dim, output_dim, dropout_rate)
    final_model = train_model(final_model, X_train, y_train, epochs=25, lr=lr)  # Daha fazla epoch
    
    # Test performansı
    final_accuracy = evaluate_model(final_model, X_test, y_test)
    print(f"Final Test Doğruluğu: {final_accuracy:.4f}")
    
    # Konfüzyon Matrisi
    y_pred = final_model(torch.tensor(X_test, dtype=torch.float32)).argmax(dim=1).detach().numpy()
    y_true = y_test.detach().numpy()
    cm = confusion_matrix(y_true, y_pred)
    print("Son Konfüzyon Matrisi:\n", cm)
    
    # Konfüzyon Matrisi Görselleştirme
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='viridis')
    plt.title("Son Model - Konfüzyon Matrisi")
    plt.show()

    return final_model, final_accuracy

