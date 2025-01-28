import torch
import torch.nn as nn
from torchviz import make_dot
import matplotlib.pyplot as plt

# Model sınıfı yeniden tanımlanır
class MediumSentimentClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.3):
        super(MediumSentimentClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# Model parametreleri
en_iyi_parametreler = {
    "input_dim": 100,  # Giriş boyutu (örnek olarak 100 verilmiştir)
    "hidden_dim": 256,
    "output_dim": 3,   # Çıkış boyutu (örneğin 3 sınıf: pozitif, negatif, nötr)
    "dropout_rate": 0.2
}

# Modeli oluştur ve bir örnek giriş tensorü ile çalıştır
model = MediumSentimentClassifier(**en_iyi_parametreler)
x = torch.randn(1, en_iyi_parametreler["input_dim"])  # Rastgele bir giriş tensorü
output = model(x)

# Modelin görselleştirilmesi
graph = make_dot(output, params=dict(model.named_parameters()))
graph.render("medium_sentiment_classifier", format="png", cleanup=True)

# Model parametrelerini ve en iyi doğruluğu matplotlib ile görselleştir
plt.figure(figsize=(8, 6))
plt.axis('off')
plt.title("Model Parametreleri ve Doğruluk", fontsize=14)
plt.text(0.1, 0.7, f"En İyi Doğruluk: 0.6670", fontsize=12, weight='bold')
plt.text(0.1, 0.6, f"Hidden Dim: {en_iyi_parametreler['hidden_dim']}", fontsize=12)
plt.text(0.1, 0.5, f"Dropout Rate: {en_iyi_parametreler['dropout_rate']}", fontsize=12)
plt.text(0.1, 0.4, f"Learning Rate: 0.001", fontsize=12)
plt.text(0.1, 0.3, f"Output Dim: {en_iyi_parametreler['output_dim']}", fontsize=12)
plt.text(0.1, 0.2, f"Input Dim: {en_iyi_parametreler['input_dim']}", fontsize=12)
plt.savefig("model_parameters.png")
plt.show()

print("Model grafiği ve parametre görselleştirmeleri kaydedildi.")
