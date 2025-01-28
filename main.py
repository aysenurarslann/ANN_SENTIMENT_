import torch
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
from data_preparation import load_data, preprocess_tweets, extract_features, balance_data
from train_test_same import train_and_test_same
from five_fold_cv import five_fold_cv
from ten_fold_cv import ten_fold_cv
from train_test_same import SimpleSentimentClassifier
from five_fold_cv import MediumSentimentClassifier
from ten_fold_cv import ComplexSentimentClassifier
from train_test_same import train_and_test_same, train_and_test_hyperparameter_search
from five_fold_cv import five_fold_cv, five_fold_cv_hyperparameter_search
from ten_fold_cv import ten_fold_cv, ten_fold_cv_hyperparameter_search
from train_test_random import random_split_hyperparameter_search, random_split_evaluation,RandomSplitClassifier
from visualize import visualize_model
from train_test_random import retrain_best_model
# Veri Yükleme ve Önişleme
file_path = 'test_2200.txt'
tweets, sentiments = load_data(file_path)
tweets = preprocess_tweets(tweets)
X, tfidf_vectorizer = extract_features(tweets)
y = np.array(sentiments)
X_balanced, y_balanced = balance_data(X, y)

# PyTorch Tensörlere Dönüştürme
X_tensor = torch.tensor(X_balanced, dtype=torch.float32)
y_tensor = torch.tensor(y_balanced, dtype=torch.long)


# Model Parametreleri
input_dim = X.shape[1]
output_dim = len(np.unique(y))

# 1. eğitm setini test seti olarak kullan
#  Hiperparametre Araması
print("\n--- Hiperparametre Araması ---")
best_model, best_params = train_and_test_hyperparameter_search(X_tensor, y_tensor, input_dim, output_dim)
#  En İyi Model ile Eğitim Seti Testi
hidden_dim, dropout_rate, best_lr = best_params  # Hiperparametrelerden en iyilerini al
print("\n--- En İyi Model ile Eğitim ve Test ---")
train_and_test_same(X_tensor, y_tensor, input_dim, hidden_dim, output_dim, dropout_rate)
model = SimpleSentimentClassifier(input_dim, hidden_dim, output_dim, dropout_rate)
visualize_model(model, input_size=100)

# #  # 2. 5-Fold Cross Validation
# # # 1. Hiperparametre Araması (5-Fold CV)
print("\n--- 5-Fold Cross Validation Hiperparametre Araması ---")
best_params = five_fold_cv_hyperparameter_search(X_tensor, y_tensor, input_dim, output_dim)
# # # 2. En İyi Model ile 5-Fold CV
hidden_dim, dropout_rate, best_lr = best_params
print("\n--- En İyi Model ile 5-Fold CV ---")
five_fold_cv(X_tensor, y_tensor, input_dim, hidden_dim, output_dim, dropout_rate)






# # 3. 10-Fold Cross Validation
# 1. Hiperparametre Araması (10-Fold CV)
print("\n--- 10-Fold Cross Validation Hiperparametre Araması ---")
best_params = ten_fold_cv_hyperparameter_search(X_tensor, y_tensor, input_dim, output_dim)
#  # 2. En İyi Model ile 10-Fold CV
hidden_dim, dropout_rate, best_lr = best_params
print("\n--- En İyi Model ile 10-Fold CV ---")
ten_fold_cv(X_tensor, y_tensor, input_dim, hidden_dim, output_dim, dropout_rate)




# # 4. •	%66-%34 eğitim test ayırarak (5 farklı rassal ayırma ile)

# # 1. Hiperparametre Araması (%66-%34)
print("\n--- Rastgele Eğitim/Test Ayırma Hiperparametre Araması ---")
best_params = random_split_hyperparameter_search(X_tensor, y_tensor, input_dim, output_dim)
# 2. En İyi Model ile %66-%34 Değerlendirme
hidden_dim, dropout_rate, best_lr = best_params
print("\n--- En İyi Model ile %66-%34 Değerlendirme ---")
random_split_evaluation(X_tensor, y_tensor, input_dim, hidden_dim, output_dim, dropout_rate)
retrain_best_model(X_tensor, y_tensor, input_dim, output_dim, best_params)
