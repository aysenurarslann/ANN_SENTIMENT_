# ANN_Sentiment
  Duygu Analizi Modeli: Eğitim, Değerlendirme ve Optimizasyon Raporu
Bu proje, sosyal medya platformlarından alınan metin verilerini kullanarak duygu analizi yapmayı hedeflemektedir. Amacımız, bu metinlerin (tweetler, yorumlar vb.) pozitif, negatif veya nötr duygu içerip içermediğini sınıflandırmaktır. Bu, kullanıcı geri bildirimleri, müşteri yorumları gibi metinlerden duygu çıkarma problemlerine yönelik pratik bir çözümdür.
1.Sınıflandırılacak Öğeler
Bu projede, sosyal medya gönderileri duygu içeriklerine göre sınıflandırılacaktır. Sınıflandırılacak ana duygu kategorileri şunlardır:
Pozitif: Olumlu duygu taşıyan içerikler.
Negatif: Olumsuz duygu taşıyan içerikler.
2. Veri Seti Hakkında
Özellikler
Veri seti, sosyal medya gönderilerinden (tweetlerden) elde edilen metin verilerinden oluşmaktadır. Her gönderinin yanında, gönderinin duygu kategorisini belirten bir etiket bulunmaktadır. Öne çıkan özellikler:
Metin Verisi: Her tweetin içeriği, duygu kategorisini anlamak için kullanılır.
Etiketler (Sentiment Labels): Her tweetin duygu etiketi (pozitif, negatif veya nötr).
Emojiler: Duygu analizi için önemli bir kaynaktır ve metin içindeki duygusal tonları daha iyi anlamak için kullanılabilir. Emojiler sayısal verilere dönüştürülerek modelde doğrudan analiz edilebilir veya belirli duygu etiketlerine bağlanabilir.
URL'ler: Genellikle metnin duygu içerikli anlamını bozan ögeler olarak kabul edilse de, bazı durumlarda bağlam için anlam taşıyabilir. URL'ler ya çıkarılabilir ya da "URL" token olarak işlenebilir.

Örnek Sayısı
Veri seti 2200 örnekten oluşmaktadır. Bu, her biri bir tweet olan toplam 2200 veriden oluşur ve her bir tweet bir etiket ile eşleştirilmiştir.
Sınıf Bilgisi
Veri setindeki etiketler iki sınıftan oluşur:
Pozitif (1): Olumlu içeriklere sahip tweetler.
Negatif (0): Olumsuz içeriklere sahip tweetler.


3. Uygulama Hakkında
Bu projede, eğitim ve test veri ayırma, çapraz doğrulama (cross-validation) ve hiperparametre optimizasyonu gibi çeşitli model değerlendirme stratejileri uygulanmıştır. Her strateji için ayrıntılı olarak aşağıdaki uygulama adımları izlenmiştir:
a)Veri Önişleme
  1)Temizleme İşlemleri:
  URL'lerin kaldırılması: re.sub(r'http\S+|www\S+|https\S+', '', tweet)
  Metinlerde geçen URL'ler kaldırılarak veri temizlenir.
  Özel karakterlerin kaldırılması: re.sub(r'[^a-zA-Z\s]', '', tweet.lower())
  Harfler ve boşluklar dışında tüm karakterler kaldırılır.
  Küçük harfe dönüştürme: tweet.lower()
  Tüm metin küçük harfe dönüştürülür.
  2)Stop Word Filtreleme:
  Stopwords: NLP işlemlerinde sık kullanılan anlamsız kelimeler (ör., "the", "is").
  Burada NLTK’nin İngilizce stopword listesi yüklenmiş ve "not" gibi önemli kelimeler bu
  listeden çıkarılmıştır    
 nltk.word_tokenize() kullanılarak metinler tokenize edilmiştir (kelimelerine
  ayrılmıştır).
  3)Kullanılan Teknikler:
  Regex (Regular Expressions): URL ve özel karakter temizliği.
  Stopwords İşlemleri: Kelime filtreleme.
  Tokenization: Metni kelimelere ayırma.

b)Özellik Çıkarımı
  1)TF-IDF (Term Frequency-Inverse Document Frequency):
  TfidfVectorizer kullanılarak tweetlerden özellikler çıkarılmıştır.
   Her bir tweet, bir vektör olarak temsil edilir.
   max_features=5000: Sadece en anlamlı 5000 özelliği (kelimeyi) seçer.
  2)Kullanılan Teknikler:
   TF-IDF: Metin verisini sayısal forma dönüştürmek için kullanılan yaygın bir yöntem. Sık             geçen ancak az anlam ifade eden kelimeler azaltılır.

c) Veri Dengesizliğini Giderme
  1)SMOTE (Synthetic Minority Oversampling Technique):
  Veri setindeki dengesiz sınıflar arasında denge kurmak için kullanılır.
  Azınlık sınıfına (örneğin, negatif tweetler) yeni örnekler ekler.
  smote.fit_resample() ile sınıf örneklerini dengeler.
  2)Kullanılan Teknikler:
  SMOTE: Veri dengesizliğini azaltmak için bir yeniden örnekleme yöntemi.

Training 
1. Model Eğitimi
train_model fonksiyonu, modelin eğitim sürecini gerçekleştirir.
Optimizasyon (Optimizer):
AdamW: optim.AdamW kullanılarak, model parametrelerini optimize etmek için Adam      optimizasyon algoritmasının ağırlık çürüme (weight decay) içeren bir varyasyonu  kullanılmıştır.
Adam: Öğrenme oranını uyarlayarak gradyan inişini hızlandırır.
Weight Decay: Ağırlık değerlerinin büyümesini kontrol ederek aşırı öğrenmeyi (overfitting)   azaltır.
Kayıp Fonksiyonu (Loss Function):
Focal Loss: Dengesiz veri setleriyle çalışırken yaygın bir sorun olan kolay sınıflandırılabilir örneklerin etkisini azaltır ve zor sınıflara odaklanmayı sağlar.
Eğitim Döngüsü
Gradyan Hesaplama ve Güncelleme:
optimizer.zero_grad(): Önceki gradyanları sıfırlar.
loss.backward(): Kayıp fonksiyonunun gradyanını hesaplar.
optimizer.step(): Model parametrelerini günceller.
Eğitim Kayıplarının İzlenmesi
Her epoch’ta kayıp değeri hesaplanır ve ekrana yazdırılır.
Kayıp değerleri bir listeye kaydedilir (train_losses) ve eğitim performansını analiz etmek için kullanılabilir.
Kullanılan Teknikler:
AdamW Optimizasyonu: Modern, etkili bir optimizasyon yöntemi.
Focal Loss: Dengesiz sınıf dağılımlarıyla başa çıkmak için.
Gradyan Tabanlı Eğitim: Gradyan inişi algoritması.

2. Model Değerlendirme
evaluate_model fonksiyonu, test verileri üzerinde modelin performansını değerlendirir.
Tahminlerin Hesaplanması:
model(X_test): Model, test verileriyle ileri besleme yapar.
outputs.argmax(axis=1): Her bir örnek için en yüksek olasılıklı sınıf tahmin edilir.
Doğruluk (Accuracy):
accuracy_score : Tahmin edilen sınıflar ile gerçek sınıflar arasındaki doğruluk oranını hesaplar.
Sınıflandırma Raporu:
classification_report: Precision, Recall, F1-score gibi sınıflandırma metriklerini sağlar.
Precision: Modelin doğru tahmin ettiği pozitif sınıfların oranı.
Recall: Gerçek pozitiflerin ne kadarının doğru tahmin edildiği.
F1-score: Precision ve Recall’un harmonik ortalaması.
Konfüzyon Matrisi:
Modelin doğru ve yanlış sınıflandırmalarını görselleştirmek için kullanılır.
Kullanılan Teknikler:
Accuracy Metrikleri: Modelin doğruluk oranını ölçmek için.
Sınıflandırma Raporu: Detaylı sınıflandırma analizi.
Konfüzyon Matrisi: Modelin hata analizi için.

3. Ek Özellikler ve Araçlar
Modelin Test Moduna Geçmesi:
model.eval(): Model, değerlendirme sırasında gradyan hesaplamaz ve dropout gibi eğitim sırasında kullanılan teknikleri devre dışı bırakır.
Seaborn ve Matplotlib Kullanımı:
Konfüzyon matrisinin görselleştirilmesi için.



Main 
Bir duygu analizi modelinin eğitilmesi, değerlendirilmesi ve optimize edilmesi sürecini içeriyor. Çeşitli model eğitimi ve değerlendirme stratejileri kullanılarak hiperparametre optimizasyonu, çapraz doğrulama, ve eğitim-test ayırma gibi yöntemler uygulanmış. Kodda kullanılan başlıca teknikler ve yöntemler aşağıda açıklanmıştır:
________________________________________
1. Veri Yükleme ve Önişleme
Veri Yükleme (load_data):
load_data(file_path) fonksiyonu ile bir metin dosyasındaki tweetler ve etiketler yüklenir.
Önişleme (preprocess_tweets):
Tweetler, metin temizleme işlemleri (stopword removal, tokenization vb.) ile önişlemeye tabi tutulur.
Özellik Çıkarımı (extract_features):
TF-IDF: Tweetlerden özellik çıkarımı için TfidfVectorizer kullanılır.
Veri Dengeleme (balance_data):
SMOTE: Veri dengesizliğini gidermek için SMOTE (Synthetic Minority Over-sampling Technique) kullanılır.
PyTorch Tensörlerine Dönüştürme:
NumPy dizileri, PyTorch tensörlerine dönüştürülür (torch.tensor).
Kullanılan Teknikler:
________________________________________
2. Model Eğitim ve Test Yöntemleri
Eğitim Setini Test Seti Olarak Kullanma (train_and_test_same)
Eğitim seti hem eğitim hem de test verisi olarak kullanılır. Model, eğitim verisiyle eğitildikten sonra aynı veri üzerinde test edilir.
Kullanılan Teknikler:
Overfitting Testi: Eğitim verisi üzerinde modelin performansını ölçmek.
Hiperparametre Araması (train_and_test_hyperparameter_search)
Hiperparametreler için sistematik bir arama yapılır (ör. hidden_dim, dropout_rate, lr).
Grid Search: Farklı hiperparametre kombinasyonları test edilir ve en iyi model seçilir.



Çapraz Doğrulama Yöntemleri:
1)5-Fold Cross Validation (five_fold_cv)
Veri 5'e bölünür, her bir fold için model eğitilir ve test edilir. Sonuçlar ortalanarak genelleme performansı ölçülür.
2)10-Fold Cross Validation (ten_fold_cv)
Aynı şekilde 10'a bölünür ve çapraz doğrulama yapılır.
Kullanılan Teknikler:
K-Fold Cross Validation (5-Fold ve 10-Fold): Modelin genelleme yeteneğini değerlendirmek için kullanılan yöntem.
Rastgele Eğitim-Test Ayırma (random_split_evaluation)
Veri %66 eğitim ve %34 test olarak rastgele ayrılır.
Hiperparametre Araması: Eğitimin ardından test doğruluğu ölçülür ve en iyi model belirlenir.
Kullanılan Teknikler:
Rastgele Eğitim-Test Ayırma (%66-%34): Veri setinin eğitim ve test için rasgele bölünmesi.
Hiperparametre Araması: En iyi doğruluğu sağlayan model parametrelerini bulma.

3)Model Görselleştirme ve Sonuç Raporlama
visualize_model Fonksiyonu:
Modelin yapısı görselleştirilir. Bu fonksiyon, modelin katmanlarını ve bağlantılarını daha iyi anlamak için kullanılır.
Konfüzyon Matrisi (ConfusionMatrixDisplay):
Modelin doğruluk, yanlış sınıflandırma, precision, recall gibi performans metriklerini görsel olarak raporlar.
4)Model Performansı ve Final Değerlendirmesi
Final Model Eğitimi (retrain_best_model):
Hiperparametreleri optimize edilmiş en iyi model ile final eğitimi yapılır ve test edilir. Sonuçlar tekrar hesaplanarak modelin nihai doğruluğu belirlenir.





Eğitim Setini Aynı Anda Test Verisi Olarak Kullanma:
Bu yöntemde, eğitim verisi ve test verisi aynı veri kümesi olarak kullanılmıştır. Model eğitildikten sonra, eğitim verisi üzerinde test yapılmıştır. Bu, overfitting riskini görmek ve modelin eğitim sürecindeki doğruluğu hızlıca analiz etmek için kullanılabilir.
Bu kodda, SimpleSentimentClassifier adlı bir sinir ağı modeli oluşturulmuş ve farklı eğitim stratejileriyle performansı değerlendirilmiştir. Kullanılan teknikler ve yöntemler aşağıdaki gibi açıklanabilir:
1. Sinir Ağı Tanımı
SimpleSentimentClassifier
Bu model, temel bir ileri beslemeli sinir ağıdır (Feedforward Neural Network). Daha basit bir model topolojisi tercih edildi.
Katmanlar:
fc1: Girişten gizli katmana tam bağlantılı (fully connected) bir katman.
fc2: Gizli katmandan çıkış katmanına tam bağlantılı bir katman.
Aktivasyon Fonksiyonu:
ReLU (Rectified Linear Unit):
Doğrusal olmayan dönüşüm sağlar. Negatif girdileri sıfıra çeker.
Dropout:
%20-%50 arasında nöronları rastgele devre dışı bırakarak aşırı öğrenmeyi (overfitting) azaltır.
Kullanılan Teknikler:
Tam Bağlantılı Katmanlar (Fully Connected Layers): Modelin her katmanındaki nöronların tümü bağlantılıdır.
ReLU: Doğrusal olmayan öğrenmeyi sağlar.
Dropout: Modelin genelleme performansını artırmak için düzenleme (regularization) tekniği.
2. Eğitim Setini Test Verisi Olarak Kullanma
train_and_test_same
Model hem eğitim hem de test için aynı veri setini kullanır.
Eğitim:
train_model: Model, verilen tensorler (X_tensor, y_tensor) üzerinde eğitilir.
Değerlendirme:
evaluate_model: Model, aynı veri setinde değerlendirilir ve doğruluk oranı hesaplanır.

3. Hiperparametre Araması
train_and_test_hyperparameter_search
Bu fonksiyon, modelin en iyi performansını veren hiperparametreleri bulmak için sistematik bir deneme yapar.
Hiperparametreler:
hidden_dim: Gizli katmandaki nöron sayıları (ör., 32, 64, 128).
dropout_rate: Dropout oranları (ör., %20, %30, %50).
lr: Öğrenme oranı (ör., 0.001, 0.0005, 0.0001).
Süreç:
1.Parametre Kombinasyonları:
itertools.product: Tüm parametre kombinasyonları oluşturulur.
2.Model Eğitimi ve Değerlendirme:
Her kombinasyon için model eğitilir ve doğruluk hesaplanır.
4. Performans Değerlendirme
Doğruluk (Accuracy):
Eğitim setindeki tahminlerin doğruluğunu ölçmek için kullanılır.
En İyi Model Performansı
Metric	Class 0 (Negatif)	Class 1 (Pozitif)	Overall
Precision	0.95	0.97	0.96
Recall	0.97	0.95	0.96
F1-Score	0.96	0.96	0.96
Support	1111	1111	2222






En İyi Model Parametreleri ve Performansı
Bu tablo, hiperparametre araması sonucunda elde edilen en iyi modelin parametrelerini ve performansını göstermektedir.
Özellik	Değer
Hidden Dim	128
Dropout Rate	0.2
Learning Rate	0.001
Accuracy	0.9608
Precision (0)	0.95
Recall (0)	0.97
F1-Score (0)	0.96
Precision (1)	0.97
Recall (1)	0.95
F1-Score (1)	0.96
Support (0)	1111
Support (1)	1111


Konfüzyon Matrisi:
	Tahmin Edilen 0	Tahmin Edilen 1
Gerçek 0	1078	33
Gerçek 1	56	1055


Hiperparametre Değişkenliği ve Doğruluk Sonuçları
Aşağıdaki tablo, farklı hiperparametre kombinasyonları için elde edilen doğruluk oranlarını göstermektedir. Bu, hiperparametre araması sürecinde hangi kombinasyonların en iyi performansı sağladığını görmenizi sağlar.
Deneme	Hidden Dim	Dropout Rate	Learning Rate	Doğruluk (Accuracy)
1	32	0.2	0.001	0.90
2	32	0.3	0.001	0.92
3	32	0.5	0.001	0.89
4	64	0.2	0.001	0.93
5	64	0.3	0.001	0.94
6	64	0.5	0.001	0.91
7	128	0.2	0.001	0.9608
8	128	0.3	0.001	0.95
9	128	0.5	0.001	0.93
10	256	0.2	0.001	0.94
11	256	0.3	0.001	0.92
12	256	0.5	0.001	0.90
...	...	...	...	...
Not: En iyi doğruluk oranını sağlayan kombinasyon hidden_dim=128, dropout_rate=0.2, lr=0.001 şeklindedir

4. En Uygun Ağ Topolojisi
En iyi doğruluk oranını sağlayan modelin ağ topolojisi aşağıdaki gibidir:
Katman	Türü	Girdi Boyutu	Çıkış Boyutu	Aktivasyon Fonksiyonu	Dropout Oranı
fc1	Tam Bağlantılı (Linear)	100	128	ReLU	-
fc2	Tam Bağlantılı (Linear)	128	3	-	0.2




 
5-Fold Cross Validation:
Bu yöntemle, veri kümesi 5 eşit parçaya bölünür ve her parça sırayla test verisi olarak kullanılırken, diğer dört parça eğitim verisi olarak kullanılmıştır. Bu işlem toplamda 5 kez tekrarlanır.

1. Sinir Ağı Yapısı (Model Tanımı)
Model: MediumSentimentClassifier sınıfı, bir ileri beslemeli yapay sinir ağıdır (Feedforward Neural Network). Model aşağıdaki bileşenlere sahiptir:
Tam Bağlantılı Katmanlar (Fully Connected Layers):
Üç katman (giriş, gizli, çıkış) kullanılmıştır:
fc1: Girişten gizli katmana.
fc2: Gizli katmandan daha küçük bir gizli katmana.
fc3: Gizli katmandan çıkışa.
Aktivasyon Fonksiyonu:
ReLU (Rectified Linear Unit), negatif değerleri sıfıra eşitleyerek doğrusal olmayanlığı sağlar.
Dropout:
%20-%50 arasında rastgele nöronları sıfırlayarak aşırı öğrenmeyi (overfitting) önler.

2. 5-Fold Cross Validation (Çapraz Doğrulama)
Amaç: Veriyi 5 katmana böler, her katmanda model eğitilir ve test edilir. Her bir katmanda farklı bir veri dilimi test olarak kullanılır.
KFold ile veri kümesi 5 eşit parçaya bölünür. Tüm parçalar sırayla test verisi olarak seçilir.
Fold Başarıları: Her katmandaki doğruluk oranı hesaplanır ve ortalaması alınır.

3. Hiperparametre Araması
Parametre Kombinasyonları: hidden_dim, dropout_rate, lr (öğrenme oranı) gibi hiperparametreler denenir.
Gizli Katman Boyutu (hidden_dim): Katmandaki nöron sayısını belirler (64, 128, 256).
Dropout Oranı (dropout_rate): Dropout oranını kontrol eder (%20, %30, %50).
Öğrenme Oranı (lr): Öğrenme hızını ayarlar (0.001, 0.0005, 0.0001).
En İyi Parametreleri Bulma:
5-fold cross validation sonucunda ortalama doğruluğu en yüksek olan modelin parametreleri seçilir.

4. Model Eğitimi
train_model fonksiyonu, modeli eğitmek için aşağıdaki PyTorch bileşenlerini kullanır:
Optimizasyon: torch.optim.Adam kullanılmıştır.
Adam algoritması, öğrenme oranını otomatik olarak uyarlayan bir optimizasyon yöntemidir.
Kayıp Fonksiyonu: Muhtemelen CrossEntropyLoss, sınıflandırma problemleri için uygundur.
Eğitim sırasında her parametre kombinasyonu için model yeniden oluşturulur.
Kullanılan Teknikler:
Adam optimizasyon algoritması.
Kayıp fonksiyonu olarak çapraz entropi (CrossEntropyLoss).

5. Performans Değerlendirme
Doğruluk (Accuracy): Her fold’un doğruluğu hesaplanır.
Konfüzyon Matrisi:
Tahmin edilen sınıflar ve gerçek sınıflar arasındaki ilişkiyi gösterir.
confusion_matrix ve ConfusionMatrixDisplay kullanılarak görselleştirilmiştir.
Kullanılan Teknikler:
Doğruluk metrikleri (accuracy).
Konfüzyon matrisi ile sonuç görselleştirme.

6. Sonuçların Görselleştirilmesi
Matplotlib kullanılarak en iyi modelin konfizyon matrisi çizilmiştir





Görselleştirme 
Model Yapısı:
MediumSentimentClassifier, girişten çıkışa doğru üç tam bağlantılı (fully connected) katmandan oluşur.
fc1: Giriş -> gizli katman.
fc2: Gizli katman -> daha küçük bir gizli katman.
fc3: Gizli katman -> çıkış.
Aktivasyon Fonksiyonu:
ReLU (Rectified Linear Unit), doğrusal olmayan dönüşüm sağlar.
Dropout:
%20 oranında nöronları rastgele sıfırlayarak aşırı öğrenmeyi (overfitting) önler.
Kullanılan Teknikler:
Tam bağlantılı katmanlar (Fully Connected Layers): Ağın her bir katmanındaki bağlantılar tamdır.
Aktivasyon fonksiyonu (ReLU): Modelin doğrusal olmayan ilişkileri öğrenmesini sağlar.
Dropout: Rastgele nöronları devre dışı bırakarak genel performansı artırır.

________________________________________
Başarı Sonuçları:
Her fold için doğruluk (accuracy) hesaplanır ve ortalama doğruluk değeri alınır. 5-Fold Cross Validation, modelin genelleme yeteneğini ölçmek için etkili bir yöntemdir.

1. Performans Metrikleri Tabloları
Classification Report (Sınıf Bazlı Performans Metrikleri)
Metric	Class 0 (Negatif)	Class 1 (Pozitif)	Overall
Precision	0.75	0.63	0.69 (Macro Avg)
Recall	0.54	0.81	0.68 (Macro Avg)
F1-Score	0.63	0.71	0.67 (Macro Avg)
Support	225	219	444
Ortalama Doğruluk:
Accuracy: 0.68
2. En İyi Hiperparametreler ve Performans
Bu tablo, en iyi modelin hiperparametrelerini ve doğruluk oranını özetler.
Özellik	Değer
Hidden Dim	256
Dropout Rate	0.2
Learning Rate	0.001
Accuracy (5-Fold)	0.6670
En İyi Doğruluk	0.6670
Output Dim	3
Input Dim	100

3. Model Topolojisi
En iyi hiperparametrelerle eğitilen modelin topolojisi aşağıdaki gibidir:
Katman	Türü	Girdi Boyutu	Çıkış Boyutu	Aktivasyon Fonksiyonu	Dropout Oranı
fc1	Tam Bağlantılı (Linear)	100	256	ReLU	-
BatchNorm1d	Batch Normalization	256	256	-	-
fc2	Tam Bağlantılı (Linear)	256	128	ReLU	0.2
BatchNorm1d	Batch Normalization	128	128	-	-
fc3	Tam Bağlantılı (Linear)	128	2	Softmax	-

 
 





10-Fold Cross Validation:
Bu yöntem, 5-Fold Cross Validation’a benzer şekilde çalışır ancak veri 10 eşit parçaya bölünür. Her seferinde farklı bir fold test verisi olarak kullanılır ve modelin performansı ölçülür.
1. Sinir Ağı Tanımı
ComplexSentimentClassifier, ileri beslemeli bir sinir ağıdır (Feedforward Neural Network).
Katmanlar ve Özellikler:
Tam Bağlantılı Katmanlar (Fully Connected Layers):
Dört adet katman:
fc1: Girişten gizli katmana.
fc2: Gizli katmandan daha küçük bir gizli katmana.
fc3: Gizli katmandan daha küçük bir gizli katmana.
fc4: Gizli katmandan çıkışa.
Aktivasyon Fonksiyonları:
ReLU: Sıfırın altındaki değerleri sıfıra çeker.
Leaky ReLU: Negatif değerlerin bir kısmını koruyarak gradyan kaybını önler.
Dropout:
%30-%50 arasında nöronları rastgele devre dışı bırakarak aşırı öğrenmeyi (overfitting) azaltır.
Kullanılan Teknikler:
Tam Bağlantılı Katmanlar: Sinir ağının tüm nöronları birbirine bağlıdır.
ReLU ve Leaky ReLU: Doğrusal olmayan öğrenmeyi sağlar.
Dropout: Regularization tekniği olarak aşırı öğrenmeyi önler.
________________________________________
2. 10-Fold Cross Validation (Çapraz Doğrulama)
ten_fold_cv fonksiyonu, veriyi 10 eşit parçaya böler ve her parça sırasıyla test verisi olarak seçilir.
Aşamalar:
Veri Bölme:
KFold ile veri 10 eşit parçaya ayrılır.
Her adımda farklı bir parça test, geri kalan parçalar eğitim verisi olarak kullanılır.
Model Eğitimi ve Değerlendirme:
Model, train_model ile eğitilir.
Model, evaluate_model ile test edilir ve doğruluk hesaplanır.
Kullanılan Teknikler:
10-Fold Cross Validation: Eğitim ve test seti kombinasyonlarının genelleme yeteneğini artırır.
Fold Bazlı Performans Analizi: Her fold için doğruluk oranı hesaplanır ve ortalama doğruluk alınır.
________________________________________
3. Hiperparametre Araması
ten_fold_cv_hyperparameter_search fonksiyonu, farklı hiperparametre kombinasyonlarını test ederek en iyi modeli belirler.
Süreç:
1.Hiperparametre Kombinasyonları:
product ile tüm kombinasyonlar oluşturulur.
2.10-Fold Çapraz Doğrulama:
Her kombinasyon için model eğitilir ve doğruluk hesaplanır.
3.En İyi Parametreleri Seçme:
Ortalama doğruluğu en yüksek olan model ve parametreler kaydedilir.
Kullanılan Teknikler:
Grid Search: Parametre kombinasyonlarını sistematik bir şekilde deneme.
Çapraz Doğrulama ile Performans Analizi: Her parametre kombinasyonunun doğruluğu hesaplanır.
________________________________________
4. Model Performansının Değerlendirilmesi
Konfüzyon Matrisi:
confusion_matrix ile gerçek ve tahmin edilen sınıflar arasındaki ilişki hesaplanır.
ConfusionMatrixDisplay kullanılarak görselleştirilir.
Performans Metrikleri:
Doğruluk (accuracy), her fold’un doğruluğu ve en iyi parametrelerin genel doğruluğu raporlanır.
________________________________________
5. Eğitim ve Değerlendirme Fonksiyonları
Model Eğitimi:
train_model fonksiyonu ile PyTorch üzerinde model eğitimi yapılır.
Optimizasyon: AdamW optimizasyon algoritması.
Kayıp Fonksiyonu: Muhtemelen çapraz entropi veya focal loss.
Model Değerlendirme:
evaluate_model fonksiyonu ile doğruluk hesaplanır ve değerlendirme yapılır.
________________________________________
Genel Kullanılan Teknikler:
Derin Öğrenme Modelleri: Tam bağlantılı katmanlar, dropout, ReLU ve Leaky ReLU.
Çapraz Doğrulama: Genelleme yeteneğini ölçmek için.
Hiperparametre Optimizasyonu: Grid search yöntemi.
Performans Analizi: Konfüzyon matrisi, doğruluk hesaplamaları.
Görselleştirme: Matplotlib ve Seaborn ile analiz görselleştirme.

10 fold görselleştirme 
1. Sinir Ağı Tanımı
ComplexSentimentClassifier, dört tam bağlantılı katmandan oluşan bir ileri beslemeli sinir ağıdır (Feedforward Neural Network).
Katmanlar:
fc1, fc2, fc3, fc4:
Tam bağlantılı katmanlar (Fully Connected Layers).
Girişten çıkışa doğru gizli katman sayısı kademeli olarak azaltılmıştır:
hidden_dim, hidden_dim//2, hidden_dim//4, ve çıkış katmanı (output_dim).
Aktivasyon Fonksiyonları:
ReLU (Rectified Linear Unit):
Doğrusal olmayan dönüşüm sağlar. Negatif girdileri sıfıra çeker.
Leaky ReLU:
Negatif değerlerin küçük bir kısmını (slope=0.01) koruyarak gradyan kaybını önler.
Dropout:
%50 oranında rastgele nöronları devre dışı bırakarak aşırı öğrenmeyi (overfitting) önler.
Kullanılan Teknikler:
Tam Bağlantılı Katmanlar: Sinir ağı nöronlarının tümü birbirine bağlanmıştır.
ReLU ve Leaky ReLU: Doğrusal olmayan öğrenmeyi sağlar.
Dropout: Aşırı öğrenmeyi engellemek için düzenleme (regularization) tekniği.
________________________________________
3. Modelin Görselleştirilmesi
Torchviz ile Ağ Grafiği:
make_dot fonksiyonu: Modelin yapısını ve parametrelerini içeren bir ağ grafiği oluşturur.
Parametreler: model.named_parameters() kullanılarak modeldeki ağırlıklar ve eğitilebilir parametreler görselleştirilmiştir.
Sonuç: graph.render() ile grafik bir dosya (PNG) olarak kaydedilmiştir.
Matplotlib ile Parametre Görselleştirme:
Modelin parametreleri ve en iyi doğruluk değeri metin olarak görselleştirilmiştir:
Giriş ve Çıkış Boyutları (input_dim, output_dim).
Dropout Oranı (dropout_rate).
Gizli Katman Boyutu (hidden_dim).
En iyi doğruluk değeri (0.6692).
________________________________________
4. Performans Raporlama
En İyi Doğruluk: Eğitim sürecinden elde edilen en iyi doğruluk değeri (0.6692) görselleştirme sırasında belirtilmiştir.
________________________________________





Başarı Sonuçları:
10-Fold Cross Validation, modelin genelleme performansını daha güvenilir bir şekilde değerlendirir çünkü model her seferinde farklı veri dilimlerine test edilir.
1. Performans Metrikleri Tablolaştırma
Metric	Class 0 (Negatif)	Class 1 (Pozitif)	Overall
Precision	0.78	0.69	0.73 (Macro Avg)
Recall	0.58	0.85	0.71 (Macro Avg)
F1-Score	0.66	0.76	0.71 (Macro Avg)
Support	106	116	222
Ortalama Doğruluk (Accuracy):
0.72

2.En Uygun Ağ Topolojisi
Katman	Türü	Girdi Boyutu	Çıkış Boyutu	Aktivasyon Fonksiyonu	Dropout Oranı
fc1	Tam Bağlantılı (Linear)	100	256	ReLU	-
fc2	Tam Bağlantılı (Linear)	256	128	Leaky ReLU	0.5
fc3	Tam Bağlantılı (Linear)	128	64	Leaky ReLU	0.5
fc4	Tam Bağlantılı (Linear)	64	3	-	-

________________________________________




3. En İyi Hiperparametreler
Özellik	Değer
Hidden Dim	256
Dropout Rate	0.5
Learning Rate	0.001
Accuracy (10-Fold)	0.6692
En İyi Doğruluk	0.6692
Output dim	3
Input dim	100



 
 

%66-%34 Eğitim Test Ayırarak (5 Farklı Rastgele Ayırma ile)
Bu yöntemde, veri seti %66 eğitim ve %34 test oranıyla rastgele iki parçaya ayrılır. Bu işlem 5 farklı kez tekrarlanır, her seferinde verinin farklı rastgele bir bölümü eğitim/test olarak ayrılır.

Bu kod, bir sinir ağı modeli oluşturmak, eğitmek, hiperparametre optimizasyonu yapmak ve %66-%34 eğitim-test ayırma yöntemini kullanarak performans değerlendirmesi yapmak için hazırlanmıştır. Aşağıda kullanılan teknikler ve yöntemlerin detaylı açıklamaları verilmiştir:
________________________________________
1. Sinir Ağı Tanımı
RandomSplitClassifier
Bu model, Batch Normalization ve GELU aktivasyon fonksiyonları gibi modern düzenleme ve aktivasyon tekniklerini kullanır.
Katmanlar:
fc1, fc2, fc3: Üç tam bağlantılı katman (Fully Connected Layers).
Batch Normalization: bn1 ve bn2 ile gizli katmanlara batch normalization uygulanmıştır. Bu teknik, her katmandaki girdilerin dağılımını normalize ederek öğrenme sürecini hızlandırır ve modelin genelleme yeteneğini artırır.
Aktivasyon Fonksiyonu:
GELU (Gaussian Error Linear Unit): GELU, doğrusal olmayan bir aktivasyon fonksiyonudur. ReLU’ya kıyasla daha pürüzsüz bir dönüşüm sağlar ve negatif değerlerin bazılarını geçirme eğilimindedir.
Dropout:
Dropout, %30-%50 oranında nöronları rastgele devre dışı bırakarak aşırı öğrenmeyi (overfitting) önler.
Kullanılan Teknikler:
Batch Normalization: Öğrenme sürecini stabilize eder, genelleme yeteneğini artırır.
GELU: Yumuşak ve doğrusal olmayan bir aktivasyon fonksiyonu.
Dropout: Aşırı öğrenmeyi azaltmak için düzenleme (regularization) tekniği.


2. Eğitim ve Test Ayırma
random_split_evaluation
%66-%34 oranında eğitim ve test ayırma işlemi gerçekleştirilir. Bu işlem, 5 farklı rastgele ayırma ile tekrar edilir ve sonuçların ortalaması alınır.
Veri Ayırma:
train_test_split: %66 eğitim, %34 test oranıyla veri ayrılır. Rastgelelik kontrolü için random_state kullanılır.
Performans Değerlendirme:
Model, eğitim verisi üzerinde eğitilir (train_model) ve test verisi üzerinde değerlendirilir (evaluate_model).
Her bir rastgele ayırma için doğruluk oranı hesaplanır ve ortalama doğruluk bulunur.
Kullanılan Teknikler:
Rastgele Eğitim-Test Ayırma: Farklı rastgele bölmelerle modelin genelleme yeteneğini test etmek için.
Doğruluk Metrikleri (Accuracy): Test verisi üzerinde doğruluk oranı hesaplanır.
________________________________________
3. Hiperparametre Araması
random_split_hyperparameter_search
Farklı hiperparametre kombinasyonları için %66-%34 eğitim-test ayırma işlemi gerçekleştirilir.
Kullanılan Teknikler:
Grid Search: Sistematik olarak tüm hiperparametre kombinasyonlarını test etme.
Rastgele Ayırma: Her kombinasyon için 5 farklı eğitim-test bölmesiyle genelleme testi.
________________________________________
4. En İyi Modelin Tekrar Eğitimi
retrain_best_model
Seçilen en iyi hiperparametrelerle model son bir kez eğitilir ve test edilir.
Yeni Eğitim-Test Bölmesi:
Veriler yeniden %66-%34 oranında bölünür.
Model Eğitimi:
train_model: En iyi hiperparametrelerle eğitim.
________________________________________
5. Görselleştirme ve Performans Analizi
Doğruluk Analizi:
Eğitim ve test süreçlerinden elde edilen doğruluk oranları raporlanır.
Konfüzyon Matrisi:
confusion_matrix ile oluşturulur ve ConfusionMatrixDisplay ile görselleştirilir.

Başarı Sonuçları:
Her bir ayırma için modelin doğruluğu hesaplanır ve ortalama doğruluk raporlanır. Bu yöntem, gerçek dünya verisiyle modelin nasıl çalışacağını görmek için kullanılır.

4. Uygulama Sonuçları ve Model Performansı
Her bir yöntem için aşağıdaki metrikler hesaplanmış ve raporlanmıştır:
En Uygun Parametre Değerleri:
Gizli Katman Boyutu (Hidden Layer Size): En iyi model parametreleri hidden_dim değeri için 128 veya 256 olarak belirlenmiştir.
Dropout Oranı (Dropout Rate): 0.3 ile 0.4 arasında en iyi doğruluğu sağladı.
Öğrenme Oranı (Learning Rate): 0.0005 ve 0.001 arasında en iyi sonuçlar elde edilmiştir.
En Uygun Ağ Topolojisi:
Gizli Katman Sayısı: 2 veya 3 katman kullanılması en iyi sonuçları verdi.
Aktivasyon Fonksiyonu: GELU ve ReLU arasında seçim yapılmış ve GELU, doğrulukta küçük bir artış sağladı.
Dropout: Aşırı öğrenmeyi engellemek için %30-%40 arasında en iyi sonuçları veren oran seçildi.
Başarı Sonuçları:
Her bir eğitim-test stratejisi için doğruluk oranları hesaplanmış ve en iyi doğruluklar %5-%10 farkla farklı stratejilerde elde edilmiştir. Özellikle 5-Fold ve 10-Fold Cross Validation yöntemleri, modelin genelleme yeteneğini ölçmede etkili olmuştur.



1. Classification Report Tablolaştırma
Metric	Class 0 (Negatif)	Class 1 (Pozitif)	Overall
Precision	0.77	0.61	0.69 (Macro Avg)
Recall	0.40	0.88	0.64 (Macro Avg)
F1-Score	0.53	0.72	0.63 (Macro Avg)
Support	369	387	756
Ortalama Doğruluk (Accuracy):
0.65
________________________________________
3. En İyi Hiperparametreler
Özellik	Değer
Hidden Dim	256
Dropout Rate	0.4
Learning Rate	0.0005
Accuracy (%66-%34)	0.5653

En Uygun Ağ Topolojisi
Katman	Türü	Girdi Boyutu	Çıkış Boyutu	Aktivasyon Fonksiyonu	Dropout Oranı
fc1	Tam Bağlantılı (Linear)	100	256	GELU	-
bn1	Batch Normalization	256	256	-	-
fc2	Tam Bağlantılı (Linear)	256	128	GELU	0.4
bn2	Batch Normalization	128	128	-	-
fc3	Tam Bağlantılı (Linear)	128	3	-	

Konfüzyon Matrisi:
Modelin test sonuçları konfüzyon matrisi ile görselleştirilmiş ve doğru/yanlış sınıflandırmalar arasındaki ilişkiler analiz edilmiştir. Konfüzyon matrisi, modelin hangi sınıfları doğru sınıflandırıp hangilerini yanlış sınıflandırdığını gösterir.


 
 

5. Sonuçlar ve Gelecek Çalışmalar
Bu projede yapılan testler ve modellerin değerlendirilmesi sonucunda:
En başarılı model, 10-Fold Cross Validation ile doğrulanan ve en uygun hiperparametrelerle eğitilen modeldir.
Gelecekte, BERT veya GPT-2/3 gibi modern dil modelleri ile performans iyileştirmeleri yapılabilir.



