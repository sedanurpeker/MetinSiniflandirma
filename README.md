# Makine Öğrenmesi ile Metin Sınıflandırma

Bu proje, **BBC Full Text Document Classification** veri setini kullanarak haber metinlerini otomatik olarak kategorilere ayırmayı amaçlamaktadır.  
Metinler üzerinde **veri temizleme**, **tokenizasyon** ve **TF-IDF özellik çıkarımı** uygulanmış; ardından dört farklı makine öğrenmesi algoritması ile sınıflandırma yapılmıştır.

---

## Proje Amacı
- İngilizce kısa haber metinlerini **Spor**, **Ekonomi**, **Magazin/Eğlence**, **Gündem/Siyaset**, **Teknoloji** kategorilerine otomatik olarak ayırmak.
- Farklı algoritmaların performansını karşılaştırmak.
- Yüksek doğruluk sağlayan ölçeklenebilir bir metin sınıflandırma sistemi geliştirmek.

---

## Kullanılan Veri Seti
- **Kaynak:** [BBC Full Text Document Classification – Kaggle](https://www.kaggle.com/datasets/shivamkushwaha/bbc-full-text-document-classification)
- **Toplam Veri:** 2225 kısa haber metni
- **Kategoriler:**
  - Business (Ekonomi)
  - Entertainment (Magazin / Eğlence)
  - Politics (Gündem / Siyaset)
  - Sport (Spor)
  - Tech (Teknoloji)
- **Avantajlar:**
  - Kısa ve tek konulu haberler → Net sınıflandırma
  - 5 farklı sınıf ile çok sınıflı karşılaştırma imkanı
  - İngilizce dil desteği ile NLP kütüphaneleriyle uyumlu

---

##  Kullanılan Yöntemler
### 1. Veri Ön İşleme
- Küçük harfe dönüştürme
- Noktalama işaretleri ve sayıları kaldırma
- Stopword temizliği
- Boşluk ve satır karakterlerini temizleme

### 2. Özellik Çıkarımı
- TF-IDF (Term Frequency – Inverse Document Frequency)
- Metinleri sayısal vektörlere dönüştürerek modele uygun hale getirme

### 3. Kullanılan Algoritmalar
- **Multinomial Naive Bayes (MNB)**
- **Logistic Regression (LR)**
- **Support Vector Machine (SVM)**
- **Random Forest (RF)**

---

## Performans Sonuçları
| Model                     | Doğruluk (%) |
|---------------------------|--------------|
| Support Vector Machine    | **97.30**    |
| Logistic Regression       | 96.63        |
| Random Forest             | 96.40        |
| Multinomial Naive Bayes   | 96.18        |

**SVM**, en yüksek doğruluk oranını elde etmiştir. Logistic Regression da dengeli performansı ile öne çıkmıştır.

---

## Kurulum
1. Gerekli kütüphaneleri yükleyin:
  ```bash
  pip install pandas numpy scikit-learn
  ```
2. Veri setini indirin ve proje klasörüne ekleyin.
3. Modeli çalıştırın:
  ```bash
  python main.py
  ```
## Kaynakça
- BBC Full Text Document Classification Dataset – Kaggle
- Géron, A. – Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow
- Jurafsky, D., Martin, J. H. – Speech and Language Processing

## Geliştirici
**Sedanur PEKER**

**İletişim:** sedanurpeker05@gmail.com
