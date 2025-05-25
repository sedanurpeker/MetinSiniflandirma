import os
import zipfile
import pandas as pd
import nltk
import string
import re
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------------------------------------------
# Veri Seti Kaynağı:
# https://www.kaggle.com/datasets/shivamkushwaha/bbc-full-text-document-classification
# BBC News Dataset, 2225 adet kısa haber metninden oluşur.
# Kategoriler: business, entertainment, politics, sport, tech
# -------------------------------------------------------


# 1. ZIP dosyasını açma
zip_path = r"C:\Users\sedan\Downloads\archive (1).zip"
extract_path = r"C:\Users\sedan\Downloads\bbc"

if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

# 2. Veri setini okuma
data = []
root_dir = os.path.join(extract_path, "bbc")

for category in os.listdir(root_dir):
    category_path = os.path.join(root_dir, category)
    if os.path.isdir(category_path):
        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            with open(file_path, 'r', encoding='latin1') as f:
                text = f.read()
                data.append({'category': category, 'text': text})

df = pd.DataFrame(data)

# 3. Ön işleme
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

df['clean_text'] = df['text'].apply(preprocess)

# 4. Özellik çıkarımı (TF-IDF)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['category']

# 5. Eğitim ve test kümeleri
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Modeller
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(kernel='linear'),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# 7. Eğitim, tahmin ve sonuçları saklama
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'accuracy': acc,
        'conf_matrix': confusion_matrix(y_test, y_pred),
        'report': classification_report(y_test, y_pred, output_dict=True)
    }
    print(f"\n========== {name} ==========")
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
# 8. Confusion matrix görselleştirme
def plot_conf_matrix(y_true, y_pred, title, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

labels = sorted(df['category'].unique())
for name, data in results.items():
    plot_conf_matrix(y_test, data['y_pred'], f"{name} Confusion Matrix", labels)

# 9. Accuracy karşılaştırma grafiği
accuracies = {name: data['accuracy'] for name, data in results.items()}

plt.figure(figsize=(8,5))
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette="viridis")
plt.ylim(0.8, 1.0)
plt.ylabel("Accuracy")
plt.title("Model Performans Karşılaştırması")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
