import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
nltk.download('stopwords')
data = pd.read_csv("spam.csv", encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']
print("\n==== Original Data ====")
print(data.head())
data['label'] = data['label'].map({'ham': 0, 'spam': 1})
print("\n==== Label Encoded Data ====")
print(data.head())
def clean_text(msg):
    msg = msg.lower()  # lowercase
    msg = ''.join([c for c in msg if c not in string.punctuation])  # remove punctuation
    words = msg.split()  # tokenization
    words = [w for w in words if w not in stopwords.words('english')]  # remove stopwords
    return ' '.join(words)
data['message'] = data['message'].apply(clean_text)

print("\n==== Cleaned Data ====")
print(data.head())
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['message'])
y = data['label']
print("\nTF-IDF feature matrix shape:", X.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
print("\n==== Naive Bayes Accuracy ====")
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\n==== Classification Report ====")
print(classification_report(y_test, y_pred))
print("\n==== Confusion Matrix ====")
print(confusion_matrix(y_test, y_pred))
sample_emails = [
    "Congratulations! You won a free iPhone. Click here now!",
    "Hey, are we meeting tomorrow for lunch?",
    "Get your free entry tickets now!",
    "Don't forget to submit the assignment today."
]
sample_cleaned = [clean_text(msg) for msg in sample_emails]
sample_vec = vectorizer.transform(sample_cleaned)
results = nb.predict(sample_vec)
print("\n" + "="*60)
print("🔹 CUSTOM EMAIL TEST RESULTS 🔹")
print("="*60)
for i, email in enumerate(sample_emails):
    if results[i] == 1:
        print(f"\033[91m[SPAM 🚫]: {email}\033[0m")  # red color for spam
    else:
        print(f"\033[92m[HAM ✅]: {email}\033[0m")  # green color for ham
print("\n📊 Summary Table 📊")
print(f"{'Type':<10}{'Count':<10}")
print(f"{'Spam':<10}{sum(results):<10}")
print(f"{'Ham':<10}{len(results)-sum(results):<10}")
print("\n🎯 Model Summary 🎯")
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"Total emails tested: {len(y_test)}")
print(f"Spam detected: {sum(y_pred)} | Ham detected: {len(y_pred)-sum(y_pred)}")
feature_names = vectorizer.get_feature_names_out()
spam_words = nb.feature_log_prob_[1]

top_indices = np.argsort(spam_words)[-10:]
top_words = [feature_names[i] for i in top_indices]
top_values = spam_words[top_indices]

plt.figure(figsize=(10,5))
plt.barh(top_words, top_values, color='orange')
plt.title("🔥 Top 10 Spam Words 🔥")
plt.xlabel("Log Probability")
plt.tight_layout()
plt.show()
