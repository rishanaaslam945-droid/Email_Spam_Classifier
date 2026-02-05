# 📧 Email Spam Classifier using Machine Learning

A Machine Learning–based Email Spam Classifier that accurately classifies messages as **Spam** or **Ham** using **Natural Language Processing (NLP)** and **TF-IDF features**.

## 📌 Project Overview

Spam emails are a major problem in digital communication.  
This project builds a scalable spam filtering system using NLP techniques and supervised machine learning.

The model analyzes the text content of emails and predicts whether the message is spam or not.

## 🧠 Technologies Used

- Python
- Natural Language Toolkit (NLTK)
- Scikit-learn
- Pandas, NumPy
- Matplotlib
- TF-IDF Vectorization
- Naive Bayes Classifier

## 🔍 Dataset

- SMS Spam Collection Dataset
- Labels:
  - **0 → Ham (Not Spam)**
  - **1 → Spam**

## ⚙️ Workflow

1. Data Loading and Exploration  
2. Text Preprocessing  
   - Lowercasing  
   - Removing punctuation  
   - Stopword removal  
3. Feature Extraction using TF-IDF  
4. Model Training (Multinomial Naive Bayes)  
5. Model Evaluation  
6. Visualization of Top Spam Words  
7. Real-time Message Prediction  

## 📊 Model Performance

- **Accuracy:** ~97%
- **High precision and recall**
- Confusion Matrix and Classification Report used for evaluation

## 📈 Visualization

- Top 10 spam-related words visualized using bar charts
- Helps in understanding common spam patterns

## 🧪 outcomes

- Successfully trained and validated ML models to classify emails as spam or ham
- Achieved high accuracy and precision using NLP + ML techniques
- Developed a scalable prototype for spam filtering
- Strengthened understanding of text classification and data preprocessing
- Can be extended to detect phishing emails, scams, or advertisements

## 🔮 Future Enhancements

- Phishing email detection
- Deep Learning models (LSTM / BERT)
- Web-based interface using Flask or Streamlit
- Multi-language spam detection

## ▶️ How to Run the Project

bash
pip install -r requirements.txt
python spam_classifier.py

👩‍🎓 Author

Rishana Aslam
Department of Artificial Intelligence & Data Science



