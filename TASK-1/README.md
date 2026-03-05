# 🎬 Movie Genre Classification using Machine Learning

## 📌 Project Overview

This project builds a Machine Learning model to classify movie genres based on their plot descriptions.

The model uses Natural Language Processing (NLP) techniques such as TF-IDF vectorization and Logistic Regression to predict the genre of a movie from its description text.

---

## 📂 Dataset

The dataset contains movie records in the format:

```
id ::: title ::: genre ::: description
```

Example:

```
1 ::: Edgar's Lunch (1998) ::: thriller ::: L.R. Brane loves his life...
```

### Dataset Statistics:
- Total Movies: 54,214
- Total Genres: 27
- Task Type: Multi-class text classification

---

## 🛠 Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression

---

## ⚙️ Machine Learning Pipeline

### 1️⃣ Data Preprocessing
- Loaded raw text dataset
- Extracted description and genre
- Cleaned and structured into DataFrame

### 2️⃣ Feature Engineering
Used TF-IDF Vectorizer with:
- Stopword removal
- 15,000 max features
- Unigrams and Bigrams
- Minimum document frequency filtering

### 3️⃣ Model Training
Used:
- Logistic Regression
- Stratified Train-Test Split (80/20)

### 4️⃣ Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score

---

## 📊 Model Performance

- Accuracy: ~65%–75% (varies slightly per run)
- Strong performance on:
  - Drama
  - Documentary
  - Comedy
- Lower performance on minority genres due to class imbalance

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install pandas scikit-learn
```

### 2️⃣ Run the Model

```bash
python train_model.py
```

---

## 📈 Improvements Made

- Switched from Naive Bayes to Logistic Regression
- Added bigram features
- Increased TF-IDF feature space
- Used stratified sampling
- Handled zero division warnings

---

## 🎯 Future Improvements

- Use Linear SVM
- Hyperparameter tuning (GridSearchCV)
- Apply class balancing techniques
- Build Streamlit web app for live prediction
- Try Deep Learning (LSTM / BERT)

---

## 💼 Internship Project Summary

This project demonstrates:
- NLP preprocessing
- Text feature engineering
- Supervised learning
- Model evaluation
- Performance optimization

The final model successfully predicts movie genres from textual descriptions using classical ML techniques.

---

## 👤 Author

Your Name  
Machine Learning Intern  