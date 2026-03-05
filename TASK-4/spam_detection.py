import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==============================
# 1. Load Dataset
# ==============================

# If dataset name is spam.csv
df = pd.read_csv("spam.csv", encoding='latin-1')

# Some datasets have extra unnamed columns — remove them
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

print("First 5 rows:")
print(df.head())

# ==============================
# 2. Convert Labels to Numeric
# ==============================

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# ==============================
# 3. Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    df['message'],
    df['label'],
    test_size=0.2,
    random_state=42
)

# ==============================
# 4. TF-IDF Vectorization
# ==============================

vectorizer = TfidfVectorizer(stop_words='english')

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ==============================
# 5. Train Model (Naive Bayes)
# ==============================

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# ==============================
# 6. Predictions
# ==============================

y_pred = model.predict(X_test_tfidf)

# ==============================
# 7. Evaluation
# ==============================

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ==============================
# 8. Test Custom Message
# ==============================

def predict_message(msg):
    msg_tfidf = vectorizer.transform([msg])
    prediction = model.predict(msg_tfidf)[0]
    
    if prediction == 1:
        return "SPAM"
    else:
        return "HAM (Not Spam)"

print("\nCustom Test:")
print(predict_message("Congratulations! You won a free ticket. Call now!"))
