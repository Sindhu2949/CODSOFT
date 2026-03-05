import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# =========================
# LOAD DATA
# =========================

rows = []

with open("train_data.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(" ::: ")
        
        if len(parts) == 4:
            movie_id = parts[0]
            title = parts[1]
            genre = parts[2]
            description = parts[3]
            
            rows.append([movie_id, title, genre, description])

df = pd.DataFrame(rows, columns=["id", "title", "genre", "description"])

print("Total rows loaded:", len(df))
print("Unique genres:", df["genre"].nunique())

# =========================
# TRAIN TEST SPLIT
# =========================

X = df["description"]
y = df["genre"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# IMPROVED TF-IDF
# =========================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=15000,      # Increased features
    ngram_range=(1,2),       # Unigrams + Bigrams
    min_df=3                 # Remove rare words
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# =========================
# STRONGER MODEL
# =========================

model = LogisticRegression(
    max_iter=2000,
    C=2,
    n_jobs=-1
)

model.fit(X_train_tfidf, y_train)

# =========================
# PREDICT
# =========================

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))
