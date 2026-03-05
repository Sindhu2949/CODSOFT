import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# =====================================
# 1. LOAD DATA
# =====================================

print("Loading dataset...")

df = pd.read_csv("fraudTrain.csv")

print("Dataset Loaded Successfully!")
print("Original Shape:", df.shape)


# =====================================
# 2. FEATURE SELECTION (Memory Safe)
# =====================================

selected_columns = [
    'amt',
    'lat',
    'long',
    'city_pop',
    'unix_time',
    'merch_lat',
    'merch_long',
    'is_fraud'
]

df = df[selected_columns]

print("Reduced Shape:", df.shape)


# =====================================
# 3. SPLIT FEATURES & TARGET
# =====================================

X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =====================================
# 4. MODEL DEFINITIONS (BALANCED)
# =====================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=500,
        class_weight="balanced"
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
}

best_model = None
best_auc = 0
best_name = ""


# =====================================
# 5. TRAIN & EVALUATE
# =====================================

for name, model in models.items():
    print(f"\n========== Training {name} ==========")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    auc = roc_auc_score(y_test, y_prob)
    print("ROC-AUC Score:", auc)

    if auc > best_auc:
        best_auc = auc
        best_model = model
        best_name = name


# =====================================
# 6. SAVE BEST MODEL
# =====================================

joblib.dump(best_model, "best_fraud_model.pkl")

print(f"\nBest Model: {best_name}")
print("Best ROC-AUC:", best_auc)
print("Best model saved as best_fraud_model.pkl")


# =====================================
# 7. ROC CURVE
# =====================================

y_prob = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - {best_name}")
plt.show()

print("\nProject Completed Successfully!")
