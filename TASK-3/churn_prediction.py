import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# ====================================
# 1. LOAD DATA
# ====================================

print("Loading dataset...")

df = pd.read_csv("Churn_Modelling.csv")

print("Dataset Loaded Successfully!")
print("Shape:", df.shape)


# ====================================
# 2. DATA CLEANING
# ====================================

# Drop unnecessary columns
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# Encode categorical columns
label_encoder = LabelEncoder()

df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Geography'] = label_encoder.fit_transform(df['Geography'])

# Target variable
df.rename(columns={'Exited': 'Churn'}, inplace=True)


# ====================================
# 3. SPLIT FEATURES & TARGET
# ====================================

X = df.drop('Churn', axis=1)
y = df['Churn']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ====================================
# 4. MODEL DEFINITIONS
# ====================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

best_model = None
best_auc = 0
best_name = ""


# ====================================
# 5. TRAIN & EVALUATE
# ====================================

for name, model in models.items():
    print(f"\n========== Training {name} ==========")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    auc = roc_auc_score(y_test, y_prob)
    print("ROC-AUC Score:", auc)

    if auc > best_auc:
        best_auc = auc
        best_model = model
        best_name = name


# ====================================
# 6. SAVE BEST MODEL
# ====================================

joblib.dump(best_model, "best_churn_model.pkl")

print(f"\nBest Model: {best_name}")
print("Best ROC-AUC:", best_auc)
print("Model saved as best_churn_model.pkl")


# ====================================
# 7. ROC CURVE
# ====================================

y_prob = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - {best_name}")
plt.show()

print("\nCustomer Churn Prediction Completed Successfully!")
