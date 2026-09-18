import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. LOAD CLEAN CROP DATASET
# ==========================================

file_path = "data/processed/crop_recommendation_clean.csv"

df = pd.read_csv(file_path)

print("========== CROP RECOMMENDATION MODEL ==========")

print("Total rows:", len(df))
print("Total columns:", len(df))


# ==========================================
# 2. SELECT INPUT FEATURES
# ==========================================

features = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

X = df[features]

# Target column
y = df["label"]


# ==========================================
# 3. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 4. CREATE RANDOM FOREST MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# ==========================================
# 5. TRAIN MODEL
# ==========================================

print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

print("Training completed!")


# ==========================================
# 6. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 7. EVALUATE MODEL
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL RESULTS ==========")

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ==========================================
# 8. SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "models/crop_recommendation_model.pkl"
)

print("\nModel saved successfully!")

print(
    "Model file: models/crop_recommendation_model.pkl"
)