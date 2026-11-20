import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =====================================
# KINESIS-ISL SEQUENCE MODEL TRAINING
# =====================================

DATASET_FILE = "dataset/gesture_sequences_fresh.csv"
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "sequence_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Load dataset
print("\nLoading gesture sequences...")

df = pd.read_csv(DATASET_FILE)

print("Dataset shape:", df.shape)

# 2. Separate features and labels
X = df.drop(columns=["label"])
y = df["label"]

print("\nGesture distribution:")
print(y.value_counts())

# 3. Check dataset
if df.isnull().sum().sum() > 0:
    raise ValueError(
        "Dataset contains missing values. "
        "Please check the CSV before training."
    )

if len(df) < 50:
    raise ValueError("Not enough sequences to train reliably.")

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining sequences:", len(X_train))
print("Testing sequences:", len(X_test))

# 5. Create sequence-feature classifier
print("\nTraining model...")

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 6. Evaluate model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n========== RESULTS ==========")
print(f"Test accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=model.classes_,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions,
        labels=model.classes_
    )
)

# 7. Save model separately
joblib.dump(model, MODEL_FILE)

print("\nModel saved successfully!")
print("Model path:", MODEL_FILE)

print("\nClasses:", list(model.classes_))