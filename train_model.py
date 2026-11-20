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

# File paths
DATA_FILE = "dataset/gestures.csv"
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "gesture_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset
print("Loading dataset...")
df = pd.read_csv(DATA_FILE)

# Validate dataset
if "label" not in df.columns:
    raise ValueError("Dataset must contain a 'label' column.")

df = df.dropna()

X = df.drop(columns=["label"])
y = df["label"]

if y.nunique() < 2:
    raise ValueError("At least two gesture classes are required.")

print("\nDataset shape:", df.shape)
print("\nSamples per class:")
print(y.value_counts())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train model
print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=model.classes_))
print("\nClass order:", list(model.classes_))

# Save trained model
joblib.dump(model, MODEL_FILE)

print(f"\nModel saved successfully: {MODEL_FILE}")
print("Training complete!")