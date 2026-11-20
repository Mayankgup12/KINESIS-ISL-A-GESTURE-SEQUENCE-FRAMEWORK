import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the existing trained model
model = joblib.load("models/gesture_model.pkl")

# Load the new session dataset
df = pd.read_csv("dataset/gestures_session2.csv")

# Separate features and labels
X = df.drop(columns=["label"])
y = df["label"]

# Match feature columns with the trained model
X = X[model.feature_names_in_]

# Predict without retraining
y_pred = model.predict(X)

# Evaluation results
accuracy = accuracy_score(y, y_pred)

print("\n===== KINESIS-ISL: SESSION 2 EVALUATION =====")
print(f"\nTotal test samples: {len(y)}")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y, y_pred, labels=model.classes_))

print("\nClass order:")
print(model.classes_)