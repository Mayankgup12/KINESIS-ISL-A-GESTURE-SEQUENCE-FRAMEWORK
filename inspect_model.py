import joblib
import pandas as pd

# Load the existing model
model = joblib.load("models/gesture_model.pkl")

# Load original dataset
df = pd.read_csv("dataset/gestures.csv")

print("\n===== MODEL INSPECTION =====")

print("\nModel classes:")
print(model.classes_)

print("\nNumber of features expected:")
print(model.n_features_in_)

print("\nOriginal dataset feature count:")
print(df.drop(columns=["label"]).shape[1])

print("\nModel feature names:")
print(model.feature_names_in_)

print("\nDataset labels:")
print(sorted(df["label"].unique()))

print("\nDataset data types:")
print(df.dtypes.value_counts())

print("\nFirst 5 rows of original dataset:")
print(df.head())
from sklearn.metrics import accuracy_score, classification_report

X = df.drop(columns=["label"])
y = df["label"]

# Match the feature names expected by the model
X.columns = model.feature_names_in_

predictions = model.predict(X)

print("\n===== ORIGINAL DATASET CHECK =====")
print(f"Accuracy: {accuracy_score(y, predictions) * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y, predictions, zero_division=0))