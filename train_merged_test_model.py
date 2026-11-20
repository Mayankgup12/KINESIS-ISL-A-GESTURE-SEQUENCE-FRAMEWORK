import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =====================================
# KINESIS-ISL MERGED MODEL TRAINING
# =====================================

DATASET_FILE = "dataset/gesture_sequences_merged_test.csv"
MODEL_FILE = "models/sequence_model_merged_test.pkl"

RESULTS_DIR = "results"

os.makedirs("models", exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 1. Load dataset
print("\nLoading merged gesture sequences...")

df = pd.read_csv(DATASET_FILE)

print("Dataset shape:", df.shape)

X = df.drop(columns=["label"])
y = df["label"]

print("\nGesture distribution:")
print(y.value_counts().sort_index())

# 2. Validate dataset
if df.isnull().sum().sum() > 0:
    raise ValueError("Dataset contains missing values.")

if X.shape[1] != 1890:
    raise ValueError(f"Expected 1890 features, found {X.shape[1]}")

if y.nunique() != 5:
    raise ValueError("Expected exactly 5 gesture classes.")

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 4. Train Random Forest
print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 5. Evaluate
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

report = classification_report(
    y_test,
    predictions,
    labels=model.classes_,
    zero_division=0
)

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=model.classes_
)

print("\n========== RESULTS ==========")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n")
print(report)
print("\nConfusion Matrix:\n")
print(matrix)

# 6. Save model
joblib.dump(model, MODEL_FILE)

# 7. Save accuracy
with open(os.path.join(RESULTS_DIR, "accuracy.txt"), "w") as file:
    file.write(f"Test Accuracy: {accuracy * 100:.2f}%\n")
    file.write(f"Training Samples: {len(X_train)}\n")
    file.write(f"Testing Samples: {len(X_test)}\n")
    file.write(f"Total Samples: {len(df)}\n")

# 8. Save classification report
with open(
    os.path.join(RESULTS_DIR, "classification_report.txt"),
    "w"
) as file:
    file.write("Kinesis-ISL Model Evaluation\n\n")
    file.write(f"Test Accuracy: {accuracy * 100:.2f}%\n\n")
    file.write(report)

# 9. Save confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=model.classes_
)

display.plot(cmap="Blues", values_format="d")
plt.title("Kinesis-ISL Confusion Matrix")
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "confusion_matrix.png"),
    dpi=300
)

plt.close()

print("\nModel and evaluation results saved successfully.")
print("Model:", MODEL_FILE)
print("Results folder:", RESULTS_DIR)
print("\nOriginal sequence_model.pkl was NOT modified.")
