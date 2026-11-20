import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================
# CONFIGURATION
# ==========================================

DATASET_FILE = "dataset/gesture_sequences_fresh.csv"
MODEL_FILE = "models/sequence_model.pkl"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

print("\n======================================")
print(" KINESIS-ISL MODEL EVALUATION")
print("======================================")

# ==========================================
# LOAD DATASET
# ==========================================

print("\nLoading dataset...")

df = pd.read_csv(DATASET_FILE)

print("Dataset shape:", df.shape)

X = df.drop(columns=["label"])
y = df["label"]

print("\nGesture distribution:")
print(y.value_counts().sort_index())

# ==========================================
# CHECK DATA
# ==========================================

if df.isnull().sum().sum() > 0:
    raise ValueError(
        "Dataset contains missing values. "
        "Please check the dataset."
    )

if X.shape[1] != 1890:
    raise ValueError(
        f"Expected 1890 features per sequence, "
        f"but found {X.shape[1]}."
    )

# ==========================================
# RECREATE THE ORIGINAL TEST SPLIT
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
# LOAD TRAINED MODEL
# ==========================================

print("\nLoading trained sequence model...")

model = joblib.load(MODEL_FILE)

print("Model loaded successfully!")
print("Model classes:", list(model.classes_))

# ==========================================
# MAKE PREDICTIONS
# ==========================================

print("\nEvaluating test samples...")

predictions = model.predict(X_test)

# ==========================================
# CALCULATE METRICS
# ==========================================

accuracy = accuracy_score(y_test, predictions)

print("\n======================================")
print(" EVALUATION RESULTS")
print("======================================")

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")

report = classification_report(
    y_test,
    predictions,
    labels=model.classes_,
    zero_division=0
)

print(report)

# ==========================================
# SAVE CLASSIFICATION REPORT
# ==========================================

report_file = os.path.join(
    RESULTS_DIR,
    "classification_report.txt"
)

with open(report_file, "w", encoding="utf-8") as file:
    file.write("Kinesis-ISL Model Evaluation\n\n")
    file.write(f"Test Accuracy: {accuracy * 100:.2f}%\n\n")
    file.write(report)

print("Classification report saved:", report_file)

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    predictions,
    labels=model.classes_
)

print("\nConfusion Matrix:")
print(cm)

# ==========================================
# SAVE CONFUSION MATRIX IMAGE
# ==========================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

fig, ax = plt.subplots(figsize=(9, 7))

display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False
)

ax.set_title("Kinesis-ISL Gesture Classification")
ax.set_xlabel("Predicted Gesture")
ax.set_ylabel("Actual Gesture")

plt.xticks(rotation=30)
plt.tight_layout()

matrix_file = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.png"
)

plt.savefig(matrix_file, dpi=300, bbox_inches="tight")
plt.close(fig)

print("\nConfusion matrix image saved:", matrix_file)

# ==========================================
# SAVE ACCURACY
# ==========================================

accuracy_file = os.path.join(
    RESULTS_DIR,
    "accuracy.txt"
)

with open(accuracy_file, "w", encoding="utf-8") as file:
    file.write(f"Test Accuracy: {accuracy * 100:.2f}%\n")
    file.write(f"Training Samples: {len(X_train)}\n")
    file.write(f"Testing Samples: {len(X_test)}\n")

print("Accuracy saved:", accuracy_file)

print("\n======================================")
print(" EVALUATION COMPLETED SUCCESSFULLY")
print("======================================")