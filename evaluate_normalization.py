import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Dataset paths
TRAIN_FILE = "dataset/gestures.csv"
TEST_FILE = "dataset/gestures_session2.csv"

# Load datasets
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

# Separate features and labels
X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]

X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]

# Ensure matching feature columns
X_test = X_test[X_train.columns]

# Split original dataset
X_train_part, X_val, y_train_part, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)

# Train raw-coordinate model
raw_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

raw_model.fit(X_train_part, y_train_part)

raw_val_accuracy = accuracy_score(
    y_val, raw_model.predict(X_val)
)

raw_test_accuracy = accuracy_score(
    y_test, raw_model.predict(X_test)
)

# Normalize coordinates relative to wrist landmark
def normalize_landmarks(data):
    normalized = data.to_numpy(dtype=float).copy()

    # Landmark 0 is the wrist; each landmark has x, y, z.
    wrist = normalized[:, 0:3].copy()

    # Subtract wrist coordinates from all 21 landmarks.
    normalized = normalized - np.tile(wrist, (1, 21))

    return pd.DataFrame(normalized, columns=data.columns, index=data.index)

X_train_norm = normalize_landmarks(X_train)
X_test_norm = normalize_landmarks(X_test)

X_train_norm_part, X_val_norm, y_train_norm_part, y_val_norm = (
    train_test_split(
        X_train_norm,
        y_train,
        test_size=0.20,
        random_state=42,
        stratify=y_train
    )
)

# Train normalized-coordinate model
norm_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

norm_model.fit(X_train_norm_part, y_train_norm_part)

norm_val_accuracy = accuracy_score(
    y_val_norm, norm_model.predict(X_val_norm)
)

norm_test_accuracy = accuracy_score(
    y_test, norm_model.predict(X_test_norm)
)

# Display results
print("\n===== KINESIS-ISL NORMALIZATION EXPERIMENT =====")
print(f"Raw coordinates - validation accuracy: {raw_val_accuracy * 100:.2f}%")
print(f"Normalized - validation accuracy:      {norm_val_accuracy * 100:.2f}%")
print()
print(f"Raw coordinates - Session 2 accuracy: {raw_test_accuracy * 100:.2f}%")
print(f"Normalized - Session 2 accuracy:      {norm_test_accuracy * 100:.2f}%")
from sklearn.metrics import classification_report, confusion_matrix

print("\n===== RAW MODEL: SESSION 2 =====")
raw_predictions = raw_model.predict(X_test)

print(classification_report(
    y_test,
    raw_predictions,
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(
    y_test,
    raw_predictions,
    labels=raw_model.classes_
))

print("\n===== NORMALIZED MODEL: SESSION 2 =====")
normalized_predictions = norm_model.predict(X_test_norm)

print(classification_report(
    y_test,
    normalized_predictions,
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(
    y_test,
    normalized_predictions,
    labels=norm_model.classes_
))

print("\nClass order:")
print(list(norm_model.classes_))