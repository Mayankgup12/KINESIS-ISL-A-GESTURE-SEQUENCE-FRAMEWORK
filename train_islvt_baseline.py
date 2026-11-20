import os
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyClassifier

import joblib


# ==========================================
# KINESIS-ISL: ISLVT BASELINE EXPERIMENT
# Separate from the existing Kinesis-ISL app.
# ==========================================

PROJECT_DIR = r"E:\Kinesis ISL"

DATASET_FILE = os.path.join(
    PROJECT_DIR,
    "external_dataset",
    "islvt_sequences",
    "islvt_sequences.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "external_dataset",
    "islvt_baseline_results"
)

MODEL_FILE = os.path.join(OUTPUT_DIR, "islvt_baseline_model.pkl")
LABELS_FILE = os.path.join(OUTPUT_DIR, "islvt_label_mapping.json")
METRICS_FILE = os.path.join(OUTPUT_DIR, "islvt_metrics.txt")
PREDICTIONS_FILE = os.path.join(OUTPUT_DIR, "islvt_predictions.csv")
CONFUSION_FILE = os.path.join(OUTPUT_DIR, "islvt_confusion_matrix.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.20

SEQUENCE_LENGTH = 30
FEATURE_COUNT = 63

FEATURE_COLUMNS = [
    f"t{t:02d}_f{f:02d}"
    for t in range(1, SEQUENCE_LENGTH + 1)
    for f in range(FEATURE_COUNT)
]


def main():
    print("\n" + "=" * 60)
    print(" KINESIS-ISL: ISLVT BASELINE MODEL")
    print("=" * 60)

    if not os.path.exists(DATASET_FILE):
        print("\nERROR: Dataset file not found:")
        print(DATASET_FILE)
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nLoading prepared sequences...")
    df = pd.read_csv(DATASET_FILE)

    required = [
        "sentence_id",
        "english_sentence",
        "isl_gloss",
        "video_file"
    ] + FEATURE_COLUMNS

    missing = [col for col in required if col not in df.columns]

    if missing:
        print("ERROR: Missing required columns:", missing)
        return

    if df["video_file"].duplicated().any():
        print("ERROR: Duplicate video records detected.")
        return

    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    # Predict the sentence ID, not the sentence text directly.
    y_raw = df["sentence_id"].astype(str).to_numpy()

    if not np.isfinite(X).all():
        print("ERROR: Non-finite feature values found.")
        return

    class_counts = pd.Series(y_raw).value_counts()

    # A stratified split requires at least two recordings per class.
    # Exclude singleton classes from this initial held-out evaluation.
    eligible_mask = np.isin(y_raw, class_counts[class_counts >= 2].index)

    X_eval = X[eligible_mask]
    y_eval_raw = y_raw[eligible_mask]
    eval_df = df.loc[eligible_mask].reset_index(drop=True)

    excluded_ids = sorted(
        class_counts[class_counts < 2].index.tolist(),
        key=lambda value: int(value)
    )

    print("\nDataset overview")
    print("-" * 40)
    print("Total videos:", len(df))
    print("Total sentence IDs:", df["sentence_id"].nunique())
    print("Videos included in evaluation:", len(eval_df))
    print("Sentence classes included:", len(np.unique(y_eval_raw)))
    print("Singleton classes excluded:", excluded_ids)

    # Split each sentence class into training and testing videos.
    # Each eligible class has exactly two recordings.
    # One recording goes to training and the other to testing.

    train_indices = []
    test_indices = []

    rng = np.random.RandomState(RANDOM_STATE)

    for sentence_id in sorted(np.unique(y_eval_raw), key=int):
        class_indices = np.where(y_eval_raw == sentence_id)[0].copy()
        rng.shuffle(class_indices)

        if len(class_indices) < 2:
            print(f"Skipping sentence ID {sentence_id}: fewer than 2 videos.")
            continue

        train_indices.append(class_indices[0])
        test_indices.append(class_indices[1])

    train_idx = np.array(train_indices, dtype=int)
    test_idx = np.array(test_indices, dtype=int)

    X_train = X_eval[train_idx]
    X_test = X_eval[test_idx]

    y_train_raw = y_eval_raw[train_idx]
    y_test_raw = y_eval_raw[test_idx]

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    print("\nSplit details")
    print("-" * 40)
    print("Training videos:", len(X_train))
    print("Testing videos:", len(X_test))
    print("Training classes:", len(np.unique(y_train)))
    print("Testing classes:", len(np.unique(y_test)))


    print("\nTraining baseline Random Forest...")

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    # A majority-class dummy baseline for comparison.
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_predictions = dummy.predict(X_test)
    dummy_accuracy = accuracy_score(y_test, dummy_predictions)

    class_names = label_encoder.classes_
    report = classification_report(
        y_test,
        predictions,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=np.arange(len(class_names))
    )

    # Map sentence IDs to English text and gloss.
    metadata = (
        df.drop_duplicates("sentence_id")
        .set_index(df.drop_duplicates("sentence_id")["sentence_id"].astype(str))
    )

    predicted_ids = label_encoder.inverse_transform(predictions)

    results = eval_df.iloc[test_idx][
        ["sentence_id", "english_sentence", "isl_gloss", "video_file"]
    ].copy()

    results["actual_sentence_id"] = y_test_raw
    results["predicted_sentence_id"] = predicted_ids
    results["predicted_english_sentence"] = [
        metadata.loc[sid, "english_sentence"]
        for sid in predicted_ids
    ]
    results["correct"] = (
        results["actual_sentence_id"].astype(str)
        == results["predicted_sentence_id"].astype(str)
    )

    # Save model and label mapping separately from the existing model.
    joblib.dump(
        {
            "model": model,
            "label_encoder": label_encoder,
            "feature_columns": FEATURE_COLUMNS,
            "sequence_length": SEQUENCE_LENGTH,
            "feature_count": FEATURE_COUNT
        },
        MODEL_FILE
    )

    label_mapping = {
        str(row["sentence_id"]): {
            "english_sentence": row["english_sentence"],
            "isl_gloss": row["isl_gloss"]
        }
        for _, row in df.drop_duplicates("sentence_id").iterrows()
    }

    with open(LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=2)

    results.to_csv(PREDICTIONS_FILE, index=False)

    cm_df = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names
    )
    cm_df.to_csv(CONFUSION_FILE)

    metrics_text = (
        "KINESIS-ISL ISLVT BASELINE RESULTS\n"
        "=================================\n"
        f"Total videos: {len(df)}\n"
        f"Videos included in evaluation: {len(eval_df)}\n"
        f"Excluded singleton sentence IDs: {excluded_ids}\n"
        f"Number of evaluated sentence classes: {len(class_names)}\n"
        f"Training videos: {len(X_train)}\n"
        f"Testing videos: {len(X_test)}\n"
        f"Random Forest accuracy: {accuracy:.4f}\n"
        f"Majority-class baseline accuracy: {dummy_accuracy:.4f}\n\n"
        "Classification report:\n"
        f"{report}\n\n"
        "Evaluation caveat:\n"
        "This is a small exploratory dataset. The split holds out videos\n"
        "from sentence classes represented in training. It does not test\n"
        "generalization to entirely unseen sentences or signers.\n"
        "If recordings are related takes, results may be optimistic.\n"
    )

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        f.write(metrics_text)

    print("\n" + "=" * 60)
    print(" BASELINE RESULTS")
    print("=" * 60)
    print(f"Random Forest accuracy: {accuracy:.2%}")
    print(f"Majority-class baseline accuracy: {dummy_accuracy:.2%}")

    print("\nClassification report:")
    print(report)

    print("\nSaved outputs:")
    print(MODEL_FILE)
    print(LABELS_FILE)
    print(METRICS_FILE)
    print(PREDICTIONS_FILE)
    print(CONFUSION_FILE)

    print("\nExisting Kinesis-ISL model was not modified.")
    print("Baseline experiment completed.")


if __name__ == "__main__":
    main()

