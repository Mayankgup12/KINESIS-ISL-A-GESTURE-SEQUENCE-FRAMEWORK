import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline


# =====================================================
# KINESIS-ISL: IMPROVED ISLVT EXPERIMENT
# Separate experiment. Does not modify the main app.
# =====================================================

PROJECT_DIR = r"E:\Kinesis ISL"

FRAME_FILE = os.path.join(
    PROJECT_DIR,
    "external_dataset",
    "islvt_landmarks",
    "islvt_frame_landmarks.csv"
)

SEQUENCE_FILE = os.path.join(
    PROJECT_DIR,
    "external_dataset",
    "islvt_sequences",
    "islvt_sequences.csv"
)

BASELINE_PREDICTIONS = os.path.join(
    PROJECT_DIR,
    "external_dataset",
    "islvt_baseline_results",
    "islvt_predictions.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "external_dataset",
    "islvt_improved_results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
EXPECTED_FRAMES = 30
LANDMARKS = 21
COORDS = 3


def get_frame_features(frame):
    """
    Convert 63 hand-landmark values into:
    1. Wrist-relative coordinates.
    2. Scale-normalized coordinates.

    Input: 63 values = 21 landmarks x 3 coordinates.
    Output: 63 normalized values.
    """

    points = np.asarray(frame, dtype=np.float32).reshape(
        LANDMARKS, COORDS
    )

    # Translate all landmarks relative to wrist (landmark 0).
    relative = points - points[0]

    # Normalize by maximum XY distance from wrist.
    xy_distances = np.linalg.norm(relative[:, :2], axis=1)
    scale = float(np.max(xy_distances))

    if scale > 1e-6:
        relative = relative / scale

    return relative.flatten()


def create_video_features(video_rows):
    """
    Create one feature vector per video.
    Each video is represented by 30 frames x 63 features.

    Input rows are sorted by frame_index.
    """

    raw = video_rows[
        [f"f{i:02d}" for i in range(63)]
    ].to_numpy(dtype=np.float32)

    # Sort frame observations chronologically.
    order = np.argsort(
        video_rows["frame_index"].to_numpy()
    )
    raw = raw[order]

    # Normalize each observed frame independently.
    normalized = np.stack([
        get_frame_features(frame) for frame in raw
    ])

    # Resample to 30 observations using interpolation.
    old_t = np.linspace(0.0, 1.0, len(normalized))
    new_t = np.linspace(0.0, 1.0, EXPECTED_FRAMES)

    resampled = np.empty(
        (EXPECTED_FRAMES, 63), dtype=np.float32
    )

    for feature_index in range(63):
        resampled[:, feature_index] = np.interp(
            new_t,
            old_t,
            normalized[:, feature_index]
        )

    # Frame-to-frame displacement.
    movement = np.diff(
        resampled, axis=0, prepend=resampled[[0]]
    )

    # Combine static hand shape and movement.
    combined = np.concatenate(
        [resampled, movement], axis=1
    )

    return combined.flatten()


def main():
    print("\n" + "=" * 60)
    print(" KINESIS-ISL: IMPROVED ISLVT EXPERIMENT")
    print("=" * 60)

    if not os.path.exists(FRAME_FILE):
        raise FileNotFoundError(FRAME_FILE)

    if not os.path.exists(SEQUENCE_FILE):
        raise FileNotFoundError(SEQUENCE_FILE)

    if not os.path.exists(BASELINE_PREDICTIONS):
        raise FileNotFoundError(BASELINE_PREDICTIONS)

    frames = pd.read_csv(FRAME_FILE)
    sequences = pd.read_csv(SEQUENCE_FILE)
    baseline = pd.read_csv(BASELINE_PREDICTIONS)

    required_frame_cols = [
        "sentence_id",
        "english_sentence",
        "isl_gloss",
        "video_file",
        "frame_index"
    ] + [f"f{i:02d}" for i in range(63)]

    missing = [
        col for col in required_frame_cols
        if col not in frames.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in frame CSV: {missing}"
        )

    # Reconstruct video-level features.
    feature_rows = []
    metadata_rows = []

    print("\nBuilding normalized video features...")

    for video_file, group in frames.groupby(
        "video_file", sort=False
    ):
        group = group.sort_values("frame_index")

        features = create_video_features(group)

        if not np.isfinite(features).all():
            print("Skipping non-finite video:", video_file)
            continue

        first = group.iloc[0]

        feature_rows.append(features)

        metadata_rows.append({
            "sentence_id": str(first["sentence_id"]),
            "english_sentence": first["english_sentence"],
            "isl_gloss": first["isl_gloss"],
            "video_file": video_file,
            "observed_frames": len(group)
        })

    feature_matrix = np.asarray(
        feature_rows, dtype=np.float32
    )

    meta = pd.DataFrame(metadata_rows)

    if meta["video_file"].duplicated().any():
        raise ValueError("Duplicate video records found.")

    print("Videos processed:", len(meta))
    print("Features per video:", feature_matrix.shape[1])

    # Match the exact train/test videos used in baseline.
    baseline_test_files = set(baseline["video_file"].astype(str))

    all_video_files = meta["video_file"].astype(str)

    # Baseline output includes only test videos, so reconstruct
    # the same deterministic per-class split used by the baseline.
    class_counts = meta["sentence_id"].value_counts()
    eligible_ids = set(
        class_counts[class_counts >= 2].index.astype(str)
    )

    eligible_mask = meta["sentence_id"].isin(eligible_ids)

    eligible_meta = meta.loc[eligible_mask].reset_index(drop=True)
    eligible_X = feature_matrix[eligible_mask.to_numpy()]

    rng = np.random.RandomState(RANDOM_STATE)

    train_indices = []
    test_indices = []

    for sentence_id in sorted(
        eligible_meta["sentence_id"].unique(), key=int
    ):
        class_indices = np.where(
            eligible_meta["sentence_id"].to_numpy()
            == sentence_id
        )[0].copy()

        rng.shuffle(class_indices)

        train_indices.append(class_indices[0])
        test_indices.append(class_indices[1])

    train_idx = np.asarray(train_indices, dtype=int)
    test_idx = np.asarray(test_indices, dtype=int)

    # Ensure the held-out videos match the baseline.
    actual_test_files = set(
        eligible_meta.iloc[test_idx]["video_file"].astype(str)
    )

    if actual_test_files != baseline_test_files:
        raise RuntimeError(
            "Train/test split mismatch with baseline. "
            "Stop: do not compare these results directly."
        )

    X_train = eligible_X[train_idx]
    X_test = eligible_X[test_idx]

    y_train_raw = eligible_meta.iloc[train_idx][
        "sentence_id"
    ].astype(str).to_numpy()

    y_test_raw = eligible_meta.iloc[test_idx][
        "sentence_id"
    ].astype(str).to_numpy()

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_raw)
    y_test = encoder.transform(y_test_raw)

    print("\nEvaluation setup")
    print("-" * 40)
    print("Training videos:", len(X_train))
    print("Testing videos:", len(X_test))
    print("Sentence classes:", len(encoder.classes_))
    print("Baseline test split matched: YES")

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf",
                class_weight="balanced"
            ))
        ])
    }

    comparison = []
    prediction_outputs = []

    for name, model in models.items():
        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        balanced = balanced_accuracy_score(
            y_test, predictions
        )

        report = classification_report(
            y_test,
            predictions,
            labels=np.arange(len(encoder.classes_)),
            target_names=encoder.classes_,
            zero_division=0
        )

        print(f"{name} accuracy: {accuracy:.2%}")
        print(f"{name} balanced accuracy: {balanced:.2%}")

        comparison.append({
            "model": name,
            "accuracy": accuracy,
            "balanced_accuracy": balanced
        })

        test_meta = eligible_meta.iloc[test_idx].reset_index(
            drop=True
        )

        predicted_ids = encoder.inverse_transform(predictions)

        output = test_meta[
            [
                "sentence_id",
                "english_sentence",
                "isl_gloss",
                "video_file"
            ]
        ].copy()

        output["model"] = name
        output["actual_sentence_id"] = y_test_raw
        output["predicted_sentence_id"] = predicted_ids

        lookup = (
            meta.drop_duplicates("sentence_id")
            .set_index("sentence_id")["english_sentence"]
            .to_dict()
        )

        output["predicted_english_sentence"] = [
            lookup[sid] for sid in predicted_ids
        ]

        output["correct"] = (
            output["actual_sentence_id"]
            == output["predicted_sentence_id"]
        )

        prediction_outputs.append(output)

        with open(
            os.path.join(OUTPUT_DIR, f"{name}_report.txt"),
            "w",
            encoding="utf-8"
        ) as f:
            f.write(report)

        joblib.dump(
            {
                "model": model,
                "label_encoder": encoder,
                "feature_type": (
                    "wrist_relative_scale_normalized_"
                    "landmarks_plus_frame_displacement"
                ),
                "feature_count": feature_matrix.shape[1]
            },
            os.path.join(OUTPUT_DIR, f"{name}_model.pkl")
        )

    # Dummy baseline for reference.
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)

    dummy_accuracy = accuracy_score(y_test, dummy_pred)

    comparison.append({
        "model": "Dummy_MostFrequent",
        "accuracy": dummy_accuracy,
        "balanced_accuracy": balanced_accuracy_score(
            y_test, dummy_pred
        )
    })

    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(
        os.path.join(OUTPUT_DIR, "model_comparison.csv"),
        index=False
    )

    pd.concat(
        prediction_outputs, ignore_index=True
    ).to_csv(
        os.path.join(OUTPUT_DIR, "improved_predictions.csv"),
        index=False
    )

    with open(
        os.path.join(OUTPUT_DIR, "experiment_notes.txt"),
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "ISLVT IMPROVED FEATURE EXPERIMENT\n"
            "=================================\n"
            f"Videos processed: {len(meta)}\n"
            f"Training videos: {len(X_train)}\n"
            f"Testing videos: {len(X_test)}\n"
            f"Classes evaluated: {len(encoder.classes_)}\n"
            f"Dummy accuracy: {dummy_accuracy:.4f}\n\n"
            "Features: wrist-relative coordinates, per-frame "
            "scale normalization, and frame displacement.\n\n"
            "Limitations:\n"
            "- Very few recordings per sentence class.\n"
            "- Only hand landmarks are used.\n"
            "- Missing-hand frames were not recovered.\n"
            "- This experiment predicts known sentence IDs; it "
            "does not generate arbitrary sentences.\n"
            "- Results are exploratory, not conclusive.\n"
        )

    print("\n" + "=" * 60)
    print(" MODEL COMPARISON")
    print("=" * 60)
    print(comparison_df.to_string(index=False))

    print("\nOutput folder:")
    print(OUTPUT_DIR)
    print("\nMain Kinesis-ISL files remain unchanged.")


if __name__ == "__main__":
    main()

