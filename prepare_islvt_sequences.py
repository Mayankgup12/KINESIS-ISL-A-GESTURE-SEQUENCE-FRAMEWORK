import os
import numpy as np
import pandas as pd

# ==========================================
# KINESIS-ISL: ISLVT SEQUENCE PREPARATION
# This script does NOT modify the existing model.
# ==========================================

PROJECT_DIR = r"E:\Kinesis ISL"

DATASET_DIR = os.path.join(
    PROJECT_DIR,
    "external_dataset",
    "islvt_landmarks"
)

INPUT_FILE = os.path.join(
    DATASET_DIR,
    "islvt_frame_landmarks.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "external_dataset",
    "islvt_sequences"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "islvt_sequences.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "islvt_sequence_summary.csv"
)

SEQUENCE_LENGTH = 30
FEATURE_COUNT = 63

FEATURE_COLUMNS = [
    f"f{i:02d}" for i in range(FEATURE_COUNT)
]


def resample_sequence(features, target_length=30):
    """
    Normalize one video's detected hand landmarks
    to a fixed number of time steps.

    Input shape:  (n_frames, 63)
    Output shape: (target_length, 63)
    """

    features = np.asarray(features, dtype=np.float32)

    if len(features) == 0:
        raise ValueError("Cannot resample an empty sequence.")

    if len(features) == 1:
        return np.repeat(features, target_length, axis=0)

    old_positions = np.linspace(0, 1, num=len(features))
    new_positions = np.linspace(0, 1, num=target_length)

    resampled = np.zeros(
        (target_length, features.shape[1]),
        dtype=np.float32
    )

    for feature_index in range(features.shape[1]):
        resampled[:, feature_index] = np.interp(
            new_positions,
            old_positions,
            features[:, feature_index]
        )

    return resampled


def main():
    print("\n" + "=" * 55)
    print(" KINESIS-ISL: ISLVT SEQUENCE PREPARATION")
    print("=" * 55)

    if not os.path.exists(INPUT_FILE):
        print("\nERROR: Landmark CSV not found!")
        print("Expected file:", INPUT_FILE)
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nLoading landmark data...")

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "sentence_id",
        "english_sentence",
        "isl_gloss",
        "video_file",
        "frame_index"
    ] + FEATURE_COLUMNS

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        print("\nERROR: Required columns are missing:")
        print(missing_columns)
        return

    print("Landmark rows:", len(df))
    print("Unique videos:", df["video_file"].nunique())
    print("Unique sentence IDs:", df["sentence_id"].nunique())

    print("\nPreparing individual video sequences...")

    sequences = []
    summaries = []

    grouped = df.groupby("video_file", sort=True)

    for video_file, group in grouped:
        group = group.sort_values("frame_index").reset_index(drop=True)

        sentence_id = group["sentence_id"].iloc[0]
        english_sentence = group["english_sentence"].iloc[0]
        isl_gloss = group["isl_gloss"].iloc[0]

        # Verify that one video has only one sentence label.
        if group["sentence_id"].nunique() != 1:
            print("WARNING: Inconsistent sentence IDs:", video_file)
            continue

        features = group[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

        # Skip sequences with invalid feature values.
        if not np.isfinite(features).all():
            print("WARNING: Invalid feature values:", video_file)
            continue

        if len(features) == 0:
            continue

        # Convert each video to 30 time steps.
        normalized = resample_sequence(
            features,
            target_length=SEQUENCE_LENGTH
        )

        record = {
            "sentence_id": sentence_id,
            "english_sentence": english_sentence,
            "isl_gloss": isl_gloss,
            "video_file": video_file,
            "original_frames": len(features)
        }

        # Save all 30 time steps and 63 features.
        for time_step in range(SEQUENCE_LENGTH):
            for feature_index in range(FEATURE_COUNT):
                column = f"t{time_step + 1:02d}_f{feature_index:02d}"
                record[column] = normalized[
                    time_step, feature_index
                ]

        sequences.append(record)

        summaries.append({
            "sentence_id": sentence_id,
            "english_sentence": english_sentence,
            "isl_gloss": isl_gloss,
            "video_file": video_file,
            "detected_frames": len(features),
            "sequence_length": SEQUENCE_LENGTH,
            "feature_count_per_frame": FEATURE_COUNT,
            "total_sequence_features": (
                SEQUENCE_LENGTH * FEATURE_COUNT
            )
        })

    if not sequences:
        print("\nERROR: No valid sequences were generated.")
        return

    sequence_df = pd.DataFrame(sequences)
    summary_df = pd.DataFrame(summaries)

    print("\nSaving prepared sequences...")

    sequence_df.to_csv(OUTPUT_FILE, index=False)
    summary_df.to_csv(SUMMARY_FILE, index=False)

    print("\n" + "=" * 55)
    print(" SEQUENCE PREPARATION RESULTS")
    print("=" * 55)

    print("Input landmark rows:", len(df))
    print("Videos processed:", len(sequence_df))
    print("Unique sentence IDs:", sequence_df["sentence_id"].nunique())
    print("Sequence length:", SEQUENCE_LENGTH)
    print("Features per frame:", FEATURE_COUNT)
    print(
        "Features per video:",
        SEQUENCE_LENGTH * FEATURE_COUNT
    )
    print("Output CSV shape:", sequence_df.shape)

    print("\nSequences per sentence ID:")
    print(
        sequence_df.groupby("sentence_id")["video_file"]
        .count()
        .to_string()
    )

    print("\nOutput files:")
    print(OUTPUT_FILE)
    print(SUMMARY_FILE)

    print("\nSequence preparation completed successfully!")
    print("Existing Kinesis-ISL model was not modified.")


if __name__ == "__main__":
    main()

