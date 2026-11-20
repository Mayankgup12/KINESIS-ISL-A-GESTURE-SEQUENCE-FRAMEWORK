import os
import numpy as np
import pandas as pd

PROJECT_DIR = r"E:\Kinesis ISL"

INPUT_FILE = os.path.join(
    PROJECT_DIR,
    "external_dataset",
    "islvt_sequences",
    "islvt_sequences.csv"
)

FEATURE_COLUMNS = [
    f"t{t:02d}_f{f:02d}"
    for t in range(1, 31)
    for f in range(63)
]


def main():
    print("\n" + "=" * 55)
    print(" KINESIS-ISL: DATASET QUALITY CHECK")
    print("=" * 55)

    if not os.path.exists(INPUT_FILE):
        print("ERROR: Sequence dataset not found:")
        print(INPUT_FILE)
        return

    df = pd.read_csv(INPUT_FILE)

    print("\n1. BASIC INFORMATION")
    print("-" * 40)
    print("Dataset shape:", df.shape)
    print("Total videos:", df["video_file"].nunique())
    print("Unique sentence IDs:", df["sentence_id"].nunique())
    print("Unique English sentences:", df["english_sentence"].nunique())
    print("Unique ISL glosses:", df["isl_gloss"].nunique())

    print("\n2. FEATURE VALIDATION")
    print("-" * 40)

    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)

    print("Expected features per video:", 1890)
    print("Actual features per video:", X.shape[1])
    print("NaN values:", int(np.isnan(X).sum()))
    print("Infinite values:", int(np.isinf(X).sum()))
    print("All feature values finite:", bool(np.isfinite(X).all()))

    print("\n3. LABEL VALIDATION")
    print("-" * 40)

    print("Missing sentence IDs:", int(df["sentence_id"].isna().sum()))
    print("Missing English sentences:", int(df["english_sentence"].isna().sum()))
    print("Missing ISL glosses:", int(df["isl_gloss"].isna().sum()))
    print("Missing video filenames:", int(df["video_file"].isna().sum()))

    print("\n4. VIDEOS PER SENTENCE")
    print("-" * 40)

    counts = df.groupby("sentence_id")["video_file"].nunique()

    print("Sentence IDs with 1 video:", int((counts == 1).sum()))
    print("Sentence IDs with 2 videos:", int((counts == 2).sum()))
    print("Sentence IDs with other video counts:", int((counts > 2).sum()))

    print("\n5. DUPLICATE CHECK")
    print("-" * 40)

    print(
        "Duplicate video filenames:",
        int(df["video_file"].duplicated().sum())
    )

    print(
        "Duplicate sentence ID + video combinations:",
        int(df.duplicated(["sentence_id", "video_file"]).sum())
    )

    print("\n6. SAMPLE LABELS")
    print("-" * 40)

    print(
        df[
            ["sentence_id", "english_sentence", "isl_gloss", "video_file"]
        ].head(10).to_string(index=False)
    )

    print("\n7. QUALITY CHECK SUMMARY")
    print("-" * 40)

    checks = {
        "Expected feature count": X.shape[1] == 1890,
        "All features finite": bool(np.isfinite(X).all()),
        "No missing sentence IDs": df["sentence_id"].notna().all(),
        "No missing English sentences": df["english_sentence"].notna().all(),
        "No missing ISL glosses": df["isl_gloss"].notna().all(),
        "No missing video filenames": df["video_file"].notna().all(),
        "One row per video": not df["video_file"].duplicated().any(),
    }

    for name, passed in checks.items():
        print(f"{'[PASS]' if passed else '[CHECK]'} {name}")

    if all(checks.values()):
        print("\nBasic quality checks passed.")
    else:
        print("\nSome checks need investigation before training.")

    print("\nNote: These checks do not prove that the labels")
    print("are semantically correct or that recordings are independent.")


if __name__ == "__main__":
    main()

