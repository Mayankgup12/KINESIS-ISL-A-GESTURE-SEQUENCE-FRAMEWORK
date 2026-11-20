from pathlib import Path
import pandas as pd
import re

DATASET_DIR = Path(
    r"E:\Kinesis ISL\external_dataset\ISLVT"
    r"\Indian Sign Language Video and Text dataset for sentences (ISLVT)"
)

EXCEL_FILE = DATASET_DIR / "marathi-English_ISLgloss1.xlsx"
OUTPUT_FILE = DATASET_DIR / "islvt_mapping_report.csv"


def main():
    if not EXCEL_FILE.exists():
        print("ERROR: Excel file not found!")
        print(EXCEL_FILE)
        return

    # Find the actual Excel header row.
    raw_df = pd.read_excel(EXCEL_FILE, header=None)
    header_row = None

    for i, row in raw_df.iterrows():
        values = [
            str(value).strip().lower()
            for value in row
            if pd.notna(value)
        ]

        if (
            any("english sentence" in value for value in values)
            and any("gloss" in value for value in values)
        ):
            header_row = i
            break

    if header_row is None:
        print("ERROR: Could not identify the Excel header.")
        return

    df = pd.read_excel(EXCEL_FILE, header=header_row)
    df.columns = [str(col).strip() for col in df.columns]

    required_columns = [
        "sr.no",
        "English sentence",
        "Indian sign language gloss",
    ]

    for column in required_columns:
        if column not in df.columns:
            print(f"ERROR: Missing Excel column: {column}")
            print("Available columns:", list(df.columns))
            return

    videos = [
        path for path in DATASET_DIR.iterdir()
        if path.is_file() and path.suffix.lower() == ".mov"
    ]

    # Extract the leading sentence ID from each video filename.
    videos_by_id = {}

    for video in videos:
        match = re.match(r"^\s*(\d+)\s+", video.stem)

        if match:
            sentence_id = int(match.group(1))
            videos_by_id.setdefault(sentence_id, []).append(video.name)

    report = []

    for _, row in df.iterrows():
        serial = row["sr.no"]
        english = row["English sentence"]
        gloss = row["Indian sign language gloss"]

        if pd.isna(serial) or pd.isna(english) or pd.isna(gloss):
            continue

        try:
            sentence_id = int(float(serial))
        except (ValueError, TypeError):
            continue

        matched_videos = sorted(videos_by_id.get(sentence_id, []))

        report.append({
            "sentence_id": sentence_id,
            "english_sentence": str(english).strip(),
            "isl_gloss": str(gloss).strip(),
            "video_count": len(matched_videos),
            "video_files": " | ".join(matched_videos),
            "mapping_status": (
                "MATCHED" if matched_videos else "NO VIDEO FOR ID"
            ),
        })

    result = pd.DataFrame(report)
    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("=" * 65)
    print("KINESIS-ISL | ISLVT ID-BASED MAPPING")
    print("=" * 65)
    print("Excel sentence IDs:", len(result))
    print("Video files:", len(videos))
    print("IDs with videos:", (result["video_count"] > 0).sum())
    print("IDs without videos:", (result["video_count"] == 0).sum())
    print("Total videos mapped:", result["video_count"].sum())

    print("\nMAPPING DETAILS")
    print("-" * 65)

    for _, row in result.iterrows():
        print(
            f"\nID {row['sentence_id']} | "
            f"Videos: {row['video_count']}"
        )
        print("English:", row["english_sentence"])
        print("Gloss:", row["isl_gloss"])

        if row["video_count"]:
            for filename in row["video_files"].split(" | "):
                print("  -", filename)
        else:
            print("  - NO VIDEO FOUND")

    print("\nReport saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()

