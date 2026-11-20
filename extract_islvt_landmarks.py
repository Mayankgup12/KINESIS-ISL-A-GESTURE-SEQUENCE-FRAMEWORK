from pathlib import Path
import re
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_DIR = Path(r"E:\Kinesis ISL")

DATASET_DIR = (
    PROJECT_DIR
    / "external_dataset"
    / "ISLVT"
    / "Indian Sign Language Video and Text dataset for sentences (ISLVT)"
)

EXCEL_FILE = DATASET_DIR / "marathi-English_ISLgloss1.xlsx"

OUTPUT_DIR = PROJECT_DIR / "external_dataset" / "islvt_landmarks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "islvt_frame_landmarks.csv"
SUMMARY_CSV = OUTPUT_DIR / "islvt_extraction_summary.csv"

# Process approximately one frame every 5 frames.
FRAME_INTERVAL = 5

# =========================================================
# MEDIAPIPE
# =========================================================

mp_hands = mp.solutions.hands


def load_sentence_metadata():
    """Load sentence IDs, English sentences, and ISL glosses."""

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
        raise ValueError("Could not find the Excel header row.")

    df = pd.read_excel(EXCEL_FILE, header=header_row)
    df.columns = [str(col).strip() for col in df.columns]

    metadata = {}

    for _, row in df.iterrows():
        if (
            pd.isna(row.get("sr.no"))
            or pd.isna(row.get("English sentence"))
            or pd.isna(row.get("Indian sign language gloss"))
        ):
            continue

        sentence_id = int(float(row["sr.no"]))

        metadata[sentence_id] = {
            "english_sentence": str(row["English sentence"]).strip(),
            "isl_gloss": str(
                row["Indian sign language gloss"]
            ).strip(),
        }

    return metadata


def extract_frame_features(hand_landmarks):
    """Return 63 values: x, y, z for each of 21 hand landmarks."""

    features = []

    for landmark in hand_landmarks.landmark:
        features.extend([
            landmark.x,
            landmark.y,
            landmark.z,
        ])

    return features


def process_video(video_path, sentence_metadata, hands):
    """Extract sampled frame landmarks from one video."""

    match = re.match(r"^\s*(\d+)\s+", video_path.stem)

    if not match:
        return [], {
            "video": video_path.name,
            "sentence_id": None,
            "sampled_frames": 0,
            "detected_frames": 0,
            "detection_rate_percent": 0.0,
            "status": "SKIPPED: no sentence ID in filename",
        }

    sentence_id = int(match.group(1))
    metadata = sentence_metadata.get(sentence_id)

    if metadata is None:
        return [], {
            "video": video_path.name,
            "sentence_id": sentence_id,
            "sampled_frames": 0,
            "detected_frames": 0,
            "detection_rate_percent": 0.0,
            "status": "SKIPPED: sentence ID missing from Excel",
        }

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return [], {
            "video": video_path.name,
            "sentence_id": sentence_id,
            "sampled_frames": 0,
            "detected_frames": 0,
            "detection_rate_percent": 0.0,
            "status": "ERROR: could not open video",
        }

    frame_rows = []
    sampled_frames = 0
    detected_frames = 0
    frame_index = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_index % FRAME_INTERVAL == 0:
            sampled_frames += 1

            # Mirror frame for consistency with webcam interaction.
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb)

            # Store only frames where at least one hand is detected.
            if results.multi_hand_landmarks:
                detected_frames += 1

                # Current pipeline supports one hand.
                hand = results.multi_hand_landmarks[0]
                features = extract_frame_features(hand)

                record = {
                    "sentence_id": sentence_id,
                    "english_sentence": metadata["english_sentence"],
                    "isl_gloss": metadata["isl_gloss"],
                    "video_file": video_path.name,
                    "frame_index": frame_index,
                }

                for i, value in enumerate(features):
                    record[f"f{i:02d}"] = value

                frame_rows.append(record)

        frame_index += 1

    cap.release()

    rate = (
        100.0 * detected_frames / sampled_frames
        if sampled_frames else 0.0
    )

    summary = {
        "video": video_path.name,
        "sentence_id": sentence_id,
        "sampled_frames": sampled_frames,
        "detected_frames": detected_frames,
        "detection_rate_percent": round(rate, 2),
        "status": "OK" if detected_frames else "NO HAND DETECTED",
    }

    return frame_rows, summary


def main():
    if not EXCEL_FILE.exists():
        raise FileNotFoundError(f"Excel file not found: {EXCEL_FILE}")

    videos = sorted(
        path for path in DATASET_DIR.iterdir()
        if path.is_file() and path.suffix.lower() == ".mov"
    )

    if not videos:
        raise FileNotFoundError(
            f"No MOV videos found in: {DATASET_DIR}"
        )

    print("=" * 65)
    print("KINESIS-ISL | ISLVT LANDMARK EXTRACTION")
    print("=" * 65)

    print("Videos found:", len(videos))
    print("Frame sampling interval:", FRAME_INTERVAL)
    print("Output directory:", OUTPUT_DIR)

    sentence_metadata = load_sentence_metadata()

    all_rows = []
    summaries = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:

        for number, video in enumerate(videos, start=1):
            print(f"\n[{number}/{len(videos)}] {video.name}")

            try:
                rows, summary = process_video(
                    video,
                    sentence_metadata,
                    hands,
                )

                all_rows.extend(rows)
                summaries.append(summary)

                print(
                    f"Sampled: {summary['sampled_frames']} | "
                    f"Detected: {summary['detected_frames']} | "
                    f"Rate: {summary['detection_rate_percent']}% | "
                    f"{summary['status']}"
                )

            except Exception as error:
                print("ERROR:", error)

                summaries.append({
                    "video": video.name,
                    "sentence_id": None,
                    "sampled_frames": 0,
                    "detected_frames": 0,
                    "detection_rate_percent": 0.0,
                    "status": f"ERROR: {error}",
                })

    if all_rows:
        pd.DataFrame(all_rows).to_csv(
            OUTPUT_CSV,
            index=False,
            encoding="utf-8-sig",
        )
    else:
        print("\nWARNING: No hand landmarks were extracted.")

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(
        SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 65)
    print("EXTRACTION COMPLETE")
    print("=" * 65)
    print("Videos processed:", len(summaries))
    print("Videos with detected hands:", sum(
        item["detected_frames"] > 0 for item in summaries
    ))
    print("Total sampled frames:", sum(
        item["sampled_frames"] for item in summaries
    ))
    print("Total frames with hands:", sum(
        item["detected_frames"] for item in summaries
    ))
    print("Landmark rows:", len(all_rows))
    print("Frame landmark CSV:", OUTPUT_CSV)
    print("Extraction summary CSV:", SUMMARY_CSV)


if __name__ == "__main__":
    main()

