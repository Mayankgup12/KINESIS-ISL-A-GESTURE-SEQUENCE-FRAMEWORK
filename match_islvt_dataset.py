from pathlib import Path
import pandas as pd
import re
import unicodedata

DATASET_DIR = Path(
    r"E:\Kinesis ISL\external_dataset\ISLVT"
    r"\Indian Sign Language Video and Text dataset for sentences (ISLVT)"
)

EXCEL_FILE = DATASET_DIR / "marathi-English_ISLgloss1.xlsx"


def normalize_text(text):
    """Normalize text for approximate matching."""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"\.mov$", "", text, flags=re.IGNORECASE)

    # Remove the sentence number at the beginning.
    text = re.sub(r"^\s*\d+\s+", "", text)

    # Remove duplicate recording markers.
    text = re.sub(r"\s*\(\d+\)\s*$", "", text)
    text = re.sub(r"\s*-\s*copy\s*$", "", text, flags=re.IGNORECASE)

    # Normalize punctuation and whitespace.
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def main():
    if not EXCEL_FILE.exists():
        print("Excel file not found:", EXCEL_FILE)
        return

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
        print("Could not find Excel header row.")
        return

    df = pd.read_excel(EXCEL_FILE, header=header_row)
    df.columns = [str(c).strip() for c in df.columns]

    english_col = "English sentence"
    gloss_col = "Indian sign language gloss"

    videos = sorted(
        p for p in DATASET_DIR.iterdir()
        if p.is_file() and p.suffix.lower() == ".mov"
    )

    # Map normalized filename text to actual video paths.
    video_lookup = {}
    for video in videos:
        key = normalize_text(video.stem)
        video_lookup.setdefault(key, []).append(video)

    matched_rows = 0
    matched_video_count = 0
    unmatched_rows = []

    print("=" * 70)
    print("KINESIS-ISL | IMPROVED ISLVT MATCHING")
    print("=" * 70)
    print("Excel rows:", len(df))
    print("Video files:", len(videos))

    for _, row in df.iterrows():
        english = row.get(english_col, "")
        gloss = row.get(gloss_col, "")

        if pd.isna(english) or pd.isna(gloss):
            continue

        gloss_key = normalize_text(gloss)
        english_key = normalize_text(english)

        candidates = list(video_lookup.get(gloss_key, []))

        # Try English sentence only if gloss has no exact filename match.
        if not candidates:
            candidates = list(video_lookup.get(english_key, []))

        # Fallback: compare tokens, allowing minor spelling differences.
        if not candidates and gloss_key:
            gloss_tokens = set(gloss_key.split())
            best_score = 0

            for key, paths in video_lookup.items():
                video_tokens = set(key.split())
                if not video_tokens:
                    continue

                score = len(gloss_tokens & video_tokens) / len(gloss_tokens)

                if score > best_score:
                    best_score = score
                    candidates = paths if score >= 0.75 else []

        if candidates:
            matched_rows += 1
            matched_video_count += len(candidates)

            print(f"\n[{row.get('sr.no', '?')}] {english}")
            print("Gloss:", gloss)
            print("Candidate videos:")
            for video in candidates:
                print("  -", video.name)
        else:
            unmatched_rows.append((english, gloss))
            print(f"\nNO MATCH: {english} | Gloss: {gloss}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("Excel data rows:", len(df))
    print("Video files:", len(videos))
    print("Rows with candidate matches:", matched_rows)
    print("Candidate video files across matched rows:", matched_video_count)
    print("Rows without matches:", len(unmatched_rows))
    print("\nNote: Candidate matches are not yet manually verified.")


if __name__ == "__main__":
    main()
