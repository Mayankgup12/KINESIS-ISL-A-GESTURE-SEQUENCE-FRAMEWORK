from pathlib import Path

DATASET_DIR = Path(
    r"E:\Kinesis ISL\external_dataset\ISLVT"
    r"\Indian Sign Language Video and Text dataset for sentences (ISLVT)"
)

print("=" * 60)
print("ISLVT VIDEO FILENAME INSPECTION")
print("=" * 60)

videos = sorted(
    p for p in DATASET_DIR.iterdir()
    if p.is_file() and p.suffix.lower() == ".mov"
)

print(f"\nTotal MOV videos: {len(videos)}")
print("\nFirst 30 filenames:\n")

for i, video in enumerate(videos[:30], start=1):
    print(f"{i:02}. {video.name}")

print("\n" + "=" * 60)
print("EXCEL FILES IN DATASET FOLDER")
print("=" * 60)

for file in DATASET_DIR.iterdir():
    if file.is_file() and file.suffix.lower() in {".xlsx", ".xls", ".csv"}:
        print(file.name)

