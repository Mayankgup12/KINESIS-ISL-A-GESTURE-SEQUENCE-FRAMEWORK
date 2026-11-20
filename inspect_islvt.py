import cv2
from pathlib import Path

# ISLVT dataset folder
DATASET_PATH = Path(
    "external_dataset/ISLVT/Indian Sign Language Video and Text dataset for sentences (ISLVT)"
)

def inspect_videos():
    if not DATASET_PATH.exists():
        print("ERROR: Dataset folder not found!")
        print("Actual dataset path check karo:", DATASET_PATH.resolve())
        return

    videos = sorted(DATASET_PATH.glob("*.MOV"))

    if not videos:
        videos = sorted(DATASET_PATH.glob("*.mov"))

    if not videos:
        print("No MOV videos found.")
        return

    print("\n========== ISLVT VIDEO INSPECTION ==========")
    print("Total videos found:", len(videos))

    total_frames = 0

    for index, video_path in enumerate(videos, start=1):
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"\n{index}. {video_path.name}")
            print("   ERROR: Video open nahi hui.")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        duration = frame_count / fps if fps > 0 else 0
        total_frames += frame_count

        print(f"\n{index}. {video_path.name}")
        print(f"   Resolution : {width} x {height}")
        print(f"   FPS        : {fps:.2f}")
        print(f"   Frames     : {frame_count}")
        print(f"   Duration   : {duration:.2f} seconds")

        cap.release()

    print("\n========== SUMMARY ==========")
    print("Videos inspected:", len(videos))
    print("Total frames:", total_frames)
    print("============================================")

if __name__ == "__main__":
    inspect_videos()

