import cv2
import mediapipe as mp
import csv
import os
import time

# ==============================
# KINESIS-ISL SEQUENCE COLLECTOR
# ==============================

DATA_DIR = "dataset"
CSV_FILE = os.path.join(DATA_DIR, "gesture_sequences_fresh.csv")

SEQUENCE_LENGTH = 30
NUM_LANDMARKS = 21

os.makedirs(DATA_DIR, exist_ok=True)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Create CSV headers
headers = ["label"] + [
    f"frame{frame}_{axis}{point}"
    for frame in range(SEQUENCE_LENGTH)
    for point in range(NUM_LANDMARKS)
    for axis in ("x", "y", "z")
]

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)

print("\n=== Kinesis-ISL Sequence Collection ===")
print("1 = HELLO")
print("2 = YES")
print("3 = NO")
print("4 = THANK_YOU")
print("5 = STOP")
print("0 = Pause")
print("Q = Quit")
print(f"Each sequence contains {SEQUENCE_LENGTH} frames.\n")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam could not be opened.")
    hands.close()
    raise SystemExit

labels = {
    ord("1"): "HELLO",
    ord("2"): "YES",
    ord("3"): "NO",
    ord("4"): "THANK_YOU",
    ord("5"): "STOP"
}

current_label = None
recording = False
sequence_frames = []
sample_count = 0
message = "Press 1-5 to record a sequence"

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Unable to read webcam.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        hand_detected = False

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            hand_detected = True

            # Capture one frame of the sequence
            if recording:
                frame_data = []

                for point in hand_landmarks.landmark:
                    frame_data.extend([
                        point.x,
                        point.y,
                        point.z
                    ])

                sequence_frames.append(frame_data)

                if len(sequence_frames) >= SEQUENCE_LENGTH:
                    # Save only if all frames were captured
                    row = [current_label]

                    for captured_frame in sequence_frames:
                        row.extend(captured_frame)

                    with open(CSV_FILE, "a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow(row)

                    sample_count += 1
                    recording = False
                    sequence_frames = []

                    message = f"Saved {current_label} sequence!"
                    print(message)

        # Display status
        if recording:
            status = (
                f"Recording {current_label}: "
                f"{len(sequence_frames)}/{SEQUENCE_LENGTH}"
            )
        else:
            status = message

        cv2.putText(
            frame,
            status[:55],
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Hand detected: {'Yes' if hand_detected else 'No'}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0) if hand_detected else (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"Sequences saved: {sample_count}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        cv2.imshow("Kinesis-ISL Sequence Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("0") and not recording:
            current_label = None
            message = "Collection paused"

        elif key in labels and not recording:
            current_label = labels[key]
            sequence_frames = []
            recording = True

            message = f"Perform gesture: {current_label}"
            print(f"\nRecording {current_label}...")

        # If a hand disappears, reject the incomplete sequence
        # rather than saving a misleading sample.
        if recording and not hand_detected:
            recording = False
            sequence_frames = []
            message = "Hand lost! Sequence discarded. Try again."
            print(message)

finally:
    cap.release()
    hands.close()
    cv2.destroyAllWindows()

print(f"\nTotal sequences saved: {sample_count}")
print(f"Dataset file: {CSV_FILE}")