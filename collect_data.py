import cv2
import mediapipe as mp
import csv
import os
import time

# Configuration
DATA_DIR = "dataset"
CSV_FILE = os.path.join(DATA_DIR, "gestures_session2.csv")

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

# Create CSV with headers if it doesn't exist
headers = ["label"] + [
    f"{axis}{i}"
    for i in range(21)
    for axis in ("x", "y", "z")
]

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)

print("\n=== Kinesis-ISL Dataset Collection ===")
print("Labels: HELLO, YES, NO, THANK_YOU, STOP")
print("Camera par sign ready rakhna.")
print("Q = Quit\n")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam could not be opened.")
    hands.close()
    raise SystemExit

current_label = None
last_saved = 0
sample_count = 0

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Unable to read webcam.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            if current_label and time.time() - last_saved >= 0.15:
                row = [current_label]

                for point in hand_landmarks.landmark:
                    row.extend([point.x, point.y, point.z])

                with open(CSV_FILE, "a", newline="") as file:
                    csv.writer(file).writerow(row)

                sample_count += 1
                last_saved = time.time()

        status = current_label if current_label else "Press 1-5 to select label"

        cv2.putText(
            frame,
            f"Label: {status}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Session samples: {sample_count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.imshow("Kinesis-ISL Dataset Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("1"):
            current_label = "HELLO"
            print("Collecting: HELLO")

        elif key == ord("2"):
            current_label = "YES"
            print("Collecting: YES")

        elif key == ord("3"):
            current_label = "NO"
            print("Collecting: NO")

        elif key == ord("4"):
            current_label = "THANK YOU"
            print("Collecting: THANK YOU")

        elif key == ord("5"):
            current_label = "STOP"
            print("Collecting: STOP")

        elif key == ord("0"):
            current_label = None
            print("Collection paused.")

        elif key == ord("q"):
            break

finally:
    cap.release()
    hands.close()
    cv2.destroyAllWindows()

print(f"\nSession samples collected: {sample_count}")
print(f"Dataset file: {CSV_FILE}")