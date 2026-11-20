import cv2
import mediapipe as mp
import csv
import os
import time

# ==================================================
# KINESIS-ISL: STATIC THANK_YOU DATA COLLECTOR
# ==================================================

DATA_DIR = "dataset"

SOURCE_CSV = os.path.join(
    DATA_DIR,
    "gesture_sequences_fresh.csv"
)

OUTPUT_CSV = os.path.join(
    DATA_DIR,
    "thankyou_static_new.csv"
)

LABEL = "THANK_YOU"

SEQUENCE_LENGTH = 30
FEATURES_PER_FRAME = 63
TOTAL_FEATURES = SEQUENCE_LENGTH * FEATURES_PER_FRAME

TARGET_SAMPLES = 30

# Increased timeout
RECORDING_TIMEOUT = 60

WINDOW_NAME = "Kinesis-ISL Static THANK_YOU Collector"

os.makedirs(DATA_DIR, exist_ok=True)


# ==================================================
# VALIDATE ORIGINAL DATASET
# ==================================================

if not os.path.exists(SOURCE_CSV):
    raise FileNotFoundError(
        "Original dataset not found!"
    )

with open(SOURCE_CSV, "r", newline="") as file:
    reader = csv.reader(file)
    headers = next(reader)

if len(headers) != 1891 or headers[0] != "label":
    raise ValueError(
        "Original dataset format is incorrect!"
    )


# ==================================================
# CREATE SEPARATE OUTPUT FILE
# ==================================================

if not os.path.exists(OUTPUT_CSV):

    with open(OUTPUT_CSV, "w", newline="") as file:

        writer = csv.writer(file)
        writer.writerow(headers)


# ==================================================
# CHECK EXISTING SAMPLES
# ==================================================

with open(OUTPUT_CSV, "r", newline="") as file:

    rows = list(csv.reader(file))


if not rows or rows[0] != headers:
    raise ValueError(
        "New dataset header is incorrect!"
    )

existing_count = len(rows) - 1

if existing_count > TARGET_SAMPLES:
    raise ValueError(
        "Unexpected number of existing samples!"
    )

for row in rows[1:]:

    if len(row) != 1891 or row[0] != LABEL:
        raise ValueError(
            "Invalid row found in new dataset!"
        )


# ==================================================
# MEDIAPIPE SETUP
# ==================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(

    static_image_mode=False,

    max_num_hands=1,

    model_complexity=1,

    min_detection_confidence=0.5,

    min_tracking_confidence=0.5

)


# ==================================================
# CAMERA SETUP
# ==================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    hands.close()

    raise RuntimeError(
        "Camera could not be opened!"
    )

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# ==================================================
# FEATURE EXTRACTION
# ==================================================

def extract_landmarks(hand_landmarks):

    features = []

    for landmark in hand_landmarks.landmark:

        features.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    return features


# ==================================================
# DRAW TEXT
# ==================================================

def draw_text(
    frame,
    text,
    position,
    color=(0, 255, 0),
    scale=0.7
):

    cv2.putText(

        frame,
        text,
        position,

        cv2.FONT_HERSHEY_SIMPLEX,

        scale,

        color,

        2

    )


# ==================================================
# DISPLAY INSTRUCTIONS
# ==================================================

print("\n======================================")
print(" KINESIS-ISL STATIC THANK_YOU COLLECTOR")
print("======================================")

print("Existing samples:", existing_count)
print("Target samples:", TARGET_SAMPLES)

print("\nGesture Instructions:")

print("1. Fold your thumb towards your palm.")
print("2. Keep the other four fingers straight.")
print("3. Keep all four fingers together.")
print("4. Hold your hand steady.")

print("\nControls:")
print("T = Record sample")
print("Q = Quit")

print("\nRecording timeout:", RECORDING_TIMEOUT, "seconds")

print("======================================\n")


# ==================================================
# RECORD ONE SAMPLE
# ==================================================

def record_sample():

    sequence = []

    detected_frames = 0

    missed_frames = 0

    print("\n--------------------------------------")
    print("Preparing for THANK_YOU sample...")
    print("--------------------------------------")

    # Countdown

    for count in [3, 2, 1]:

        ret, frame = cap.read()

        if not ret:

            print("Camera frame failed.")
            return None

        frame = cv2.flip(frame, 1)

        draw_text(
            frame,
            f"Starting in {count}",
            (100, 100),
            (0, 255, 255),
            1.5
        )

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1000) & 0xFF

        if key == ord("q"):

            print("Recording cancelled.")
            return None

    print("\nRecording started!")
    print("Hold your THANK_YOU pose steadily.")

    # Timeout starts AFTER countdown

    start_time = time.time()

    # Recording loop

    while len(sequence) < TOTAL_FEATURES:

        elapsed_time = time.time() - start_time

        remaining_time = max(
            0,
            RECORDING_TIMEOUT - int(elapsed_time)
        )

        if elapsed_time > RECORDING_TIMEOUT:

            print("\nTIMEOUT!")

            print(
                "Detected frames:",
                detected_frames
            )

            print(
                "Missed frames:",
                missed_frames
            )

            print(
                "Collected features:",
                len(sequence)
            )

            print(
                "Required features:",
                TOTAL_FEATURES
            )

            return None

        ret, frame = cap.read()

        if not ret:

            missed_frames += 1
            continue

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb)

        # ------------------------------------------
        # HAND NOT DETECTED
        # ------------------------------------------

        if not results.multi_hand_landmarks:

            missed_frames += 1

            draw_text(
                frame,
                "SHOW YOUR HAND",
                (20, 40),
                (0, 0, 255)
            )

        # ------------------------------------------
        # HAND DETECTED
        # ------------------------------------------

        else:

            hand_landmarks = (
                results.multi_hand_landmarks[0]
            )

            features = extract_landmarks(
                hand_landmarks
            )

            if len(features) == FEATURES_PER_FRAME:

                sequence.extend(features)

                detected_frames += 1

                mp_draw.draw_landmarks(

                    frame,

                    hand_landmarks,

                    mp_hands.HAND_CONNECTIONS

                )

        # ------------------------------------------
        # LIVE PROGRESS
        # ------------------------------------------

        progress = (
            len(sequence) // FEATURES_PER_FRAME
        )

        draw_text(
            frame,
            f"Recording: {progress}/{SEQUENCE_LENGTH}",
            (20, 40),
            (0, 255, 0)
        )

        draw_text(
            frame,
            f"Detected: {detected_frames}",
            (20, 75),
            (255, 255, 0)
        )
        draw_text(
            frame,
            f"Time left: {remaining_time}s",
            (20, 110),
            (0, 255, 255)
        )

        cv2.imshow(
            WINDOW_NAME,
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            print("\nRecording cancelled by user.")

            return None

    # ------------------------------------------
    # VALIDATE SAMPLE
    # ------------------------------------------

    if len(sequence) != TOTAL_FEATURES:

        print("Invalid feature count.")

        return None

    print("\nSample recording completed!")

    print("Detected frames:", detected_frames)

    print("Total features:", len(sequence))

    return sequence


# ==================================================
# MAIN CAMERA LOOP
# ==================================================

try:

    while existing_count < TARGET_SAMPLES:

        ret, frame = cap.read()

        if not ret:

            print("Camera frame could not be read.")

            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb)

        # Draw detected hand

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(

                    frame,

                    hand_landmarks,

                    mp_hands.HAND_CONNECTIONS

                )

        # Display information

        draw_text(
            frame,
            f"Samples: {existing_count}/{TARGET_SAMPLES}",
            (20, 35),
            (0, 255, 0)
        )

        draw_text(
            frame,
            "T = Record | Q = Quit",
            (20, 70),
            (255, 255, 255)
        )

        draw_text(
            frame,
            "Hold static THANK_YOU pose",
            (20, 110),
            (0, 255, 255)
        )

        cv2.imshow(
            WINDOW_NAME,
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        # Quit

        if key == ord("q"):

            print("\nCollection stopped by user.")

            break

        # Record

        if key == ord("t"):

            sequence = record_sample()

            if sequence is not None:

                with open(
                    OUTPUT_CSV,
                    "a",
                    newline=""
                ) as file:

                    writer = csv.writer(file)

                    writer.writerow(
                        [LABEL] + sequence
                    )

                existing_count += 1

                print("\n======================================")

                print(
                    f"SUCCESS: Sample {existing_count}/{TARGET_SAMPLES} saved."
                )

                print("======================================")

            else:

                print(
                    "\nSample discarded. Please try again."
                )

            time.sleep(0.5)

    # ==================================================
    # COLLECTION COMPLETED
    # ==================================================

    if existing_count >= TARGET_SAMPLES:

        print("\n======================================")

        print("THANK_YOU COLLECTION COMPLETED!")

        print(
            "Total samples:",
            existing_count
        )

        print(
            "Saved file:",
            OUTPUT_CSV
        )

        print("======================================")


finally:

    cap.release()

    hands.close()

    cv2.destroyAllWindows()

    print("\nCamera released.")

    print("Output file:", OUTPUT_CSV)

    print("Original dataset was not modified.")