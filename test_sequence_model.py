import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from collections import deque

# ---------------- CONFIGURATION ----------------

MODEL_FILE = "models/sequence_model.pkl"
SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.40
PREDICTION_COOLDOWN = 1.5

# ---------------- LOAD MODEL ----------------

print("Loading Kinesis-ISL sequence model...")

model = joblib.load(MODEL_FILE)

print("Model loaded successfully!")
print("Supported gestures:", list(model.classes_))

# ---------------- MEDIAPIPE SETUP ----------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Store the latest 30 frames of hand landmarks.
sequence = deque(maxlen=SEQUENCE_LENGTH)

# ---------------- WEBCAM SETUP ----------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam could not be opened.")
    hands.close()
    raise SystemExit

last_prediction_time = 0
predicted_gesture = "Collecting frames..."
confidence = 0.0

print("\nWebcam started.")
print("Show a gesture and hold/perform it in front of the camera.")
print("Press Q to quit.")

# ---------------- MAIN LOOP ----------------

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Unable to read webcam.")
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Extract 21 landmarks × (x, y, z) = 63 features.
            frame_features = []

            for landmark in hand_landmarks.landmark:
                frame_features.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            sequence.append(frame_features)

            # Predict only after collecting 30 consecutive frames.
            if len(sequence) == SEQUENCE_LENGTH:
                current_time = time.time()

                if current_time - last_prediction_time >= PREDICTION_COOLDOWN:
                    sequence_features = np.array(sequence).flatten()
                    sequence_features = sequence_features.reshape(1, -1)

                    probabilities = model.predict_proba(sequence_features)[0]

                    best_index = np.argmax(probabilities)
                    confidence = float(probabilities[best_index])

                    if confidence >= CONFIDENCE_THRESHOLD:
                        predicted_gesture = model.classes_[best_index]
                    else:
                        predicted_gesture = "Uncertain"

                    last_prediction_time = current_time

        else:
            # Reset when the hand disappears.
            sequence.clear()
            predicted_gesture = "Show your hand"
            confidence = 0.0

        # ---------------- DISPLAY ----------------

        cv2.putText(
            frame,
            f"Gesture: {predicted_gesture}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence * 100:.1f}%",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Frames: {len(sequence)}/{SEQUENCE_LENGTH}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Press Q to quit",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow("Kinesis-ISL | Live Sequence Testing", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
    print("\nWebcam closed. Testing finished.")
