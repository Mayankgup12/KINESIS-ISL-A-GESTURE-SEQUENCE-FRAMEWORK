import cv2
import pandas as pd
import mediapipe as mp
import joblib
import numpy as np
import time
import pyttsx3
import threading

# Load trained model
model = joblib.load("models/gesture_model.pkl")

def speak_text(text):
    def run_speech():
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()

    threading.Thread(target=run_speech, daemon=True).start()

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Webcam setup
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam could not be opened.")
    hands.close()
    raise SystemExit

# Sentence builder variables
sentence = []
last_added_gesture = None
last_added_time = 0

# Settings
CONFIDENCE_THRESHOLD = 0.75
GESTURE_COOLDOWN = 1.5

print("Controls:")
print("  Hold gesture to recognize")
print("  SPACE = Add space")
print("  BACKSPACE = Remove last word")
print("  S = Speak sentence")
print("  C = Clear sentence")
print("  Q = Quit")

while True:
    success, frame = cap.read()

    if not success:
        print("Could not read webcam frame.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    prediction_text = "No hand detected"
    current_time = time.time()

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
    

        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        features = []

        for landmark in hand_landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])

        input_data = pd.DataFrame(
             [features],
        columns=model.feature_names_in_
)

        probabilities = model.predict_proba(input_data)[0]
        best_index = np.argmax(probabilities)

        confidence = float(probabilities[best_index])
        predicted_label = model.classes_[best_index]

        if confidence >= CONFIDENCE_THRESHOLD:
            prediction_text = (
                f"{predicted_label} ({confidence * 100:.1f}%)"
            )

            # Add a gesture only once per cooldown period.
            if (
                predicted_label != last_added_gesture
                and current_time - last_added_time >= GESTURE_COOLDOWN
            ):
                sentence.append(predicted_label)
                last_added_gesture = predicted_label
                last_added_time = current_time
                print("Sentence:", " ".join(sentence))
        else:
            prediction_text = "Uncertain gesture"
    else:
        last_added_gesture = None


    # Draw prediction panel
    cv2.rectangle(frame, (10, 10), (620, 75), (0, 0, 0), -1)

    cv2.putText(
        frame,
        "Prediction: " + prediction_text,
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # Display sentence
    sentence_text = " ".join(sentence)

    cv2.rectangle(
        frame,
        (10, frame.shape[0] - 90),
        (frame.shape[1] - 10, frame.shape[0] - 15),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        "Sentence:",
        (20, frame.shape[0] - 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # Show the sentence, truncated if it is too long for the window.
    display_sentence = sentence_text[-65:]

    cv2.putText(
        frame,
        display_sentence,
        (20, frame.shape[0] - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow("Kinesis-ISL | Sentence Builder", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("c"):
        sentence.clear()
        last_added_gesture = None
        last_added_time = 0
        print("Sentence cleared.")

    elif key == ord(" ") and sentence:
        sentence.append(" ")

    elif key == 8 and sentence:
        sentence.pop()
        print("Sentence:", " ".join(sentence))

    elif key == ord("s"):
        text_to_speak = " ".join(sentence)

        if text_to_speak.strip():
            print("Speaking:", text_to_speak)
            speak_text(text_to_speak)
        else:
            print("Sentence is empty!")

cap.release()
hands.close()
cv2.destroyAllWindows()