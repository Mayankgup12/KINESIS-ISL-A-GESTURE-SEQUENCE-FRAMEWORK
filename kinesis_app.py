import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3
import tkinter as tk
import time

from tkinter import messagebox
from collections import deque
from threading import Thread
from sentence_engine import SentenceFormationEngine


# ==========================================
# KINESIS-ISL CONFIGURATION
# ==========================================

MODEL_FILE = "models/sequence_model_merged_test.pkl"

SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.40
PREDICTION_COOLDOWN = 1.5

GESTURE_TEXT = {
    "HELLO": "Hello",
    "YES": "Yes",
    "NO": "No",
    "THANK_YOU": "Thank you",
    "STOP": "Stop",
}


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading Kinesis-ISL model...")

try:
    model = joblib.load(MODEL_FILE)
    print("Model loaded successfully!")
    print("Supported gestures:", list(model.classes_))

    # Check that the loaded model matches the app's sequence format.
    expected_features = SEQUENCE_LENGTH * 21 * 3
    if getattr(model, "n_features_in_", None) != expected_features:
        raise ValueError(
            f"Model expects {getattr(model, 'n_features_in_', 'unknown')} "
            f"features, but the app generates {expected_features}."
        )

except Exception as error:
    raise SystemExit(
        f"Unable to load a compatible model: {error}\n"
        "Check models/sequence_model.pkl."
    )


# ==========================================
# MEDIAPIPE SETUP
# ==========================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)


# ==========================================
# APPLICATION VARIABLES
# ==========================================

cap = None
camera_running = False

sequence = deque(maxlen=SEQUENCE_LENGTH)
recent_predictions = deque(maxlen=3)

sentence_engine = SentenceFormationEngine()

last_prediction_time = 0.0
last_added_gesture = None

current_gesture = "Waiting..."
current_confidence = 0.0


# ==========================================
# TEXT-TO-SPEECH
# ==========================================

def speak_text():
    text = sentence_var.get().strip()

    if not text:
        status_var.set("Please recognize a gesture first.")
        return

    def speak_in_background():
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            engine.stop()

            root.after(
                0,
                lambda: status_var.set("Speech completed."),
            )

        except Exception as error:
            root.after(
                0,
                lambda err=error: status_var.set(f"Speech error: {err}"),
            )

    Thread(target=speak_in_background, daemon=True).start()


# ==========================================
# CLEAR SENTENCE
# ==========================================

def clear_sentence():
    global last_added_gesture

    sentence_engine.clear()
    sentence_var.set("")
    last_added_gesture = None

    status_var.set("Sentence cleared.")


# ==========================================
# START CAMERA
# ==========================================

def start_camera():
    global cap, camera_running
    global last_prediction_time, last_added_gesture
    global current_gesture, current_confidence

    if camera_running:
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        cap.release()
        cap = None
        messagebox.showerror(
            "Camera Error",
            "Unable to open webcam. Close other camera apps and try again.",
        )
        return

    camera_running = True

    sequence.clear()
    recent_predictions.clear()

    last_prediction_time = 0.0
    last_added_gesture = None
    current_gesture = "Show your hand"
    current_confidence = 0.0

    status_var.set("Camera running. Show your hand.")
    update_camera()


# ==========================================
# STOP CAMERA
# ==========================================

def stop_camera():
    global cap, camera_running
    global current_gesture, current_confidence

    camera_running = False

    if cap is not None:
        cap.release()
        cap = None

    sequence.clear()
    recent_predictions.clear()

    cv2.destroyAllWindows()

    current_gesture = "Camera stopped"
    current_confidence = 0.0

    gesture_var.set(current_gesture)
    confidence_var.set("Confidence: --")
    status_var.set("Camera stopped.")


# ==========================================
# UPDATE CAMERA AND PREDICT
# ==========================================

def update_camera():
    global camera_running
    global current_gesture, current_confidence
    global last_prediction_time, last_added_gesture

    if not camera_running or cap is None:
        return

    ret, frame = cap.read()

    if not ret:
        stop_camera()
        status_var.set("Unable to read webcam.")
        return

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
        )

        # One frame = 21 landmarks x (x, y, z) = 63 values.
        frame_features = []

        for landmark in hand_landmarks.landmark:
            frame_features.extend([
                landmark.x,
                landmark.y,
                landmark.z,
            ])

        sequence.append(frame_features)

        # Predict only after collecting 30 consecutive frames.
        if len(sequence) == SEQUENCE_LENGTH:
            now = time.time()

            if now - last_prediction_time >= PREDICTION_COOLDOWN:
                # Shape: (30, 63); flattened shape: (1890,).
                live_sequence = np.asarray(sequence, dtype=np.float32)
                features = live_sequence.flatten()

                # Diagnostic information for checking live input.
                print(
                    "\nLive sequence shape:",
                    live_sequence.shape,
                    "| Flattened feature count:",
                    features.size,
                    flush=True,
                )

                print(
                    "Live wrist Y (first, middle, last):",
                    round(float(live_sequence[0, 1]), 4),
                    round(float(live_sequence[15, 1]), 4),
                    round(float(live_sequence[29, 1]), 4),
                    flush=True,
                )

                # Ensure that the model receives exactly 1 x 1890 features.
                features = features.reshape(1, -1)

                # IMPORTANT: calculate probabilities before using them below.
                probabilities = model.predict_proba(features)[0]
                print("DEBUG: Prediction started", flush=True)

                probabilities = model.predict_proba(features)[0]

                print("DEBUG: Probabilities calculated:", probabilities, flush=True)

                # Diagnostic: display each class probability.
                probability_info = " | ".join(
                    f"{gesture}: {probabilities[i]:.2f}"
                    for i, gesture in enumerate(model.classes_)
                )

                print(
                    f"Live probabilities: {probability_info}",
                    flush=True,
                )

                best_index = int(np.argmax(probabilities))
                predicted_gesture = model.classes_[best_index]
                current_confidence = float(probabilities[best_index])

                if current_confidence >= CONFIDENCE_THRESHOLD:
                    current_gesture = predicted_gesture
                    recent_predictions.append(predicted_gesture)

                    # Require a majority among the last three predictions.
                    if len(recent_predictions) == 3:
                        stable_gesture = max(
                            set(recent_predictions),
                            key=recent_predictions.count,
                        )

                        if (
                            recent_predictions.count(stable_gesture) >= 2
                            and stable_gesture != last_added_gesture
                        ):
                            sentence = sentence_engine.add_gesture(
                                stable_gesture
                            )

                            sentence_var.set(sentence)
                            last_added_gesture = stable_gesture

                            word = GESTURE_TEXT.get(
                                stable_gesture,
                                stable_gesture,
                            )

                            status_var.set(f"Recognized: {word}")

                else:
                    current_gesture = "Uncertain"
                    recent_predictions.clear()

                last_prediction_time = now

    else:
        # Clear the sequence if the hand disappears.
        sequence.clear()
        recent_predictions.clear()

        current_gesture = "Show your hand"
        current_confidence = 0.0

        # Showing the hand again allows a gesture to be added again.
        last_added_gesture = None

    # Update application labels.
    gesture_var.set(f"Gesture: {current_gesture}")
    confidence_var.set(
        f"Confidence: {current_confidence * 100:.1f}%"
    )

    # Display webcam window.
    cv2.putText(
        frame,
        current_gesture,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        f"Frames: {len(sequence)}/{SEQUENCE_LENGTH}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Kinesis-ISL | Gesture Recognition", frame)

    # Press Q in the webcam window to stop the camera.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        stop_camera()
        return

    root.after(10, update_camera)


# ==========================================
# BUILD GUI
# ==========================================

root = tk.Tk()

root.title("Kinesis-ISL | Sign Gesture Recognition")
root.geometry("650x470")
root.resizable(False, False)

title_label = tk.Label(
    root,
    text="KINESIS-ISL",
    font=("Arial", 24, "bold"),
)
title_label.pack(pady=(15, 3))

subtitle_label = tk.Label(
    root,
    text="Gesture Sequence Recognition System",
    font=("Arial", 11),
)
subtitle_label.pack(pady=(0, 15))

gesture_var = tk.StringVar(value="Gesture: Waiting...")
confidence_var = tk.StringVar(value="Confidence: --")
sentence_var = tk.StringVar(value="")
status_var = tk.StringVar(value="Click Start Camera to begin.")

gesture_label = tk.Label(
    root,
    textvariable=gesture_var,
    font=("Arial", 16, "bold"),
)
gesture_label.pack(pady=5)

confidence_label = tk.Label(
    root,
    textvariable=confidence_var,
    font=("Arial", 11),
)
confidence_label.pack(pady=3)

sentence_heading = tk.Label(
    root,
    text="Recognized Sentence:",
    font=("Arial", 12, "bold"),
)
sentence_heading.pack(pady=(15, 5))

sentence_entry = tk.Entry(
    root,
    textvariable=sentence_var,
    font=("Arial", 14),
    width=45,
    justify="center",
)
sentence_entry.pack(pady=5, ipady=5)

button_frame = tk.Frame(root)
button_frame.pack(pady=20)

start_button = tk.Button(
    button_frame,
    text="Start Camera",
    command=start_camera,
    width=13,
)
start_button.grid(row=0, column=0, padx=5)

stop_button = tk.Button(
    button_frame,
    text="Stop Camera",
    command=stop_camera,
    width=13,
)
stop_button.grid(row=0, column=1, padx=5)

speak_button = tk.Button(
    button_frame,
    text="Speak",
    command=speak_text,
    width=13,
)
speak_button.grid(row=0, column=2, padx=5)

clear_button = tk.Button(
    button_frame,
    text="Clear",
    command=clear_sentence,
    width=13,
)
clear_button.grid(row=0, column=3, padx=5)

status_label = tk.Label(
    root,
    textvariable=status_var,
    font=("Arial", 10),
    wraplength=580,
)
status_label.pack(pady=5)

footer_label = tk.Label(
    root,
    text="Kinesis-ISL | Academic Minor Project",
    font=("Arial", 9),
)
footer_label.pack(side="bottom", pady=10)


# ==========================================
# CLOSE APPLICATION SAFELY
# ==========================================

def close_application():
    stop_camera()
    hands.close()
    root.destroy()


# ==========================================
# KEYBOARD SHORTCUTS
# ==========================================

def handle_keypress(event):
    key = event.keysym.lower()

    if key == "m":
        speak_text()

    elif key == "c":
        clear_sentence()

    elif key == "s":
        if camera_running:
            stop_camera()
        else:
            start_camera()

    elif key == "q":
        close_application()


root.bind("<KeyPress>", handle_keypress)
root.protocol("WM_DELETE_WINDOW", close_application)

print("Starting Kinesis-ISL application...")
root.mainloop()
