import tkinter as tk
from tkinter import messagebox
from collections import deque
from threading import Thread
import time

import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3

from PIL import Image, ImageTk, ImageDraw

from sentence_engine import SentenceFormationEngine


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_FILE = "models/sequence_model_merged_test.pkl"

SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.40
PREDICTION_COOLDOWN = 1.5

GESTURE_TEXT = {
    "HELLO": "Hello",
    "YES": "Yes",
    "NO": "No",
    "THANK_YOU": "Thank You",
    "STOP": "Stop"
}

BG = "#0B1120"
PANEL = "#151F32"
PANEL_LIGHT = "#1C2940"
ACCENT = "#38BDF8"
GREEN = "#22C55E"
WHITE = "#F1F5F9"
MUTED = "#94A3B8"
RED = "#F87171"


# ==========================================
# MAIN APPLICATION
# ==========================================

class KinesisDashboard:

    def __init__(self, root):

        self.root = root
        self.root.title("Kinesis-ISL | Sign Language Translation")
        self.root.geometry("1100x720")
        self.root.minsize(950, 650)
        self.root.configure(bg=BG)

        # Load trained model
        try:
            self.model = joblib.load(MODEL_FILE)

            if getattr(self.model, "n_features_in_", None) != 1890:
                raise ValueError(
                    "Model feature count does not match 1890."
                )

        except Exception as error:
            messagebox.showerror(
                "Model Loading Error",
                f"Unable to load model:\n{error}"
            )
            self.root.destroy()
            return

        # MediaPipe
        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.mp_draw = mp.solutions.drawing_utils

        # Application state
        self.cap = None
        self.camera_running = False

        self.sequence = deque(maxlen=SEQUENCE_LENGTH)
        self.recent_predictions = deque(maxlen=3)

        self.last_prediction_time = 0

        self.engine = SentenceFormationEngine()

        self.camera_photo = None

        self.current_gesture = "Waiting..."
        self.current_confidence = 0.0

        self.create_dashboard()
        self.show_camera_placeholder()

        self.root.protocol("WM_DELETE_WINDOW", self.close_app)

    # ======================================
    # DASHBOARD UI
    # ======================================

    def create_dashboard(self):

        # Header
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=25, pady=(18, 12))

        title = tk.Label(
            header,
            text="KINESIS-ISL",
            font=("Segoe UI", 25, "bold"),
            bg=BG,
            fg=ACCENT
        )
        title.pack(side="left")

        subtitle = tk.Label(
            header,
            text="Gesture-Sequence Based Contextual Sign Language Translation",
            font=("Segoe UI", 10),
            bg=BG,
            fg=MUTED
        )
        subtitle.pack(side="left", padx=18, pady=(10, 0))

        # Main content
        content = tk.Frame(self.root, bg=BG)
        content.pack(fill="both", expand=True, padx=25, pady=10)

        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2)
        content.rowconfigure(0, weight=1)

        # Camera panel
        camera_panel = tk.Frame(
            content,
            bg=PANEL,
            highlightbackground="#263650",
            highlightthickness=1
        )
        camera_panel.grid(
            row=0, column=0,
            sticky="nsew",
            padx=(0, 12)
        )

        camera_title = tk.Label(
            camera_panel,
            text="LIVE CAMERA FEED",
            font=("Segoe UI", 12, "bold"),
            bg=PANEL,
            fg=WHITE
        )
        camera_title.pack(anchor="w", padx=18, pady=15)

        self.camera_label = tk.Label(
            camera_panel,
            bg="#080D18",
            text="Camera is currently stopped",
            fg=MUTED,
            font=("Segoe UI", 13)
        )
        self.camera_label.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=(0, 15)
        )

        # Right information panel
        right_panel = tk.Frame(content, bg=BG)
        right_panel.grid(row=0, column=1, sticky="nsew")

        # Gesture card
        gesture_card = tk.Frame(
            right_panel,
            bg=PANEL,
            highlightbackground="#263650",
            highlightthickness=1
        )
        gesture_card.pack(fill="x", pady=(0, 12))

        tk.Label(
            gesture_card,
            text="DETECTED GESTURE",
            font=("Segoe UI", 10, "bold"),
            bg=PANEL,
            fg=MUTED
        ).pack(anchor="w", padx=18, pady=(16, 5))

        self.gesture_label = tk.Label(
            gesture_card,
            text="Waiting...",
            font=("Segoe UI", 25, "bold"),
            bg=PANEL,
            fg=ACCENT
        )
        self.gesture_label.pack(anchor="w", padx=18, pady=5)

        self.confidence_label = tk.Label(
            gesture_card,
            text="Confidence: 0%",
            font=("Segoe UI", 10),
            bg=PANEL,
            fg=MUTED
        )
        self.confidence_label.pack(anchor="w", padx=18, pady=(5, 18))

        # Sentence card
        sentence_card = tk.Frame(
            right_panel,
            bg=PANEL,
            highlightbackground="#263650",
            highlightthickness=1
        )
        sentence_card.pack(fill="both", expand=True, pady=(0, 12))

        tk.Label(
            sentence_card,
            text="GENERATED SENTENCE",
            font=("Segoe UI", 10, "bold"),
            bg=PANEL,
            fg=MUTED
        ).pack(anchor="w", padx=18, pady=(16, 12))

        self.sentence_label = tk.Label(
            sentence_card,
            text="Your translated sentence will appear here.",
            font=("Segoe UI", 14),
            bg=PANEL,
            fg=WHITE,
            wraplength=300,
            justify="left",
            anchor="nw"
        )
        self.sentence_label.pack(
            fill="both",
            expand=True,
            padx=18,
            pady=(0, 15)
        )

        # Controls
        controls = tk.Frame(right_panel, bg=BG)
        controls.pack(fill="x")

        self.start_button = tk.Button(
            controls,
            text="▶  Start Camera",
            command=self.start_camera,
            bg=ACCENT,
            fg="#06111F",
            activebackground="#7DD3FC",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            cursor="hand2",
            pady=11
        )
        self.start_button.pack(fill="x", pady=4)

        self.stop_button = tk.Button(
            controls,
            text="■  Stop Camera",
            command=self.stop_camera,
            bg=PANEL_LIGHT,
            fg=WHITE,
            activebackground="#334155",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            cursor="hand2",
            pady=11
        )
        self.stop_button.pack(fill="x", pady=4)

        self.speak_button = tk.Button(
            controls,
            text="🔊  Speak Sentence",
            command=self.speak_sentence,
            bg=PANEL_LIGHT,
            fg=WHITE,
            activebackground="#334155",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            cursor="hand2",
            pady=11
        )
        self.speak_button.pack(fill="x", pady=4)

        self.clear_button = tk.Button(
            controls,
            text="↻  Clear Sentence",
            command=self.clear_sentence,
            bg=PANEL_LIGHT,
            fg=WHITE,
            activebackground="#334155",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            cursor="hand2",
            pady=11
        )
        self.clear_button.pack(fill="x", pady=4)

        # Status bar
        footer = tk.Frame(
            self.root,
            bg=PANEL,
            height=38
        )
        footer.pack(fill="x", side="bottom")

        self.status_label = tk.Label(
            footer,
            text="● System Ready",
            font=("Segoe UI", 9),
            bg=PANEL,
            fg=GREEN
        )
        self.status_label.pack(side="left", padx=25, pady=9)

        tk.Label(
            footer,
            text="Kinesis-ISL | AI & Data Science",
            font=("Segoe UI", 9),
            bg=PANEL,
            fg=MUTED
        ).pack(side="right", padx=25)

    # ======================================
    # CAMERA PLACEHOLDER
    # ======================================

    def show_camera_placeholder(self):

        image = Image.new("RGB", (640, 420), "#080D18")

        draw = ImageDraw.Draw(image)

        draw.multiline_text(
            (320, 210),
            "KINESIS-ISL\n\nCamera Feed",
            fill="#64748B",
            anchor="mm",
            align="center",
            spacing=12
        )

        self.camera_photo = ImageTk.PhotoImage(image)

        self.camera_label.configure(
            image=self.camera_photo,
            text=""
        )

    # ======================================
    # START CAMERA
    # ======================================

    def start_camera(self):

        if self.camera_running:
            return

        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None

            messagebox.showerror(
                "Camera Error",
                "Unable to access webcam."
            )
            return

        self.sequence.clear()
        self.recent_predictions.clear()

        self.last_prediction_time = 0
        self.camera_running = True

        self.status_label.config(
            text="CAMERA RUNNING",
            fg=GREEN
        )

        self.start_button.config(state="disabled")

        self.update_camera()

    # ======================================
    # CAMERA PROCESSING
    # ======================================

    def update_camera(self):

        if not self.camera_running or self.cap is None:
            return

        ret, frame = self.cap.read()

        if not ret:
            self.stop_camera()
            self.status_label.config(
                text="CAMERA READ ERROR",
                fg=RED
            )
            return

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

                features = []

                for landmark in hand_landmarks.landmark:

                    features.extend([
                        landmark.x,
                        landmark.y,
                        landmark.z
                    ])

                if len(features) == 63:

                    self.sequence.append(features)

                    if len(self.sequence) == SEQUENCE_LENGTH:
                        self.predict_gesture()

        else:
            self.sequence.clear()

        cv2.putText(
            frame,
            f"Gesture: {self.current_gesture}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (56, 189, 248),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {self.current_confidence:.1f}%",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (34, 197, 94),
            2
        )

        rgb_display = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(rgb_display)
        image.thumbnail((640, 420))

        self.camera_photo = ImageTk.PhotoImage(image)

        self.camera_label.configure(
            image=self.camera_photo,
            text=""
        )

        self.root.after(15, self.update_camera)

    # ======================================
    # GESTURE PREDICTION
    # ======================================

    def predict_gesture(self):

        current_time = time.time()

        if current_time - self.last_prediction_time < PREDICTION_COOLDOWN:
            return

        input_data = np.array(
            self.sequence,
            dtype=np.float32
        ).flatten().reshape(1, -1)

        probabilities = self.model.predict_proba(input_data)[0]

        best_index = np.argmax(probabilities)
        confidence = probabilities[best_index]

        if confidence < CONFIDENCE_THRESHOLD:
            return

        prediction = self.model.classes_[best_index]

        self.recent_predictions.append(prediction)

        if len(self.recent_predictions) < 3:
            return

        stable_gesture = max(
            set(self.recent_predictions),
            key=self.recent_predictions.count
        )

        self.last_prediction_time = current_time

        self.current_gesture = GESTURE_TEXT.get(
            stable_gesture,
            stable_gesture
        )

        self.current_confidence = confidence * 100

        self.gesture_label.config(
            text=self.current_gesture
        )

        self.confidence_label.config(
            text=f"Confidence: {self.current_confidence:.1f}%"
        )

        sentence = self.engine.add_gesture(stable_gesture)

        self.sentence_label.config(text=sentence)

    # ======================================
    # TEXT TO SPEECH
    # ======================================

    def speak_sentence(self):

        sentence = self.sentence_label.cget("text")

        if (
            not sentence
            or sentence == "Your translated sentence will appear here."
        ):
            return

        Thread(
            target=self.speak_text,
            args=(sentence,),
            daemon=True
        ).start()

    def speak_text(self, sentence):

        try:

            engine = pyttsx3.init()
            engine.setProperty("rate", 145)

            engine.say(sentence)
            engine.runAndWait()
            engine.stop()

        except Exception as error:

            print("Speech Error:", error)

    # ======================================
    # CLEAR SENTENCE
    # ======================================

    def clear_sentence(self):

        self.engine = SentenceFormationEngine()

        self.sequence.clear()
        self.recent_predictions.clear()

        self.current_gesture = "Waiting..."
        self.current_confidence = 0.0

        self.gesture_label.config(text="Waiting...")

        self.confidence_label.config(
            text="Confidence: 0%"
        )

        self.sentence_label.config(
            text="Your translated sentence will appear here."
        )

        self.status_label.config(
            text="SENTENCE CLEARED",
            fg=GREEN
        )

    # ======================================
    # STOP CAMERA
    # ======================================

    def stop_camera(self):

        self.camera_running = False

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.sequence.clear()
        self.recent_predictions.clear()

        self.start_button.config(state="normal")

        self.status_label.config(
            text="CAMERA STOPPED",
            fg=MUTED
        )

        self.show_camera_placeholder()

    # ======================================
    # CLOSE APPLICATION
    # ======================================

    def close_app(self):

        self.stop_camera()

        if self.hands:
            self.hands.close()

        self.root.destroy()


# ==========================================
# APPLICATION ENTRY POINT
# ==========================================

if __name__ == "__main__":

    root = tk.Tk()

    app = KinesisDashboard(root)

    root.mainloop()
