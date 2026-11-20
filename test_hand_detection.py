import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

hands = mp.solutions.hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print("Camera test started. Press Q to exit.")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Camera frame failed")
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            print("HAND DETECTED")

            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS
                )

        cv2.imshow("Hand Detection Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    hands.close()
    cv2.destroyAllWindows()