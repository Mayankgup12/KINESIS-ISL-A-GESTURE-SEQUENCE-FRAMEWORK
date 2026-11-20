class SentenceFormationEngine:

    def __init__(self):
        self.gesture_sequence = []

        self.gesture_words = {
            "HELLO": "hello",
            "YES": "yes",
            "NO": "no",
            "THANK_YOU": "thank you",
            "STOP": "stop"
        }

    def add_gesture(self, gesture):

        if gesture not in self.gesture_words:
            return self.get_sentence()

        # Prevent consecutive duplicate gestures.
        if (
            self.gesture_sequence
            and self.gesture_sequence[-1] == gesture
        ):
            return self.get_sentence()

        self.gesture_sequence.append(gesture)

        return self.get_sentence()

    def get_sentence(self):

        gestures = self.gesture_sequence

        if not gestures:
            return ""

        # Rule 1: Greeting + gratitude.
        if gestures == ["HELLO", "THANK_YOU"]:
            return "Hello! Thank you."

        # Rule 2: Greeting + agreement.
        if gestures == ["HELLO", "YES"]:
            return "Hello! Yes, I agree."

        # Rule 3: Greeting + disagreement.
        if gestures == ["HELLO", "NO"]:
            return "Hello! No, I disagree."

        # Rule 4: Greeting + agreement + gratitude.
        if gestures == ["HELLO", "YES", "THANK_YOU"]:
            return "Hello! Yes, thank you."

        # Rule 5: Greeting + disagreement + gratitude.
        if gestures == ["HELLO", "NO", "THANK_YOU"]:
            return "Hello! No, thank you."

        # Rule 6: STOP gesture.
        if gestures == ["STOP"]:
            return "Please stop."

        # Rule 7: Agreement.
        if gestures == ["YES"]:
            return "Yes, I agree."

        # Rule 8: Disagreement.
        if gestures == ["NO"]:
            return "No, I disagree."

        # Rule 9: Gratitude.
        if gestures == ["THANK_YOU"]:
            return "Thank you."

        # Rule 10: Greeting.
        if gestures == ["HELLO"]:
            return "Hello!"

        # Fallback for combinations without a defined rule.
        words = [
            self.gesture_words[g]
            for g in gestures
        ]

        sentence = " ".join(words)

        return sentence.capitalize() + "."

    def clear(self):
        self.gesture_sequence.clear()


if __name__ == "__main__":

    engine = SentenceFormationEngine()

    test_cases = [
        ["HELLO"],
        ["HELLO", "YES"],
        ["HELLO", "NO"],
        ["HELLO", "YES", "THANK_YOU"],
        ["HELLO", "NO", "THANK_YOU"],
        ["STOP"]
    ]

    for gestures in test_cases:

        engine.clear()

        for gesture in gestures:
            sentence = engine.add_gesture(gesture)

        print("Gestures:", " -> ".join(gestures))
        print("Generated sentence:", sentence)
        print("-" * 40)