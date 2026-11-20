# Kinesis-ISL
### A Gesture-Sequence Framework for Contextual Sign Language Translation

## 1. Project Overview

Kinesis-ISL is a machine learning-based sign language gesture recognition system designed to recognize hand gesture sequences and convert them into meaningful contextual sentences.

Unlike basic gesture recognition systems that identify individual gestures and display their corresponding words, Kinesis-ISL incorporates a contextual sentence formation mechanism that combines recognized gestures to generate meaningful expressions.

The system also provides real-time camera-based gesture recognition through a graphical user interface and supports text-to-speech functionality.

## 2. Key Features

- Real-time hand gesture recognition.
- Gesture-sequence-based classification.
- Recognition of five gesture classes.
- Contextual sentence formation.
- Text-to-speech output.
- Graphical user interface with embedded camera feed.
- Model evaluation and performance reporting.

## 3. Supported Gestures

The current prototype supports:

| Gesture | Meaning |
|---|---|
| HELLO | Greeting |
| YES | Agreement |
| NO | Disagreement |
| THANK_YOU | Gratitude |
| STOP | Request to stop |

## 4. Machine Learning Approach

Kinesis-ISL uses the Random Forest Classification algorithm for gesture-sequence recognition.

### Model Configuration

- Algorithm: Random Forest Classifier
- Number of estimators: 300
- Random state: 42
- Class weighting: Balanced
- Input features: 1890
- Sequence length: 30 frames
- Hand landmarks: 21
- Coordinates: X, Y, Z

Each sequence contains:

30 frames × 21 landmarks × 3 coordinates = 1890 features.

### Dataset Information

The merged experimental dataset contains 130 gesture sequences.

| Gesture | Samples |
|---|---:|
| HELLO | 20 |
| YES | 20 |
| NO | 20 |
| THANK_YOU | 50 |
| STOP | 20 |
| **Total** | **130** |

The dataset was divided using an 80:20 stratified train-test split.

- Training samples: 104
- Testing samples: 26

## 5. Model Evaluation

The latest merged model achieved the following results on its held-out test split:

| Metric | Result |
|---|---:|
| Test Accuracy | 100% |
| Precision (macro avg) | 1.00 |
| Recall (macro avg) | 1.00 |
| F1-score (macro avg) | 1.00 |

Note: These results represent the current experimental dataset and test split. They do not establish generalization across unseen users, environments, or all forms of Indian Sign Language.

## 6. Contextual Sentence Formation

A key component of Kinesis-ISL is its sentence formation engine.

The system processes recognized gesture sequences and combines their meanings according to contextual rules.

Examples:

- HELLO → Hello!
- HELLO + YES → Hello! Yes, I agree.
- HELLO + NO → Hello! No, I disagree.
- HELLO + YES + THANK_YOU → Hello! Yes, thank you.
- STOP → Please stop.

This component differentiates Kinesis-ISL from systems that only provide isolated gesture-to-word mappings.

## 7. Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Scikit-learn
- Tkinter
- Pillow
- Joblib
- pyttsx3
- Matplotlib

## 8. Project Structure

```text
Kinesis ISL/
│
├── dataset/
├── models/
├── results/
├── external_dataset/
├── archive/
│
├── backup_before_frontend/
├── backup_before_thankyou_retraining/
│
├── kinesis_dashboard.py
├── kinesis_app.py
├── sentence_engine.py
├── train_merged_test_model.py
├── evaluate_sequence_model.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 9. Installation and Execution

### Step 1: Clone or download the project

Download the project files to your local system.

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

### Step 3: Activate the environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 4: Install dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Step 5: Run the application

```bash
python -u kinesis_dashboard.py
```

Allow camera access when prompted.

## 10. Future Scope

Potential future improvements include:

- Expanding the gesture vocabulary.
- Incorporating larger and more diverse datasets.
- Evaluating performance across multiple users.
- Exploring deep learning-based sequence recognition.
- Improving contextual sentence generation.
- Supporting more complex sentence structures.
- Exploring multilingual text and speech output.

## 11. Project Objective

The primary objective of Kinesis-ISL is to explore how machine learning-based gesture-sequence recognition and contextual sentence formation can contribute towards more meaningful sign language communication.

---

**Project:** Kinesis-ISL  
**Domain:** Artificial Intelligence and Machine Learning  
**Department:** Artificial Intelligence and Data Science (AIDS)