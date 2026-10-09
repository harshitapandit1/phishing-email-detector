# Phishing Email Detection System

A production-ready machine learning system that classifies emails as **Phishing** or **Safe** using Scikit-learn, trained on **18,000+ real emails** from a public Hugging Face dataset. It features a Vite + React web UI for real-time testing.

## Features

- **Real dataset**: Trained on 18,650 real emails (7,328 phishing, 11,322 safe) from the [zefang-liu/phishing-email-dataset](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset) (Kaggle Phishing Email Detection corpus).
- **Feature extraction**:
  - **Structural features** (15): URL count, IP-based URLs, suspicious TLDs, max URL length, `@` count, urgency keywords, credential keywords, financial keywords, subject/body length, exclamation marks, uppercase ratio, HTML tags, digit count.
  - **TF-IDF features** (3,000): 1–2 gram TF-IDF over subject + body text with stop-word removal and sublinear scaling.
- **Multi-model training**: Random Forest, Logistic Regression, Gradient Boosting — all tuned with `GridSearchCV` + 3-fold stratified cross-validation.
- **Best model**: Logistic Regression — 97.6% accuracy, 97.6% F1-score on held-out test data.
- **Web UI**: Vite + React frontend with real-time classification, confidence bars, confusion matrix display, and example emails.
- **Model persistence**: best model + feature pipeline saved with `joblib`.
- **Error handling & logging**: throughout all modules.

## Project Structure

```
├── phishing_detector/
│   ├── __init__.py        # Package exports
│   ├── data.py            # Real dataset loader (Hugging Face)
│   ├── features.py        # Structural + TF-IDF feature extractors
│   ├── train.py           # Training pipeline (tune, evaluate, save)
│   ├── infer.py           # Inference module + CLI
│   ├── server.py          # Flask API server (used by the web UI)
│   └── app.py             # Standalone Gradio UI (alternative)
├── src/
│   ├── App.jsx            # React UI for email classification
│   ├── App.css            # Styled UI components
│   ├── main.jsx           # React entry point
│   └── index.css          # Global styles
├── models/                # Saved model artifacts (auto-created on train)
├── python-server-plugin.js # Vite plugin to auto-start Flask server
├── vite.config.js         # Vite config with Python server + API proxy
├── package.json
├── requirements.txt
└── README.md
```

## Installation

### Python dependencies

```bash
pip install -r requirements.txt
```

### Node.js dependencies

```bash
npm install
```

## Quick Start

### 1. Train the model (required once)

```bash
python -m phishing_detector.train
```

This downloads the real dataset from Hugging Face, extracts features, tunes all three models with cross-validation, and saves the best model to `models/`.

### 2. Launch the web UI

```bash
npm run dev
```

This starts both the Vite dev server (port 5173) and the Flask inference API (port 8000) automatically. Open `http://localhost:5173` in your browser to use the detector.

### 3. Test via CLI

```bash
python -m phishing_detector.infer "Urgent: account suspended" "Verify at http://192.168.0.5/verify now"
```

### 4. Use in Python

```python
from phishing_detector.infer import PhishingDetector

detector = PhishingDetector()
result = detector.predict(
    subject="Urgent: Your account has been suspended",
    body="Verify at http://192.168.0.5/verify or lose access in 24 hours.",
)
print(result)
# {'label': 'Phishing', 'confidence': 0.99, 'phishing_prob': 0.99, 'safe_prob': 0.01}
```

## How It Works

The classifier combines two types of features:

1. **Structural features** — numeric signals extracted via regex and keyword matching: number of URLs, presence of IP-address URLs, suspicious TLDs (.xyz, .click, .win, etc.), urgency/credential/financial keyword counts, subject/body length, exclamation marks, uppercase ratio, HTML tags, and digit count.

2. **TF-IDF features** — the text content of subject + body is vectorized using TF-IDF with 1–2 gram ranges and stop-word removal, capturing vocabulary-level phishing signals.

These are concatenated into a single sparse matrix and fed to three classifiers. Each is tuned with `GridSearchCV` (3-fold stratified CV, F1-score), and the best-performing model (Logistic Regression, 97.6% accuracy) is saved for inference.

## Model Performance (on real data)

| Model               | Accuracy | Precision | Recall | F1    |
|---------------------|----------|-----------|--------|-------|
| Logistic Regression | 97.6%    | 96.4%     | 98.8%  | 97.6% |
| Random Forest       | 96.4%    | 94.0%     | 99.0%  | 96.4% |
| Gradient Boosting   | 94.1%    | 90.9%     | 97.9%  | 94.3% |

## Adding Your Own Data

Replace the dataset loader in `data.py` with your own CSV:

```python
import pandas as pd
df = pd.read_csv("your_emails.csv")  # columns: subject, body, label (1=phishing, 0=safe)
```

## License

This project is for educational and defensive security purposes.
