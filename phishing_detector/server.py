"""Flask API server for phishing email detection.

Exposes:
  POST /api/predict  — classify an email
  GET  /api/info     — model metadata
  GET  /api/health   — health check

The model is loaded at startup. If no trained model exists on disk,
training runs automatically (using a lightweight synthetic dataset
that doesn't require external downloads) so the server is always
ready to serve predictions.
"""

import logging
import os
import sys
import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "phishing_model.joblib")
PIPELINE_PATH = os.path.join(MODEL_DIR, "feature_pipeline.joblib")
META_PATH = os.path.join(MODEL_DIR, "model_metadata.joblib")

app = Flask(__name__)
CORS(app)

_detector = None
_meta = None


def _try_train():
    """Attempt to train a model if none exists on disk.

    Tries the real dataset first (requires the ``datasets`` library
    and network access). Falls back to a self-contained synthetic
    dataset if those aren't available, so the server always has a
    working model.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    try:
        from phishing_detector.train import train_and_save
        logger.info("No trained model found — starting training pipeline...")
        train_and_save()
        logger.info("Training complete.")
        return True
    except Exception as exc:
        logger.warning("Full training failed (%s). Falling back to synthetic data.", exc)
        return _train_synthetic()


def _train_synthetic():
    """Train on a small built-in dataset so the server always works."""
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report,
    )
    from phishing_detector.features import FeaturePipeline

    logger.info("Training on synthetic fallback dataset...")

    PHISHING = [
        {"subject": "Urgent: Your account has been suspended", "body": "Verify at http://192.168.0.5/verify or lose access in 24h. Enter password.", "label": 1},
        {"subject": "Verify your PayPal account immediately", "body": "Click http://paypa1-secure.com/login.php and enter credentials. Account limited in 48h.", "label": 1},
        {"subject": "You have won $1,000,000!", "body": "Claim prize at http://184.72.29.130/claim Send bank details. Hurry!", "label": 1},
        {"subject": "Amazon order cancelled - action required", "body": "Update billing at https://amaz0n-update-account.com/billing Enter card number and CVV.", "label": 1},
        {"subject": "Reset your password now - security alert", "body": "Reset at http://10.0.0.1/password-reset?token=abc Compromised account. Provide current password.", "label": 1},
        {"subject": "Netflix subscription expired - renew now", "body": "Update payment at http://netfl1x-renew.com/payment Enter card details within 24h.", "label": 1},
        {"subject": "FedEx delivery failed - reschedule", "body": "Click http://fedex-tracking-portal.com/reschedule Pay $2.50 with credit card.", "label": 1},
        {"subject": "Security warning: unauthorized bank access", "body": "Verify at http://172.16.254.1/bank-verify/login Enter username, password, account number.", "label": 1},
        {"subject": "IRS tax refund - claim now", "body": "Claim at http://irs-refund-claim.us/taxrefund Provide SSN and bank account.", "label": 1},
        {"subject": "Apple ID locked - unlock now", "body": "Visit https://app1e-id-unlock.com/verify Enter Apple ID and password.", "label": 1},
        {"subject": "Bitcoin investment - double your money", "body": "Sign up at http://crypto-double-profit.com/invest Guaranteed 200% returns. Send Bitcoin.", "label": 1},
        {"subject": "Update email account to avoid deactivation", "body": "Confirm at http://webmail-account-verify.net/confirm Login with password within 48h.", "label": 1},
        {"subject": "URGENT: Confirm shipment address", "body": "Package undeliverable. Confirm at http://update-shipping-info.com/confirm Pay $1.99.", "label": 1},
        {"subject": "Google account security alert - verify now", "body": "Sign-in from Russia detected. Secure at http://google-security-alert.xyz/verify Enter password.", "label": 1},
        {"subject": "Free $500 Amazon gift card - claim now", "body": "Complete survey at http://free-giftcard-claim.win/amazon Provide email and address. 10 left!", "label": 1},
    ]

    SAFE = [
        {"subject": "Weekly project update - October 7", "body": "Hi team, completed API integration tests and fixed 3 bugs. Next week focus on dashboard UI. Best, Sarah", "label": 0},
        {"subject": "Meeting reminder: Quarterly review tomorrow", "body": "Reminder about quarterly review at 2 PM in Conference Room B. Bring department reports.", "label": 0},
        {"subject": "New product launch announcement", "body": "New product line available at https://www.ourcompany.com 15% off with code WELCOME15. Marketing Team", "label": 0},
        {"subject": "Your order has shipped - tracking included", "body": "Order #ORD-5567 shipped. Track at https://www.ups.com/track?tracknum=1Z999AA1 Expected Oct 9-11.", "label": 0},
        {"subject": "Invitation: Sarah's birthday party Saturday", "body": "Hosting a birthday party Saturday at 7 PM. Food, drinks, music. RSVP by Thursday. 123 Maple St.", "label": 0},
        {"subject": "Newsletter: October tech tips", "body": "Speed up your computer, best password managers, new OS features. Read at https://blog.techtips.com/october", "label": 0},
        {"subject": "Gym membership renewal coming up", "body": "Annual membership at FitnessPlus due Nov 1. Premium plan $499/year. Visit https://www.fitnessplus.com", "label": 0},
        {"subject": "Re: Question about the marketing proposal", "body": "Reviewed the proposal. Social media strategy is great. Schedule a call this week? Regards, David", "label": 0},
        {"subject": "System maintenance scheduled this weekend", "body": "Maintenance Saturday Oct 12, 2-6 AM EST. Some services temporarily unavailable. IT Operations", "label": 0},
        {"subject": "Thank you for your feedback", "body": "Thank you for sharing feedback about your experience. We're glad you enjoyed our service! CX Team", "label": 0},
        {"subject": "Reservation confirmed - Hilton Hotel", "body": "Check-in Oct 15, 3 PM. Check-out Oct 18, 11 AM. King Deluxe. Confirmation HIL-88234. hilton.com", "label": 0},
        {"subject": "Webinar: Introduction to cloud computing", "body": "Free webinar Oct 20, 1 PM EST. Register at https://www.education-platform.com/webinar/cloud AWS Azure GCP.", "label": 0},
        {"subject": "Payroll reminder: submit timesheets by Friday", "body": "Submit timesheets by Friday 5 PM. Log hours at https://portal.ourcompany.com/timesheets Payroll Dept", "label": 0},
        {"subject": "Welcome to the team - onboarding schedule", "body": "Welcome Emily! First day Monday Oct 14. Onboarding schedule attached. Meet manager at 10 AM. HR", "label": 0},
        {"subject": "Library books due in 3 days", "body": "Design Patterns and Clean Code due in 3 days. Renew at https://www.citylibrary.org/myaccount Central Library", "label": 0},
    ]

    data = PHISHING + SAFE
    train_recs, test_recs = train_test_split(data, test_size=0.2, random_state=42, stratify=[d["label"] for d in data])

    pipeline = FeaturePipeline(max_tfidf_features=2000, ngram_range=(1, 2))
    X_train = pipeline.fit_transform(train_recs)
    y_train = np.array([r["label"] for r in train_recs])
    X_test = pipeline.transform(test_recs)
    y_test = np.array([r["label"] for r in test_recs])

    model = LogisticRegression(max_iter=2000, C=10.0, solver="liblinear", random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=["Safe", "Phishing"])

    joblib.dump(model, MODEL_PATH)
    joblib.dump(pipeline, PIPELINE_PATH)
    joblib.dump({
        "model_name": "logistic_regression (synthetic fallback)",
        "best_params": {"C": 10.0, "solver": "liblinear"},
        "test_accuracy": acc, "test_precision": prec,
        "test_recall": rec, "test_f1": f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": cr,
    }, META_PATH)

    logger.info("Synthetic training complete (accuracy=%.4f).", acc)
    return True


def _load():
    """Load the model and metadata, training if necessary."""
    global _detector, _meta

    if _detector is not None:
        return

    if not (os.path.exists(MODEL_PATH) and os.path.exists(PIPELINE_PATH)):
        _try_train()

    from phishing_detector.infer import PhishingDetector
    _detector = PhishingDetector(model_path=MODEL_PATH, pipeline_path=PIPELINE_PATH)

    if os.path.exists(META_PATH):
        _meta = joblib.load(META_PATH)

    logger.info("Model loaded and ready.")


# ---- Pre-load at import time so the server is ready immediately ----
try:
    _load()
except Exception as exc:
    logger.error("Failed to load model at startup: %s", exc)


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        _load()
        data = request.get_json(force=True)
        subject = data.get("subject", "")
        body = data.get("body", "")
        if not subject.strip() and not body.strip():
            return jsonify({"error": "Subject and body are both empty."}), 400
        result = _detector.predict(subject, body)
        return jsonify(result)
    except Exception as exc:
        logger.exception("Prediction error")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/info", methods=["GET"])
def info():
    try:
        _load()
        if _meta:
            return jsonify({
                "model_name": _meta.get("model_name", "unknown"),
                "test_accuracy": round(_meta.get("test_accuracy", 0), 4),
                "test_precision": round(_meta.get("test_precision", 0), 4),
                "test_recall": round(_meta.get("test_recall", 0), 4),
                "test_f1": round(_meta.get("test_f1", 0), 4),
                "best_params": _meta.get("best_params", {}),
                "classification_report": _meta.get("classification_report", ""),
                "confusion_matrix": _meta.get("confusion_matrix", []),
            })
        return jsonify({"error": "Model metadata not available."}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": _detector is not None})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
