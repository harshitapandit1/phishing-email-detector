"""Inference module for the phishing email detector.

Loads the saved model + feature pipeline and provides:
  - ``predict_email(subject, body)`` -> dict with label, confidence, probabilities
  - ``PhishingDetector`` class wrapping the above for reuse
  - CLI entry point: ``python -m phishing_detector.infer``

The detector can be used programmatically or from the command line.
"""

import logging
import os
import sys
import joblib
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "phishing_model.joblib")
PIPELINE_PATH = os.path.join(MODEL_DIR, "feature_pipeline.joblib")
META_PATH = os.path.join(MODEL_DIR, "model_metadata.joblib")


class PhishingDetector:
    """Load the trained model + pipeline and classify new emails.

    Parameters
    ----------
    model_path : str
        Path to the joblib-saved classifier.
    pipeline_path : str
        Path to the joblib-saved ``FeaturePipeline``.
    """

    def __init__(self, model_path: str = MODEL_PATH, pipeline_path: str = PIPELINE_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}. Run training first: python -m phishing_detector.train"
            )
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(
                f"Pipeline file not found: {pipeline_path}. Run training first."
            )
        self.model = joblib.load(model_path)
        self.pipeline = joblib.load(pipeline_path)
        logger.info("Model and pipeline loaded successfully.")

    def predict(self, subject: str, body: str) -> dict:
        """Classify a single email.

        Returns
        -------
        dict with keys:
            label        – "Phishing" or "Safe"
            confidence   – float (probability of the predicted class, 0–1)
            phishing_prob – probability the email is phishing
            safe_prob     – probability the email is safe
        """
        if not isinstance(subject, str) or not isinstance(body, str):
            raise TypeError("subject and body must be strings.")

        if not subject.strip() and not body.strip():
            raise ValueError("Both subject and body are empty – cannot classify.")

        record = {"subject": subject, "body": body}
        features = self.pipeline.transform([record])
        pred = int(self.model.predict(features)[0])
        probs = self.model.predict_proba(features)[0] if hasattr(self.model, "predict_proba") else None

        if probs is not None:
            phishing_prob = float(probs[1])
            safe_prob = float(probs[0])
            confidence = max(phishing_prob, safe_prob)
        else:
            phishing_prob = float(pred)
            safe_prob = 1.0 - phishing_prob
            confidence = 1.0

        label = "Phishing" if pred == 1 else "Safe"
        return {
            "label": label,
            "confidence": round(confidence, 4),
            "phishing_prob": round(phishing_prob, 4),
            "safe_prob": round(safe_prob, 4),
        }

    def predict_batch(self, emails: list[dict]) -> list[dict]:
        """Classify multiple emails. Each dict must have 'subject' and 'body'."""
        return [self.predict(e["subject"], e["body"]) for e in emails]


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

_detector_cache: PhishingDetector | None = None


def predict_email(subject: str, body: str) -> dict:
    """Module-level convenience wrapper that lazily loads the detector."""
    global _detector_cache
    if _detector_cache is None:
        _detector_cache = PhishingDetector()
    return _detector_cache.predict(subject, body)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    """Interactive command-line interface."""
    print("=" * 60)
    print("  Phishing Email Detector – CLI")
    print("=" * 60)

    try:
        detector = PhishingDetector()
    except FileNotFoundError as exc:
        print(f"\nError: {exc}")
        print("\nYou need to train the model first. Run:")
        print("  python -m phishing_detector.train")
        sys.exit(1)

    if len(sys.argv) > 2:
        # Non-interactive: subject and body passed as arguments
        subject = sys.argv[1]
        body = sys.argv[2]
        result = detector.predict(subject, body)
        print(f"\nSubject : {subject}")
        print(f"Body    : {body[:80]}...")
        print(f"Result  : {result['label']} (confidence: {result['confidence']:.1%})")
        print(f"  Phishing probability: {result['phishing_prob']:.1%}")
        print(f"  Safe probability    : {result['safe_prob']:.1%}")
        return

    print("\nType an email to classify. Press Enter after each field.\n")
    while True:
        subject = input("Subject (or 'quit' to exit): ").strip()
        if subject.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        body = input("Body: ").strip()
        if body.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        try:
            result = detector.predict(subject, body)
            print(f"\n  >>> Result: {result['label']}  (confidence: {result['confidence']:.1%})")
            print(f"  >>> Phishing prob: {result['phishing_prob']:.1%}  |  Safe prob: {result['safe_prob']:.1%}\n")
        except (ValueError, TypeError) as exc:
            print(f"\n  Error: {exc}\n")


if __name__ == "__main__":
    _cli()
