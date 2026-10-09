"""Gradio web UI for the phishing email detector.

Launches a local web app where you can paste an email subject and body
and get a real-time Phishing / Safe classification with confidence.

Usage:
    python -m phishing_detector.app
Then open the printed URL in your browser.
"""

import logging
import sys

import gradio as gr

from .infer import PhishingDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_app():
    """Create and launch the Gradio interface."""
    try:
        detector = PhishingDetector()
    except FileNotFoundError as exc:
        logger.error("Could not load model: %s", exc)
        print(f"\nError: {exc}")
        print("Train the model first:  python -m phishing_detector.train")
        sys.exit(1)

    def classify(subject: str, body: str) -> tuple[str, str, str]:
        """Gradio callback returning formatted output for the UI."""
        try:
            result = detector.predict(subject, body)
        except (ValueError, TypeError) as exc:
            return ("Error", str(exc), "")

        label = result["label"]
        confidence = result["confidence"]

        if label == "Phishing":
            verdict = f"⚠️  PHISHING DETECTED  (confidence: {confidence:.1%})"
            verdict_color = "#e74c3c"
        else:
            verdict = f"✅  SAFE  (confidence: {confidence:.1%})"
            verdict_color = "#27ae60"

        detail_html = (
            f"<div style='font-size:16px; padding:12px; border-radius:8px; "
            f"background:{verdict_color}; color:white; text-align:center; "
            f"font-weight:bold; margin-bottom:12px;'>"
            f"{verdict}"
            f"</div>"
            f"<div style='text-align:center; font-size:14px;'>"
            f"Phishing probability: <b>{result['phishing_prob']:.1%}</b><br>"
            f"Safe probability: <b>{result['safe_prob']:.1%}</b>"
            f"</div>"
        )
        return (label, detail_html, "")

    with gr.Blocks(
        title="Phishing Email Detector",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# 🛡️  Phishing Email Detector\n"
            "Paste an email's subject and body below to check whether it's "
            "phishing or safe. The model uses URL analysis, keyword detection, "
            "and text features to make its decision."
        )

        with gr.Row():
            subject_input = gr.Textbox(
                label="Email Subject",
                placeholder="e.g. Urgent: Your account has been suspended",
                lines=1,
            )
        with gr.Row():
            body_input = gr.Textbox(
                label="Email Body",
                placeholder="Paste the full email body here...",
                lines=8,
            )

        classify_btn = gr.Button("Classify Email", variant="primary")
        label_output = gr.Label(label="Classification", num_top_classes=2)
        detail_output = gr.HTML(label="Details")
        clear_btn = gr.Button("Clear")

        classify_btn.click(
            fn=classify,
            inputs=[subject_input, body_input],
            outputs=[label_output, detail_output, gr.Textbox(visible=False)],
        )
        clear_btn.click(
            fn=lambda: ("", "", ""),
            inputs=[],
            outputs=[subject_input, body_input, detail_output],
        )

        gr.Examples(
            examples=[
                [
                    "Urgent: Your account has been suspended",
                    "Dear Customer, we detected suspicious activity. "
                    "Verify your identity at http://192.168.0.5/verify or "
                    "your account will be closed within 24 hours. "
                    "Enter your username and password now.",
                ],
                [
                    "Meeting reminder: Quarterly review tomorrow at 2 PM",
                    "Hello everyone, this is a reminder about our quarterly "
                    "review meeting tomorrow at 2 PM in Conference Room B. "
                    "Please bring your department reports. See you there.",
                ],
                [
                    "Verify your PayPal account immediately",
                    "Dear PayPal user, unusual activity detected. "
                    "Click here: http://paypa1-secure.com/login.php and enter "
                    "your email and password. Account will be limited in 48 hours.",
                ],
                [
                    "Your order has shipped - tracking number included",
                    "Hi John, your order #ORD-5567 has shipped. Track at "
                    "https://www.ups.com/track?tracknum=1Z999AA10123456784. "
                    "Expected delivery: October 9-11. Thanks for shopping!",
                ],
            ],
            inputs=[subject_input, body_input],
        )

    return app


def main():
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
