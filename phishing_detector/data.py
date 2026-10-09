"""Real phishing + legitimate email dataset loader.

Downloads the ``zefang-liu/phishing-email-dataset`` from Hugging Face
(18,650 real emails — 7,328 phishing, 11,322 safe — originally from the
Kaggle "Phishing Email Detection" dataset) and processes each raw email
into a structured record with ``subject``, ``body``, and ``label``.

The raw emails in this dataset are stored as a single text blob per row.
We split on the first newline to heuristically separate subject from body;
if no clear subject line exists the entire text is used as the body.
"""

import logging
import re
from datasets import load_dataset

logger = logging.getLogger(__name__)

_LABEL_MAP = {
    "Phishing Email": 1,
    "Safe Email": 0,
}

# Minimal cleaning: collapse excessive whitespace, strip control chars
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_WS_RE = re.compile(r"[ \t]+")


def _clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = _CTRL_RE.sub("", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def _split_subject_body(raw_text: str) -> tuple[str, str]:
    """Heuristically split a raw email text into (subject, body).

    Many rows in this dataset are a single concatenated blob. We look
    for the first line break; if the first line is short (<=120 chars)
    we treat it as the subject, otherwise the whole thing is the body
    with an empty subject.
    """
    text = _clean_text(raw_text)
    if not text:
        return "", ""

    lines = text.split("\n", 1)
    if len(lines) == 2 and len(lines[0]) <= 120:
        return lines[0].strip(), lines[1].strip()
    return "", text


def load_real_dataset(max_samples: int | None = None, seed: int = 42) -> list[dict]:
    """Load and process the real phishing email dataset.

    Parameters
    ----------
    max_samples : int or None
        If set, downsample to at most this many records (balanced).
    seed : int
        Random seed for reproducible downsampling.

    Returns
    -------
    list of dicts with keys: subject, body, label (1=phishing, 0=safe)
    """
    logger.info("Downloading zefang-liu/phishing-email-dataset from Hugging Face ...")
    ds = load_dataset("zefang-liu/phishing-email-dataset", split="train")

    records: list[dict] = []
    skipped = 0
    for row in ds:
        raw = row.get("Email Text", "")
        email_type = row.get("Email Type", "")
        label = _LABEL_MAP.get(email_type)
        if label is None:
            skipped += 1
            continue
        subject, body = _split_subject_body(raw)
        if not subject and not body:
            skipped += 1
            continue
        records.append({"subject": subject, "body": body, "label": label})

    logger.info("Processed %d records (%d skipped).", len(records), skipped)

    if max_samples is not None and len(records) > max_samples:
        import random as _rng
        rng = _rng.Random(seed)
        phishing = [r for r in records if r["label"] == 1]
        safe = [r for r in records if r["label"] == 0]
        per_class = max_samples // 2
        rng.shuffle(phishing)
        rng.shuffle(safe)
        records = phishing[:per_class] + safe[:per_class]
        rng.shuffle(records)
        logger.info("Downsampled to %d records (balanced).", len(records))

    return records


# Keep backward compatibility with old code that imported generate_dataset
def generate_dataset(seed: int = 42) -> list[dict]:
    """Backward-compatible alias that loads the real dataset."""
    return load_real_dataset(max_samples=10000, seed=seed)


if __name__ == "__main__":
    import random
    data = load_real_dataset()
    phishing = sum(1 for d in data if d["label"] == 1)
    safe = len(data) - phishing
    print(f"Total: {len(data)}  Phishing: {phishing}  Safe: {safe}")
    print("\nSample phishing email:")
    for d in data:
        if d["label"] == 1:
            print(f"  Subject: {d['subject'][:80]}")
            print(f"  Body: {d['body'][:200]}")
            break
    print("\nSample safe email:")
    for d in data:
        if d["label"] == 0:
            print(f"  Subject: {d['subject'][:80]}")
            print(f"  Body: {d['body'][:200]}")
            break
