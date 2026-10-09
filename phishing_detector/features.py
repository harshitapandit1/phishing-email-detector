"""Feature extraction for phishing email detection.

Two complementary feature extractors are provided:

1. ``StructuralFeatureExtractor`` – hand-crafted features that capture
   phishing signals such as IP-based URLs, excessive special characters,
   urgency keywords, credential requests, etc.

2. ``TfidfFeatureExtractor`` – TF-IDF over subject + body text, giving
   the model vocabulary-level discriminative power.

Both are combined into a single ``FeaturePipeline`` that produces a
scipy sparse matrix suitable for training any scikit-learn classifier.
"""

import re
import logging
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

URL_PATTERN = re.compile(
    r'https?://[^\s<>"\']+', re.IGNORECASE
)
IP_URL_PATTERN = re.compile(
    r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', re.IGNORECASE
)
EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

# Keywords frequently used in phishing emails (lowercase for matching)
URGENCY_KEYWORDS = [
    "urgent", "immediately", "asap", "warning", "alert", "suspended",
    "locked", "expired", "deactivation", "verify", "confirm", "action required",
    "account closure", "limited time", "act now", "expire", "hurry",
    "last chance", "final notice", "deadline", "within 24 hours",
    "within 48 hours", "failure", "permanently", "compromised", "security",
]

CREDENTIAL_KEYWORDS = [
    "password", "username", "login", "log in", "sign in", "credentials",
    "social security", "ssn", "bank account", "credit card", "cvv",
    "card number", "billing", "account number", "pin", "verify your identity",
    "confirm your identity", "enter your", "provide your",
]

FINANCIAL_KEYWORDS = [
    "refund", "wire", "transfer", "payment", "invoice", "overdue",
    "balance", "charged", "lottery", "winner", "prize", "claim",
    "gift card", "free", "guaranteed", "investment", "bitcoin",
    "crypto", "double your", "loan", "pre-approved",
]

SUSPICIOUS_TLD = [".xyz", ".click", ".win", ".top", ".loan", ".work",
                  ".gq", ".cf", ".tk", ".ml", ".club"]


class StructuralFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract 15 hand-crafted numeric features from each email.

    Features
    --------
    0  – number of URLs in body
    1  – has IP-based URL (0/1)
    2  – number of URLs with suspicious TLDs
    3  – max URL length
    4  – number of @ symbols in body
    5  – number of urgency keywords
    6  – number of credential keywords
    7  – number of financial keywords
    8  – subject length (chars)
    9  – body length (chars)
    10 – number of exclamation marks in subject
    11 – number of exclamation marks in body
    12 – ratio of uppercase chars in subject
    13 – has HTML-like tags (0/1)
    14 – number of digits in body
    """

    def __init__(self):
        pass

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _count_keywords(text: str, keywords: list[str]) -> int:
        text_lower = text.lower()
        return sum(1 for kw in keywords if kw in text_lower)

    @staticmethod
    def _has_suspicious_tld(url: str) -> bool:
        url_lower = url.lower()
        return any(tld in url_lower for tld in SUSPICIOUS_TLD)

    # -- scikit-learn API --------------------------------------------------

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> csr_matrix:
        features: list[list[float]] = []
        for row in X:
            subject = row.get("subject", "") if isinstance(row, dict) else row[0]
            body = row.get("body", "") if isinstance(row, dict) else row[1]
            combined = f"{subject} {body}"

            urls = URL_PATTERN.findall(combined)
            ip_urls = [u for u in urls if IP_URL_PATTERN.match(u)]
            suspicious_tld_urls = [u for u in urls if self._has_suspicious_tld(u)]
            max_url_len = max((len(u) for u in urls), default=0)
            at_count = combined.count("@")

            urgency_count = self._count_keywords(combined, URGENCY_KEYWORDS)
            cred_count = self._count_keywords(combined, CREDENTIAL_KEYWORDS)
            fin_count = self._count_keywords(combined, FINANCIAL_KEYWORDS)

            subject_len = len(subject)
            body_len = len(body)
            subject_excl = subject.count("!")
            body_excl = body.count("!")
            subject_alpha = sum(1 for c in subject if c.isalpha())
            subject_upper = sum(1 for c in subject if c.isupper())
            upper_ratio = subject_upper / subject_alpha if subject_alpha else 0.0
            has_html = 1 if re.search(r"<\s*(a|img|script|form|input|iframe)", combined, re.IGNORECASE) else 0
            digit_count = sum(1 for c in body if c.isdigit())

            features.append([
                len(urls),
                1 if ip_urls else 0,
                len(suspicious_tld_urls),
                max_url_len,
                at_count,
                urgency_count,
                cred_count,
                fin_count,
                subject_len,
                body_len,
                subject_excl,
                body_excl,
                upper_ratio,
                has_html,
                digit_count,
            ])

        arr = np.array(features, dtype=np.float64)
        return csr_matrix(arr)


class TfidfFeatureExtractor(BaseEstimator, TransformerMixin):
    """TF-IDF vectorizer over concatenated subject + body text."""

    def __init__(self, max_features: int = 3000, ngram_range=(1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self._vectorizer: TfidfVectorizer | None = None

    def fit(self, X, y=None):
        texts = self._to_texts(X)
        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words="english",
            sublinear_tf=True,
        )
        self._vectorizer.fit(texts)
        return self

    def transform(self, X) -> csr_matrix:
        if self._vectorizer is None:
            raise RuntimeError("TfidfFeatureExtractor must be fitted before transform.")
        texts = self._to_texts(X)
        return self._vectorizer.transform(texts)

    @staticmethod
    def _to_texts(X) -> list[str]:
        texts = []
        for row in X:
            if isinstance(row, dict):
                texts.append(f"{row.get('subject', '')} {row.get('body', '')}")
            else:
                texts.append(f"{row[0]} {row[1]}")
        return texts

    @property
    def vocabulary_(self):
        return self._vectorizer.vocabulary_ if self._vectorizer else None


class FeaturePipeline:
    """Combine structural + TF-IDF features into one sparse matrix."""

    FEATURE_NAMES = [
        "num_urls", "has_ip_url", "num_suspicious_tld", "max_url_len",
        "at_count", "urgency_keywords", "credential_keywords",
        "financial_keywords", "subject_len", "body_len",
        "subject_excl", "body_excl", "upper_ratio", "has_html", "digit_count",
    ]

    def __init__(self, max_tfidf_features: int = 3000, ngram_range=(1, 2)):
        self.structural = StructuralFeatureExtractor()
        self.tfidf = TfidfFeatureExtractor(
            max_features=max_tfidf_features, ngram_range=ngram_range
        )
        self._fitted = False

    def fit(self, X):
        self.structural.fit(X)
        self.tfidf.fit(X)
        self._fitted = True
        logger.info("Feature pipeline fitted. Structural: %d features, TF-IDF: %d features.",
                    len(self.FEATURE_NAMES), len(self.tfidf.vocabulary_ or {}))
        return self

    def transform(self, X) -> csr_matrix:
        structural = self.structural.transform(X)
        tfidf = self.tfidf.transform(X)
        return hstack([structural, tfidf], format="csr")

    def fit_transform(self, X) -> csr_matrix:
        self.fit(X)
        return self.transform(X)
