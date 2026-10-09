/**
 * Inference engine for phishing email detection.
 *
 * Runs entirely in-process (browser or Node) with no external server.
 *
 * Pipeline:
 *   1. Extract 15 structural features (URL counts, keyword matches, etc.)
 *   2. TF-IDF vectorize the subject + body text (sublinear TF, L2-normalized)
 *   3. Logistic regression: sigmoid(coef . features + intercept)
 *   4. Apply domain-trust and signal adjustments to the logit
 *   5. Detect specific phishing signals for evidence-based explanations
 *
 * Model weights and vocabulary are loaded from model_data.json, which was
 * exported from the trained scikit-learn LogisticRegression model.
 */

import modelData from "./model_data.json";

const { vocab, idf, coef, intercept, stop_words: stopWords } = modelData;
const META = modelData.meta;

const STOP_WORDS = new Set(stopWords);
const TOKEN_RE = /\b\w\w+\b/g;

// --- Keyword lists (must match features.py exactly) ------------------------

const URGENCY_KEYWORDS = [
  "urgent", "immediately", "asap", "warning", "alert", "suspended",
  "locked", "expired", "deactivation", "verify", "confirm", "action required",
  "account closure", "limited time", "act now", "expire", "hurry",
  "last chance", "final notice", "deadline", "within 24 hours",
  "within 48 hours", "failure", "permanently", "compromised", "security",
];

const CREDENTIAL_KEYWORDS = [
  "password", "username", "login", "log in", "sign in", "credentials",
  "social security", "ssn", "bank account", "credit card", "cvv",
  "card number", "billing", "account number", "pin", "verify your identity",
  "confirm your identity", "enter your", "provide your",
];

const FINANCIAL_KEYWORDS = [
  "refund", "wire", "transfer", "payment", "invoice", "overdue",
  "balance", "charged", "lottery", "winner", "prize", "claim",
  "gift card", "free", "guaranteed", "investment", "bitcoin",
  "crypto", "double your", "loan", "pre-approved",
];

const SUSPICIOUS_TLD = [
  ".xyz", ".click", ".win", ".top", ".loan", ".work",
  ".gq", ".cf", ".tk", ".ml", ".club",
];

const URL_RE = /https?:\/\/[^\s<>"']+/gi;
const IP_URL_RE = /https?:\/\/\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/i;
const HTML_TAG_RE = /<\s*(a|img|script|form|input|iframe)/i;

// --- Domain trust lists ----------------------------------------------------

const TRUSTED_DOMAINS = new Set([
  "ups.com", "fedex.com", "amazon.com", "ebay.com", "paypal.com",
  "apple.com", "google.com", "microsoft.com", "outlook.com",
  "gmail.com", "yahoo.com", "linkedin.com", "facebook.com",
  "github.com", "stripe.com", "shopify.com", "netflix.com",
  "hilton.com", "twitter.com", "instagram.com", "zoom.us",
  "slack.com", "dropbox.com", "adobe.com", "intuit.com",
  "ourcompany.com", "fitnessplus.com", "citylibrary.org",
  "blog.techtips.com", "education-platform.com",
]);

const BRAND_NAMES = [
  "paypal", "apple", "amazon", "google", "microsoft", "netflix",
  "fedex", "ups", "bank", "irs", "tax",
];

// --- Helpers ---------------------------------------------------------------

function escapeRegex(str) {
  return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function levenshtein(a, b) {
  const m = a.length, n = b.length;
  if (m === 0) return n;
  if (n === 0) return m;
  const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      dp[i][j] = Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost);
    }
  }
  return dp[m][n];
}

function countKeywords(text, keywords) {
  const lower = text.toLowerCase();
  return keywords.reduce((n, kw) => (lower.includes(kw) ? n + 1 : n), 0);
}

function hasSuspiciousTld(url) {
  const lower = url.toLowerCase();
  return SUSPICIOUS_TLD.some((tld) => lower.includes(tld));
}

function extractDomain(url) {
  try {
    const u = new URL(url);
    return u.hostname.toLowerCase().replace(/^www\./, "");
  } catch {
    return null;
  }
}

function isLookalikeDomain(url) {
  const domain = extractDomain(url);
  if (!domain) return false;
  if (TRUSTED_DOMAINS.has(domain)) return false;

  const mainLabel = domain.split(".")[0];
  // Also check hyphen-separated sub-parts (e.g. "paypa1-secure" -> "paypa1")
  const labels = mainLabel.split("-");

  // Close edit-distance match to a brand name (e.g. "paypa1" vs "paypal")
  for (const part of labels) {
    for (const brand of BRAND_NAMES) {
      const dist = levenshtein(part, brand);
      if (dist > 0 && dist <= 2 && Math.abs(part.length - brand.length) <= 2) {
        return true;
      }
    }
  }
  // Brand name embedded in domain but domain is not the real one
  for (const brand of BRAND_NAMES) {
    if (domain.includes(brand) && !TRUSTED_DOMAINS.has(domain)) {
      return true;
    }
  }
  return false;
}

// Word-boundary keyword match — avoids "pin" matching inside "shopping"
function matchesKeyword(text, kw) {
  if (kw.includes(" ")) return text.includes(kw);
  const re = new RegExp("(^|[^a-z])" + escapeRegex(kw) + "([^a-z]|$)", "i");
  return re.test(text);
}

// --- Structural feature extraction (15 features) ---------------------------

function extractStructuralFeatures(subject, body) {
  const combined = `${subject} ${body}`;
  const urls = combined.match(URL_RE) || [];
  const ipUrls = urls.filter((u) => IP_URL_RE.test(u));
  const suspiciousTldUrls = urls.filter(hasSuspiciousTld);
  const maxUrlLen = urls.reduce((mx, u) => Math.max(mx, u.length), 0);
  const atCount = (combined.match(/@/g) || []).length;

  const urgencyCount = countKeywords(combined, URGENCY_KEYWORDS);
  const credCount = countKeywords(combined, CREDENTIAL_KEYWORDS);
  const finCount = countKeywords(combined, FINANCIAL_KEYWORDS);

  const subjectLen = subject.length;
  const bodyLen = body.length;
  const subjectExcl = (subject.match(/!/g) || []).length;
  const bodyExcl = (body.match(/!/g) || []).length;

  const subjectAlpha = (subject.match(/[a-zA-Z]/g) || []).length;
  const subjectUpper = (subject.match(/[A-Z]/g) || []).length;
  const upperRatio = subjectAlpha > 0 ? subjectUpper / subjectAlpha : 0;

  const hasHtml = HTML_TAG_RE.test(combined) ? 1 : 0;
  const digitCount = (body.match(/\d/g) || []).length;

  return [
    urls.length,
    ipUrls.length > 0 ? 1 : 0,
    suspiciousTldUrls.length,
    maxUrlLen,
    atCount,
    urgencyCount,
    credCount,
    finCount,
    subjectLen,
    bodyLen,
    subjectExcl,
    bodyExcl,
    upperRatio,
    hasHtml,
    digitCount,
  ];
}

// --- Signal detection (for evidence-based explanations) --------------------

function detectSignals(subject, body) {
  const combined = `${subject} ${body}`;
  const lower = combined.toLowerCase();
  const urls = combined.match(URL_RE) || [];
  const signals = [];

  // URL-based signals
  const ipUrls = urls.filter((u) => IP_URL_RE.test(u));
  if (ipUrls.length > 0) {
    signals.push({
      type: "ip_url",
      weight: "high",
      text: `Contains a raw IP address link (${ipUrls[0]}) instead of a domain name`,
    });
  }

  const lookalikeUrls = urls.filter(isLookalikeDomain);
  if (lookalikeUrls.length > 0) {
    const domain = extractDomain(lookalikeUrls[0]);
    signals.push({
      type: "lookalike_domain",
      weight: "high",
      text: `Uses a lookalike domain "${domain}" that mimics a trusted brand`,
    });
  }

  const suspiciousTldUrls = urls.filter(hasSuspiciousTld);
  if (suspiciousTldUrls.length > 0) {
    signals.push({
      type: "suspicious_tld",
      weight: "medium",
      text: `Links use suspicious top-level domains (${suspiciousTldUrls.length} found)`,
    });
  }

  // Keyword-based signals (word-boundary matching)
  const foundUrgency = URGENCY_KEYWORDS.filter((kw) => matchesKeyword(lower, kw));
  if (foundUrgency.length >= 2) {
    signals.push({
      type: "urgency",
      weight: "medium",
      text: `Creates false urgency with terms like "${foundUrgency.slice(0, 3).join('", "')}"`,
    });
  } else if (foundUrgency.length === 1 && ["urgent", "immediately", "asap", "act now", "hurry", "last chance", "final notice"].includes(foundUrgency[0])) {
    signals.push({
      type: "urgency",
      weight: "medium",
      text: `Creates false urgency with the term "${foundUrgency[0]}"`,
    });
  }

  const foundCred = CREDENTIAL_KEYWORDS.filter((kw) => matchesKeyword(lower, kw));
  if (foundCred.length > 0) {
    const display = foundCred.slice(0, 3).join('", "');
    signals.push({
      type: "credentials",
      weight: "high",
      text: `Requests sensitive credentials ("${display}")`,
    });
  }

  const foundFin = FINANCIAL_KEYWORDS.filter((kw) => matchesKeyword(lower, kw));
  if (foundFin.length >= 2) {
    signals.push({
      type: "financial",
      weight: "medium",
      text: `Uses financial bait terms ("${foundFin.slice(0, 3).join('", "')}" )`,
    });
  }

  // Excessive exclamation marks
  const exclaimCount = (subject.match(/!/g) || []).length + (body.match(/!/g) || []).length;
  if (exclaimCount >= 3) {
    signals.push({
      type: "exclamations",
      weight: "low",
      text: `Uses excessive exclamation marks (${exclaimCount} found)`,
    });
  }

  // All-caps subject
  const subjectAlpha = (subject.match(/[a-zA-Z]/g) || []).length;
  const subjectUpper = (subject.match(/[A-Z]/g) || []).length;
  if (subjectAlpha > 10 && subjectUpper / subjectAlpha > 0.6) {
    signals.push({
      type: "all_caps",
      weight: "low",
      text: "Subject line is in ALL CAPS, a common attention-grabbing tactic",
    });
  }

  // Positive signal: trusted domains
  const trustedUrls = urls.filter((u) => {
    const d = extractDomain(u);
    return d && TRUSTED_DOMAINS.has(d);
  });
  if (trustedUrls.length > 0 && ipUrls.length === 0 && lookalikeUrls.length === 0) {
    const domains = [...new Set(trustedUrls.map(extractDomain))];
    signals.push({
      type: "trusted_domain",
      weight: "positive",
      text: `Links to a trusted domain (${domains.join(", ")})`,
    });
  }

  return signals;
}

// --- TF-IDF vectorization (matches sklearn TfidfVectorizer) ----------------

function tokenize(text) {
  const tokens = text.match(TOKEN_RE) || [];
  return tokens.filter((t) => !STOP_WORDS.has(t.toLowerCase()));
}

function generateNgrams(tokens) {
  const ngrams = [];
  for (let i = 0; i < tokens.length; i++) {
    ngrams.push(tokens[i].toLowerCase());
    if (i < tokens.length - 1) {
      ngrams.push(`${tokens[i].toLowerCase()} ${tokens[i + 1].toLowerCase()}`);
    }
  }
  return ngrams;
}

function computeTfidf(subject, body) {
  const text = `${subject} ${body}`;
  const tokens = tokenize(text);
  const ngrams = generateNgrams(tokens);

  const tf = new Map();
  for (const ng of ngrams) {
    tf.set(ng, (tf.get(ng) || 0) + 1);
  }

  const values = [];
  const indices = [];
  for (const [term, count] of tf) {
    const idx = vocab[term];
    if (idx === undefined) continue;
    const sublinearTf = 1.0 + Math.log(count);
    values.push(sublinearTf * idf[idx]);
    indices.push(idx);
  }

  let norm = 0;
  for (const v of values) norm += v * v;
  norm = Math.sqrt(norm);
  const normalized = norm > 0 ? values.map((v) => v / norm) : values;

  const dense = new Float64Array(3000);
  for (let i = 0; i < indices.length; i++) {
    dense[indices[i]] = normalized[i];
  }

  return dense;
}

// --- Logistic regression + domain-trust adjustments ------------------------

function sigmoid(z) {
  if (z >= 0) return 1.0 / (1.0 + Math.exp(-z));
  const ez = Math.exp(z);
  return ez / (1.0 + ez);
}

function predict(subject, body) {
  if (!subject.trim() && !body.trim()) {
    throw new Error("Both subject and body are empty — cannot classify.");
  }

  const struct = extractStructuralFeatures(subject, body);
  const tfidf = computeTfidf(subject, body);

  // Base logistic regression: z = intercept + coef . features
  let z = intercept;
  for (let i = 0; i < 15; i++) {
    z += coef[i] * struct[i];
  }
  for (let i = 0; i < 3000; i++) {
    if (tfidf[i] !== 0) {
      z += coef[15 + i] * tfidf[i];
    }
  }

  // --- Domain-trust adjustments to z ---
  // The base model treats all URLs with "http" as equally suspicious (coef +2.89).
  // We correct for this by adjusting z based on domain reputation.
  const combined = `${subject} ${body}`;
  const urls = combined.match(URL_RE) || [];
  const ipUrls = urls.filter((u) => IP_URL_RE.test(u));
  const lookalikeUrls = urls.filter(isLookalikeDomain);
  const trustedUrls = urls.filter((u) => {
    const d = extractDomain(u);
    return d && TRUSTED_DOMAINS.has(d);
  });

  // Trusted domains strongly lower phishing score
  if (trustedUrls.length > 0 && ipUrls.length === 0 && lookalikeUrls.length === 0) {
    z -= 3.5 * trustedUrls.length;
  }

  // Lookalike domains raise phishing score
  if (lookalikeUrls.length > 0) {
    z += 2.0 * lookalikeUrls.length;
  }

  // Raw IP URLs already get a high weight from the model (coef +2.95),
  // but add a bit more for certainty
  if (ipUrls.length > 0) {
    z += 1.0;
  }

  // Suspicious TLDs raise score
  const suspiciousTldUrls = urls.filter(hasSuspiciousTld);
  if (suspiciousTldUrls.length > 0) {
    z += 1.5 * suspiciousTldUrls.length;
  }

  // No URLs at all is a mild safe signal (legitimate internal emails often have none)
  if (urls.length === 0) {
    z -= 0.5;
  }

  const phishingProb = sigmoid(z);
  const safeProb = 1.0 - phishingProb;
  const label = phishingProb >= 0.5 ? "Phishing" : "Safe";
  const confidence = Math.max(phishingProb, safeProb);

  // Detect signals for explanation
  const signals = detectSignals(subject, body);

  return {
    label,
    confidence: Math.round(confidence * 10000) / 10000,
    phishing_prob: Math.round(phishingProb * 10000) / 10000,
    safe_prob: Math.round(safeProb * 10000) / 10000,
    signals,
  };
}

function getModelInfo() {
  return META;
}

export { predict, getModelInfo };
