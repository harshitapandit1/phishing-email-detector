import { useState, useCallback } from "react";
import "./App.css";
import { predict, getModelInfo } from "./inference.js";

const EXAMPLES = [
  {
    label: "Account Suspended",
    type: "phishing",
    subject: "Urgent: Your account has been suspended",
    body: "Dear Customer, We have detected suspicious activity on your account. Your account has been suspended for security reasons. Please verify your identity immediately by clicking the link below: http://192.168.0.5/verify?id=account or you will lose access permanently. Login with your username and password to confirm. Failure to do so within 24 hours will result in account closure.",
  },
  {
    label: "PayPal Spoof",
    type: "phishing",
    subject: "Verify your PayPal account immediately",
    body: "Dear PayPal user, We noticed unusual activity in your PayPal account. To protect your account, we need you to confirm your identity. Please click here: http://paypa1-secure.com/login.php and enter your email and password. If you don't verify within 48 hours, your account will be limited.",
  },
  {
    label: "Meeting Reminder",
    type: "safe",
    subject: "Meeting reminder: Quarterly review tomorrow at 2 PM",
    body: "Hello everyone, This is a reminder about our quarterly review meeting scheduled for tomorrow at 2 PM in Conference Room B. Please bring your department reports and Q3 metrics. The agenda is attached. See you there. Regards, Management Office",
  },
  {
    label: "Order Shipped",
    type: "safe",
    subject: "Your order has shipped - tracking number included",
    body: "Hi John, Great news! Your order #ORD-5567 has shipped and is on its way. You can track your package at https://www.ups.com/track?tracknum=1Z999AA10123456784. Expected delivery: October 9-11. If you have any questions, reply to this email. Thanks for shopping with us!",
  },
];

/* ── Icons ────────────────────────────────────────────────────────── */

const ShieldIcon = ({ size = 24 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
  </svg>
);

const AlertTriangleIcon = ({ size = 24 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
    <line x1="12" y1="9" x2="12" y2="13" />
    <line x1="12" y1="17" x2="12.01" y2="17" />
  </svg>
);

const CheckCircleIcon = ({ size = 24 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
    <polyline points="22 4 12 14.01 9 11.01" />
  </svg>
);

const ScanIcon = ({ size = 20 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 7V5a2 2 0 0 1 2-2h2" />
    <path d="M17 3h2a2 2 0 0 1 2 2v2" />
    <path d="M21 17v2a2 2 0 0 1-2 2h-2" />
    <path d="M7 21H5a2 2 0 0 1-2-2v-2" />
    <line x1="7" y1="12" x2="17" y2="12" />
  </svg>
);

const TrashIcon = ({ size = 18 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="3 6 5 6 21 6" />
    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
  </svg>
);

const Spinner = ({ size = 20 }) => (
  <svg className="spin" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
    <path d="M21 12a9 9 0 1 1-6.219-8.56" strokeLinecap="round" />
  </svg>
);

const InfoIcon = ({ size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10" />
    <line x1="12" y1="16" x2="12" y2="12" />
    <line x1="12" y1="8" x2="12.01" y2="8" />
  </svg>
);

/* ── Confidence Ring Component ────────────────────────────────────── */

function ConfidenceRing({ percent, isPhishing }) {
  const radius = 22;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (percent / 100) * circumference;

  return (
    <div className="confidence-ring">
      <svg width="56" height="56">
        <circle className="confidence-ring-bg" cx="28" cy="28" r={radius} />
        <circle
          className="confidence-ring-fill"
          cx="28"
          cy="28"
          r={radius}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          style={{ transition: "stroke-dashoffset 0.8s var(--ease-out)" }}
        />
      </svg>
      <span className="confidence-ring-text">{percent.toFixed(0)}%</span>
    </div>
  );
}

/* ── Main Component ──────────────────────────────────────────────── */

export default function App() {
  const [subject, setSubject] = useState("");
  const [body, setBody] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  // Model info is available synchronously — no async loading needed.
  const modelInfo = getModelInfo();

  const classify = useCallback(() => {
    if (!subject.trim() && !body.trim()) {
      setError("Please enter at least a subject or body to classify.");
      setResult(null);
      return;
    }
    setError(null);
    setLoading(true);
    setResult(null);

    // Small delay so the loading animation is visible — the actual
    // computation is instant (< 1ms) since it runs in-process.
    setTimeout(() => {
      try {
        const res = predict(subject, body);
        setResult(res);
      } catch (err) {
        setError(err.message || "Something went wrong during classification.");
      } finally {
        setLoading(false);
      }
    }, 400);
  }, [subject, body]);

  const loadExample = (ex) => {
    setSubject(ex.subject);
    setBody(ex.body);
    setResult(null);
    setError(null);
  };

  const clear = () => {
    setSubject("");
    setBody("");
    setResult(null);
    setError(null);
  };

  const isPhishing = result?.label === "Phishing";
  const confidencePct = result ? result.confidence * 100 : 0;
  const phishingPct = result ? result.phishing_prob * 100 : 0;
  const safePct = result ? result.safe_prob * 100 : 0;

  return (
    <div className="app">
      {/* ── Header ──────────────────────────────────────────────── */}
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-icon"><ShieldIcon size={22} /></span>
            <span className="logo-text">Phishing Detector</span>
          </div>
          <div className="header-right">
            {modelInfo && (
              <span className="badge-model">
                {modelInfo.model_name.replace(/_/g, " ")}
              </span>
            )}
          </div>
        </div>
      </header>

      <main className="main">
        {/* ── Hero ──────────────────────────────────────────────── */}
        <section className="hero">
          <div className="hero-badge">
            <span className="hero-badge-dot" />
            ML-Powered Email Security
          </div>
          <h1 className="hero-title">
            Detect phishing emails<br />with <span className="hero-title-accent">machine learning</span>
          </h1>
          <p className="hero-subtitle">
            Trained on 18,000+ real emails using TF-IDF text analysis, URL feature
            extraction, and structural signal detection. Paste any email to get an
            instant classification with confidence scores.
          </p>

          {/* Stats */}
          {modelInfo && (
            <div className="stats">
              <Stat label="Accuracy" value={`${(modelInfo.test_accuracy * 100).toFixed(1)}%`} />
              <Stat label="Precision" value={`${(modelInfo.test_precision * 100).toFixed(1)}%`} />
              <Stat label="Recall" value={`${(modelInfo.test_recall * 100).toFixed(1)}%`} />
              <Stat label="F1 Score" value={`${(modelInfo.test_f1 * 100).toFixed(1)}%`} />
            </div>
          )}
        </section>

        {/* ── Detector ──────────────────────────────────────────── */}
        <section className="detector">
          {/* Left: Input */}
          <div className="panel panel-input">
            <div className="panel-header">
              <h2 className="panel-title">Email Content</h2>
              <span className="panel-sub">Paste an email to analyze</span>
            </div>

            <div className="field">
              <label className="field-label" htmlFor="subject">
                <span className="field-label-num">1</span>
                Subject Line
              </label>
              <input
                id="subject"
                className="field-input"
                type="text"
                value={subject}
                onChange={(e) => setSubject(e.target.value)}
                placeholder="e.g. Urgent: Your account has been suspended"
              />
            </div>

            <div className="field">
              <label className="field-label" htmlFor="body">
                <span className="field-label-num">2</span>
                Email Body
              </label>
              <textarea
                id="body"
                className="field-textarea"
                value={body}
                onChange={(e) => setBody(e.target.value)}
                placeholder="Paste the full email body here..."
                rows={8}
              />
            </div>

            <div className="actions">
              <button
                className={`btn-classify ${loading ? "is-loading" : ""}`}
                onClick={classify}
                disabled={loading || (!subject.trim() && !body.trim())}
              >
                {loading ? (<><Spinner /> Analyzing...</>) : (<><ScanIcon /> Detect Phishing</>)}
              </button>
              <button
                className="btn-clear"
                onClick={clear}
                disabled={loading || (!subject && !body)}
              >
                <TrashIcon /> Clear
              </button>
            </div>

            {/* Examples */}
            <div className="examples">
              <p className="examples-title">Quick examples</p>
              <div className="examples-row">
                {EXAMPLES.map((ex) => (
                  <button
                    key={ex.label}
                    className={`example-chip ${ex.type === "phishing" ? "chip-danger" : "chip-safe"}`}
                    onClick={() => loadExample(ex)}
                  >
                    <span className="chip-dot" />
                    {ex.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Right: Result */}
          <div className="panel panel-result">
            <div className="panel-header">
              <h2 className="panel-title">Analysis Result</h2>
              <span className="panel-sub">Classification & confidence</span>
            </div>

            <div className="result-body">
              {/* Empty state */}
              {!loading && !result && !error && (
                <div className="empty-state">
                  <div className="empty-icon"><ShieldIcon size={40} /></div>
                  <p className="empty-title">Awaiting analysis</p>
                  <p className="empty-desc">
                    Enter an email subject and body, then click<br />
                    "Detect Phishing" to see the result.
                  </p>
                </div>
              )}

              {/* Loading state */}
              {loading && (
                <div className="loading-state">
                  <div className="loading-ring"><Spinner size={32} /></div>
                  <p className="loading-text">Analyzing email content...</p>
                  <p className="loading-sub">Extracting features and running classification</p>
                </div>
              )}

              {/* Error state */}
              {error && !loading && (
                <div className="error-state">
                  <div className="error-icon"><AlertTriangleIcon size={28} /></div>
                  <p className="error-text">{error}</p>
                </div>
              )}

              {/* Result */}
              {!loading && result && (
                <div className="result-content">
                  {/* Verdict badge with confidence ring */}
                  <div
                    className={`verdict ${isPhishing ? "verdict-danger" : "verdict-safe"}`}
                    style={{ animation: "scaleIn 0.4s var(--ease-out)" }}
                  >
                    <div className="verdict-icon-wrap">
                      {isPhishing ? <AlertTriangleIcon size={28} /> : <CheckCircleIcon size={28} />}
                    </div>
                    <div className="verdict-info">
                      <span className="verdict-label">
                        {isPhishing ? "Phishing Detected" : "Safe Email"}
                      </span>
                      <span className="verdict-conf">
                        Confidence: <span className="verdict-conf-pct">{confidencePct.toFixed(1)}%</span>
                      </span>
                    </div>
                    <ConfidenceRing percent={confidencePct} isPhishing={isPhishing} />
                  </div>

                  {/* Probability breakdown */}
                  <div className="prob-block">
                    <p className="prob-block-title">Probability Breakdown</p>

                    <div className="prob-item">
                      <div className="prob-item-head">
                        <span className="prob-item-name">
                          <span className="prob-dot prob-dot-danger" />
                          Phishing
                        </span>
                        <span className="prob-item-val">{phishingPct.toFixed(1)}%</span>
                      </div>
                      <div className="prob-bar">
                        <div
                          className="prob-bar-fill prob-bar-danger"
                          style={{ width: `${phishingPct}%`, animation: "barGrow 0.8s var(--ease-out)" }}
                        />
                      </div>
                    </div>

                    <div className="prob-item">
                      <div className="prob-item-head">
                        <span className="prob-item-name">
                          <span className="prob-dot prob-dot-safe" />
                          Safe
                        </span>
                        <span className="prob-item-val">{safePct.toFixed(1)}%</span>
                      </div>
                      <div className="prob-bar">
                        <div
                          className="prob-bar-fill prob-bar-safe"
                          style={{ width: `${safePct}%`, animation: "barGrow 0.8s var(--ease-out)" }}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Detected signals */}
                  <div className={`interpretation ${isPhishing ? "interp-danger" : "interp-safe"}`}>
                    <div className="interp-icon">
                      {isPhishing ? <AlertTriangleIcon size={18} /> : <InfoIcon size={18} />}
                    </div>
                    <div className="interp-content">
                      <p className="interp-summary">
                        {isPhishing
                          ? "This email shows phishing indicators. Do not click any links or share personal information."
                          : "This email appears legitimate. No strong phishing signals were detected."}
                      </p>
                      {result.signals && result.signals.length > 0 && (
                        <ul className="signal-list">
                          {result.signals.map((sig) => (
                            <li key={sig.type} className={`signal-item signal-${sig.weight}`}>
                              {sig.weight === "positive"
                                ? <CheckCircleIcon size={14} />
                                : <AlertTriangleIcon size={14} />}
                              <span>{sig.text}</span>
                            </li>
                          ))}
                        </ul>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </section>

        {/* ── Confusion Matrix ───────────────────────────────────── */}
        {modelInfo?.confusion_matrix && (
          <section className="panel panel-metrics">
            <div className="panel-header">
              <h2 className="panel-title">Confusion Matrix</h2>
              <span className="panel-sub">Test set performance (2,000 emails)</span>
            </div>
            <div className="cm-wrap">
              <div className="cm-grid">
                <div className="cm-cell cm-corner" />
                <div className="cm-cell cm-col-hd">Pred. Safe</div>
                <div className="cm-cell cm-col-hd">Pred. Phishing</div>

                <div className="cm-cell cm-row-hd">Actual Safe</div>
                <div className="cm-cell cm-tn">{modelInfo.confusion_matrix[0][0]}</div>
                <div className="cm-cell cm-fp">{modelInfo.confusion_matrix[0][1]}</div>

                <div className="cm-cell cm-row-hd">Actual Phishing</div>
                <div className="cm-cell cm-fn">{modelInfo.confusion_matrix[1][0]}</div>
                <div className="cm-cell cm-tp">{modelInfo.confusion_matrix[1][1]}</div>
              </div>
              <div className="cm-legend">
                <span className="cm-leg-item"><span className="cm-leg-dot cm-leg-correct" /> Correct predictions</span>
                <span className="cm-leg-item"><span className="cm-leg-dot cm-leg-warn" /> False positives</span>
                <span className="cm-leg-item"><span className="cm-leg-dot cm-leg-error" /> False negatives</span>
              </div>
            </div>
          </section>
        )}
      </main>

      <footer className="footer">
        <p>
          Phishing Email Detector &middot; Trained on 18,000+ real emails with
          Scikit-learn &middot; {modelInfo ? modelInfo.model_name.replace(/_/g, " ") : "Logistic Regression"}
        </p>
      </footer>
    </div>
  );
}

/* ── Small components ──────────────────────────────────────────────── */

function Stat({ label, value }) {
  return (
    <div className="stat">
      <span className="stat-value">{value}</span>
      <span className="stat-label">{label}</span>
    </div>
  );
}
