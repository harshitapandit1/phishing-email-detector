"""Training pipeline for the phishing email detector.

Trains three models (Random Forest, Logistic Regression, Gradient Boosting),
performs hyperparameter tuning with GridSearchCV + cross-validation, evaluates
each on the held-out test set, and saves the best model + feature pipeline
to disk with joblib.

Usage:
    python -m phishing_detector.train
"""

import logging
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

from .data import generate_dataset
from .features import FeaturePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

MODEL_CONFIGS = {
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
        "params": {
            "n_estimators": [150, 300],
            "max_depth": [20, None],
            "min_samples_split": [2, 5],
        },
    },
    "logistic_regression": {
        "estimator": LogisticRegression(max_iter=2000, random_state=42),
        "params": {
            "C": [0.1, 1.0, 10.0],
            "solver": ["liblinear"],
        },
    },
    "gradient_boosting": {
        "estimator": GradientBoostingClassifier(random_state=42),
        "params": {
            "n_estimators": [100],
            "max_depth": [3, 5],
            "learning_rate": [0.1],
        },
    },
}


def load_and_split(test_size: float = 0.2, random_state: int = 42):
    """Generate the dataset, split into train/test, and return as list-of-dicts."""
    data = generate_dataset(seed=random_state)
    df = pd.DataFrame(data)
    logger.info("Dataset: %d samples (phishing=%d, safe=%d)",
                len(df), df["label"].sum(), len(df) - df["label"].sum())

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["label"]
    )
    train_records = train_df[["subject", "body", "label"]].to_dict("records")
    test_records = test_df[["subject", "body", "label"]].to_dict("records")
    return train_records, test_records


def tune_model(name: str, config: dict, X_train, y_train) -> dict:
    """Run GridSearchCV for one model and return the best estimator + metadata."""
    logger.info("Tuning '%s' with GridSearchCV ...", name)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator=config["estimator"],
        param_grid=config["params"],
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    logger.info("  Best params: %s  (CV F1=%.4f)", search.best_params_, search.best_score_)
    return {
        "name": name,
        "estimator": search.best_estimator_,
        "best_params": search.best_params_,
        "cv_f1": search.best_score_,
    }


def evaluate_model(name: str, estimator, X_test, y_test) -> dict:
    """Compute standard classification metrics + confusion matrix."""
    y_pred = estimator.predict(X_test)
    y_prob = estimator.predict_proba(X_test)[:, 1] if hasattr(estimator, "predict_proba") else None

    metrics = {
        "name": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, target_names=["Safe", "Phishing"]),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }
    return metrics


def print_metrics(metrics: dict):
    """Pretty-print metrics for one model."""
    logger.info("--- %s ---", metrics["name"])
    logger.info("Accuracy : %.4f", metrics["accuracy"])
    logger.info("Precision: %.4f", metrics["precision"])
    logger.info("Recall   : %.4f", metrics["recall"])
    logger.info("F1-score : %.4f", metrics["f1"])
    logger.info("Confusion Matrix:\n%s", metrics["confusion_matrix"])
    logger.info("Classification Report:\n%s", metrics["classification_report"])


def get_feature_importance(estimator, pipeline: FeaturePipeline, top_n: int = 20) -> pd.DataFrame:
    """Return top-N features by importance for the best model, if available."""
    if not hasattr(estimator, "feature_importances_"):
        return pd.DataFrame()

    importances = estimator.feature_importances_
    struct_names = FeaturePipeline.FEATURE_NAMES
    tfidf_vocab = pipeline.tfidf.vocabulary_ or {}
    tfidf_names = sorted(tfidf_vocab, key=tfidf_vocab.get)
    all_names = struct_names + tfidf_names

    if len(all_names) != len(importances):
        logger.warning("Feature name count (%d) != importance count (%d); using indices.",
                       len(all_names), len(importances))
        all_names = [f"feature_{i}" for i in range(len(importances))]

    df = pd.DataFrame({"feature": all_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
    return df


def train_and_save():
    """Full training pipeline: load, feature-extract, tune, evaluate, save best."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_records, test_records = load_and_split()

    pipeline = FeaturePipeline(max_tfidf_features=3000, ngram_range=(1, 2))
    X_train = pipeline.fit_transform(train_records)
    y_train = np.array([r["label"] for r in train_records])
    X_test = pipeline.transform(test_records)
    y_test = np.array([r["label"] for r in test_records])

    logger.info("Feature matrix: train=%s, test=%s", X_train.shape, X_test.shape)

    # --- Train & tune all models ------------------------------------------
    results: list[dict] = []
    for name, config in MODEL_CONFIGS.items():
        tuned = tune_model(name, config, X_train, y_train)
        metrics = evaluate_model(name, tuned["estimator"], X_test, y_test)
        print_metrics(metrics)
        results.append({**tuned, **metrics})

    # --- Select best by F1 ------------------------------------------------
    best = max(results, key=lambda r: r["f1"])
    logger.info("Best model: '%s' (F1=%.4f, Accuracy=%.4f)", best["name"], best["f1"], best["accuracy"])

    # --- Feature importance (if available) --------------------------------
    fi_df = get_feature_importance(best["estimator"], pipeline, top_n=20)
    if not fi_df.empty:
        logger.info("Top 20 features:\n%s", fi_df.to_string(index=False))

    # --- Save artifacts ---------------------------------------------------
    model_path = os.path.join(MODEL_DIR, "phishing_model.joblib")
    pipeline_path = os.path.join(MODEL_DIR, "feature_pipeline.joblib")
    meta_path = os.path.join(MODEL_DIR, "model_metadata.joblib")

    joblib.dump(best["estimator"], model_path)
    joblib.dump(pipeline, pipeline_path)
    joblib.dump({
        "model_name": best["name"],
        "best_params": best["best_params"],
        "cv_f1": best["cv_f1"],
        "test_accuracy": best["accuracy"],
        "test_precision": best["precision"],
        "test_recall": best["recall"],
        "test_f1": best["f1"],
        "confusion_matrix": best["confusion_matrix"].tolist(),
        "classification_report": best["classification_report"],
    }, meta_path)

    logger.info("Saved model -> %s", model_path)
    logger.info("Saved pipeline -> %s", pipeline_path)
    logger.info("Saved metadata -> %s", meta_path)

    # Also save all model results for comparison
    all_results_path = os.path.join(MODEL_DIR, "all_results.joblib")
    joblib.dump([
        {
            "name": r["name"],
            "best_params": r["best_params"],
            "cv_f1": r["cv_f1"],
            "accuracy": r["accuracy"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
            "confusion_matrix": r["confusion_matrix"].tolist(),
        }
        for r in results
    ], all_results_path)

    return best, pipeline, results, fi_df


if __name__ == "__main__":
    train_and_save()
