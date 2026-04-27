"""
02_train_baseline_model.py

Train a baseline phishing classifier using the cleaned Nazario, Ling-Spam,
and Enron datasets.

Pipeline:
1. Load cleaned datasets
2. Combine into one dataset
3. Merge subject + body into a single text field
4. Split into training and test sets
5. Vectorize text using TF-IDF (unigrams + bigrams)
6. Train Logistic Regression
7. Evaluate on a held-out test set
8. Save model, vectorizer, and metrics
"""

from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATA_DIR = Path("data/clean")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "baseline_logreg.joblib"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
METRICS_PATH = MODEL_DIR / "baseline_metrics.json"

TEST_SIZE = 0.2
RANDOM_STATE = 42
THRESHOLD = 0.5


# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------

def load_clean_datasets() -> pd.DataFrame:
    """
    Load the cleaned Nazario, Ling-Spam, and Enron datasets and combine them
    into a single DataFrame.
    """
    nazario = pd.read_csv(DATA_DIR / "nazario_clean.csv")
    ling = pd.read_csv(DATA_DIR / "ling_clean.csv")
    enron = pd.read_csv(DATA_DIR / "enron_clean.csv")

    nazario["source"] = "nazario"
    ling["source"] = "ling"
    enron["source"] = "enron"

    combined = pd.concat([nazario, ling, enron], ignore_index=True)

    required_columns = {"subject", "body", "label"}
    if not required_columns.issubset(combined.columns):
        raise ValueError(
            f"Missing required columns. Found: {list(combined.columns)}"
        )

    combined["label"] = combined["label"].astype(int)
    return combined


def build_text_field(df: pd.DataFrame) -> pd.Series:
    """Combine subject and body into a single text field."""
    subject = df["subject"].fillna("").astype(str)
    body = df["body"].fillna("").astype(str)
    return (subject + " " + body).str.strip()


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def compute_metrics(y_true, y_proba, threshold: float) -> dict:
    """
    Compute confusion-matrix counts and derived metrics at a given threshold.
    """
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_proba)

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    balanced_accuracy = (recall + specificity) / 2
    alert_volume = tp + fp

    return {
        "threshold": float(threshold),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "specificity": float(specificity),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "balanced_accuracy": float(balanced_accuracy),
        "alert_volume": int(alert_volume),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_baseline_model(df: pd.DataFrame):
    """
    Train the baseline phishing classifier and return:
    - fitted vectorizer
    - fitted model
    - evaluation metrics
    """
    x_text = build_text_field(df)
    y = df["label"].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x_text,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        lowercase=True,
        stop_words="english",
    )

    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    model = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
    )
    model.fit(x_train_vec, y_train)

    y_proba = model.predict_proba(x_test_vec)[:, 1]
    metrics = compute_metrics(y_test, y_proba, THRESHOLD)

    return vectorizer, model, metrics


# ---------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------

def print_dataset_summary(df: pd.DataFrame) -> None:
    """Print a compact summary of the combined dataset."""
    total = len(df)
    phishing = int(df["label"].sum())
    legitimate = total - phishing

    phishing_pct = (phishing / total) * 100 if total > 0 else 0.0
    legitimate_pct = (legitimate / total) * 100 if total > 0 else 0.0

    print("\n================ BASELINE TRAINING SUMMARY ================\n")
    print(f"Total Samples : {total}")
    print(
        f"Phishing      : {phishing} ({phishing_pct:.1f}%)"
    )
    print(
        f"Legitimate    : {legitimate} ({legitimate_pct:.1f}%)"
    )
    print(f"Train/Test    : {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)}")
    print(f"Threshold     : {THRESHOLD}")


def print_metrics_summary(metrics: dict) -> None:
    """Print a compact, presentation-friendly summary of baseline metrics."""
    print("\n---------------- CONFUSION MATRIX ----------------\n")
    print(f"TP: {metrics['TP']} | FP: {metrics['FP']}")
    print(f"TN: {metrics['TN']} | FN: {metrics['FN']}")

    print("\n---------------- PERFORMANCE ----------------\n")
    print(f"Accuracy       : {metrics['accuracy']:.4f}")
    print(f"Precision      : {metrics['precision']:.4f}")
    print(f"Recall         : {metrics['recall']:.4f}")
    print(f"F1 Score       : {metrics['f1']:.4f}")
    print(f"ROC-AUC        : {metrics['roc_auc']:.4f}")
    print(f"Specificity    : {metrics['specificity']:.4f}")

    print("\n------------- RISK / GOVERNANCE VIEW -------------\n")
    print(f"False Pos Rate : {metrics['false_positive_rate']:.4f}")
    print(f"False Neg Rate : {metrics['false_negative_rate']:.4f}")
    print(f"Alert Volume   : {metrics['alert_volume']}")


def print_saved_artifacts() -> None:
    """Print saved output artifact names."""
    print("\n---------------- SAVED OUTPUTS ----------------\n")
    print(f"Model      -> {MODEL_PATH.name}")
    print(f"Vectorizer -> {VECTORIZER_PATH.name}")
    print(f"Metrics    -> {METRICS_PATH.name}")
    print("\n=================================================\n")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    df = load_clean_datasets()
    print_dataset_summary(df)

    vectorizer, model, metrics = train_baseline_model(df)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print_metrics_summary(metrics)
    print_saved_artifacts()


if __name__ == "__main__":
    main()