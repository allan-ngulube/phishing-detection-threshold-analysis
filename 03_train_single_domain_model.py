"""
train_single_dataset_baseline.py

Train a baseline phishing classifier (TF-IDF + Logistic Regression) on a
single cleaned dataset.

Supported datasets:
- enron
- ling

Note:
- nazario is phishing-only (label = 1), so it cannot be used by itself to
  train a binary classifier.

Examples:
    python train_single_dataset_baseline.py --dataset enron
    python train_single_dataset_baseline.py --dataset ling
"""

from pathlib import Path
import argparse
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

SUPPORTED_DATASETS = {
    "enron": "enron_clean.csv",
    "ling": "ling_clean.csv",
    "nazario": "nazario_clean.csv",
}

TEST_SIZE = 0.2
RANDOM_STATE = 42
THRESHOLD = 0.5


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def build_text_field(df: pd.DataFrame) -> pd.Series:
    """Combine subject and body into a single text field."""
    subject = df["subject"].fillna("").astype(str)
    body = df["body"].fillna("").astype(str)
    return (subject + " " + body).str.strip()


def load_single_dataset(dataset_name: str) -> pd.DataFrame:
    """
    Load one cleaned dataset and validate that it can be used for binary
    classification.
    """
    dataset_name = dataset_name.lower().strip()

    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Choose from: {list(SUPPORTED_DATASETS.keys())}"
        )

    path = DATA_DIR / SUPPORTED_DATASETS[dataset_name]
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    required_columns = {"subject", "body", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"{dataset_name} is missing required columns. "
            f"Found: {list(df.columns)}"
        )

    df["label"] = df["label"].astype(int)

    unique_labels = set(df["label"].dropna().unique())
    if len(unique_labels) < 2:
        raise ValueError(
            f"{dataset_name} has only one class present: {unique_labels}. "
            "Pick 'enron' or 'ling' for single-dataset training."
        )

    return df


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def compute_metrics(y_true, y_proba, threshold: float) -> dict:
    """Compute confusion-matrix counts and derived metrics."""
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

def train_and_evaluate(df: pd.DataFrame):
    """
    Train a single-dataset baseline model and return:
    - fitted vectorizer
    - fitted model
    - metrics
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

    metrics.update(
        {
            "n_total": int(len(df)),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "label_distribution": {
                int(k): int(v)
                for k, v in df["label"].value_counts().sort_index().to_dict().items()
            },
        }
    )

    return vectorizer, model, metrics


# ---------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------

def print_dataset_summary(dataset_name: str, metrics: dict) -> None:
    """Print a compact summary of the dataset and train/test split."""
    label_dist = metrics["label_distribution"]
    total = metrics["n_total"]
    phishing = label_dist.get(1, 0)
    legitimate = label_dist.get(0, 0)

    phishing_pct = (phishing / total) * 100 if total > 0 else 0.0
    legitimate_pct = (legitimate / total) * 100 if total > 0 else 0.0

    print("\n================ SINGLE-DATASET BASELINE ================\n")
    print(f"Dataset        : {dataset_name}")
    print(f"Total Samples  : {total}")
    print(f"Phishing       : {phishing} ({phishing_pct:.1f}%)")
    print(f"Legitimate     : {legitimate} ({legitimate_pct:.1f}%)")
    print(f"Train/Test     : {metrics['n_train']}/{metrics['n_test']}")
    print(f"Threshold      : {metrics['threshold']}")


def print_metrics_summary(metrics: dict) -> None:
    """Print a compact, presentation-friendly metric summary."""
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


def print_saved_artifacts(model_path: Path, vectorizer_path: Path, metrics_path: Path) -> None:
    """Print saved artifact names."""
    print("\n---------------- SAVED OUTPUTS ----------------\n")
    print(f"Model      -> {model_path.name}")
    print(f"Vectorizer -> {vectorizer_path.name}")
    print(f"Metrics    -> {metrics_path.name}")
    print("\n=================================================\n")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a single-dataset phishing baseline model."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="enron",
        help="Dataset to use: enron | ling | nazario",
    )
    args = parser.parse_args()

    dataset_name = args.dataset.lower().strip()
    df = load_single_dataset(dataset_name)

    vectorizer, model, metrics = train_and_evaluate(df)

    model_path = MODEL_DIR / f"baseline_logreg_{dataset_name}.joblib"
    vectorizer_path = MODEL_DIR / f"tfidf_vectorizer_{dataset_name}.joblib"
    metrics_path = MODEL_DIR / f"baseline_metrics_{dataset_name}.json"

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print_dataset_summary(dataset_name, metrics)
    print_metrics_summary(metrics)
    print_saved_artifacts(model_path, vectorizer_path, metrics_path)


if __name__ == "__main__":
    main()