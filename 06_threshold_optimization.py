"""
06_threshold_optimization.py

Analyze how phishing detection performance changes as the decision threshold varies.

- Uses the combined cleaned dataset (Nazario + Ling-Spam + Enron)
- Trains a TF-IDF + Logistic Regression model
- Evaluates multiple thresholds
- Reports confusion-matrix counts and derived metrics
- Saves results to CSV
"""

from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATA_DIR = Path("data/clean")

OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "threshold_metrics_combined.csv"

THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_combined_dataset() -> pd.DataFrame:
    """Load and combine cleaned Nazario, Ling-Spam, and Enron datasets."""
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
# Training
# ---------------------------------------------------------------------

def train_model_and_get_probs(df: pd.DataFrame):
    """
    Train TF-IDF + Logistic Regression and return:
    - y_test
    - predicted probabilities for the phishing class
    """
    x_text = build_text_field(df)
    y = df["label"].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x_text,
        y,
        test_size=0.2,
        random_state=42,
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
    return y_test.reset_index(drop=True), y_proba


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def compute_metrics_at_threshold(
    y_true: pd.Series,
    y_proba,
    threshold: float,
) -> dict:
    """Compute confusion-matrix counts and derived metrics at one threshold."""
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    total = tp + tn + fp + fn

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    balanced_accuracy = (recall + specificity) / 2
    alert_volume = tp + fp
    predicted_positive_rate = (tp + fp) / total if total > 0 else 0.0

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
        "specificity": float(specificity),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "balanced_accuracy": float(balanced_accuracy),
        "alert_volume": int(alert_volume),
        "predicted_positive_rate": float(predicted_positive_rate),
    }


def evaluate_at_thresholds(y_true: pd.Series, y_proba, thresholds: list[float]) -> pd.DataFrame:
    """Evaluate model outputs across multiple thresholds."""
    rows = [compute_metrics_at_threshold(y_true, y_proba, t) for t in thresholds]
    return pd.DataFrame(rows)


def format_results_compact(results_df: pd.DataFrame, roc_auc: float) -> str:
    """Create a clean, aligned console table."""
    lines = []

    header = (
        f"{'Thr':<4} | "
        f"{'TP':>6} | "
        f"{'TN':>6} | "
        f"{'FP':>6} | "
        f"{'FN':>6} | "
        f"{'Recall':>7} | "
        f"{'Spec.':>7} | "
        f"{'FPR':>7} | "
        f"{'FNR':>7} | "
        f"{'Prec.':>7} | "
        f"{'F1':>7} | "
        f"{'Alerts':>7} | "
        f"{'AUC':>5}"
    )

    lines.append(header)
    lines.append("-" * len(header))

    for _, row in results_df.iterrows():
        lines.append(
            f"{row['threshold']:<4.1f} | "
            f"{int(row['TP']):>6} | "
            f"{int(row['TN']):>6} | "
            f"{int(row['FP']):>6} | "
            f"{int(row['FN']):>6} | "
            f"{row['recall']*100:>6.1f}% | "
            f"{row['specificity']*100:>6.1f}% | "
            f"{row['false_positive_rate']*100:>6.1f}% | "
            f"{row['false_negative_rate']*100:>6.1f}% | "
            f"{row['precision']*100:>6.1f}% | "
            f"{row['f1']*100:>6.1f}% | "
            f"{int(row['alert_volume']):>7} | "
            f"{roc_auc:>5.3f}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print("\n=============== THRESHOLD ANALYSIS: COMBINED DATASET ===============\n")

    df = load_combined_dataset()

    print(f"Total samples: {len(df)}")
    print("Label distribution:")
    print(df["label"].value_counts().sort_index().rename(index={0: "Legitimate", 1: "Phishing"}))

    y_test, y_proba = train_model_and_get_probs(df)
    roc_auc = roc_auc_score(y_test, y_proba)

    results_df = evaluate_at_thresholds(y_test, y_proba, THRESHOLDS)

    print("\nThreshold results:")
    print(format_results_compact(results_df, roc_auc))

    results_df["roc_auc"] = roc_auc
    results_df.to_csv(OUTPUT_FILE, index=False)

    print("\n================ SAVED OUTPUT ================\n")
    print(f"CSV -> {OUTPUT_FILE.name}")
    print("\nDone.\n")


if __name__ == "__main__":
    main()