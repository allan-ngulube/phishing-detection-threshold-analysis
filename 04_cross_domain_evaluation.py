from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/clean")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "nazario": DATA_DIR / "nazario_clean.csv",
    "ling": DATA_DIR / "ling_clean.csv",
    "enron": DATA_DIR / "enron_clean.csv",
}

THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def build_text_field(df: pd.DataFrame) -> pd.Series:
    """Combine subject and body into a single text field."""
    subject = df["subject"].fillna("").astype(str)
    body = df["body"].fillna("").astype(str)
    return (subject + " " + body).str.strip()


def load_dataset(name: str) -> pd.DataFrame:
    """Load a cleaned dataset and validate required columns."""
    path = DATASETS[name]

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    required_columns = {"subject", "body", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"[{name}] Missing required columns. Found: {list(df.columns)}"
        )

    df["label"] = df["label"].astype(int)
    return df


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> dict:
    """Compute confusion matrix counts and derived metrics at a given threshold."""
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    balanced_accuracy = (recall + specificity) / 2
    alert_volume = int(tp + fp)
    predicted_positive_rate = (tp + fp) / total if total > 0 else 0.0
    predicted_negative_rate = (tn + fn) / total if total > 0 else 0.0

    return {
        "threshold": float(threshold),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "true_positive_rate": float(recall),
        "specificity": float(specificity),
        "true_negative_rate": float(specificity),
        "f1": float(f1),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "balanced_accuracy": float(balanced_accuracy),
        "alert_volume": int(alert_volume),
        "predicted_positive_rate": float(predicted_positive_rate),
        "predicted_negative_rate": float(predicted_negative_rate),
        "confusion_matrix": cm.tolist(),
    }


def format_results_compact(rows: list[dict]) -> str:
    """Create a compact console table for one target dataset."""
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

    for row in rows:
        roc_auc_value = row["roc_auc"]
        auc_str = roc_auc_value if isinstance(roc_auc_value, str) else f"{roc_auc_value:.3f}"

        lines.append(
            f"{row['threshold']:<4.1f} | "
            f"{row['TP']:>6} | "
            f"{row['TN']:>6} | "
            f"{row['FP']:>6} | "
            f"{row['FN']:>6} | "
            f"{row['recall'] * 100:>6.1f}% | "
            f"{row['specificity'] * 100:>6.1f}% | "
            f"{row['false_positive_rate'] * 100:>6.1f}% | "
            f"{row['false_negative_rate'] * 100:>6.1f}% | "
            f"{row['precision'] * 100:>6.1f}% | "
            f"{row['f1'] * 100:>6.1f}% | "
            f"{row['alert_volume']:>7} | "
            f"{auc_str:>5}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model training and evaluation
# ---------------------------------------------------------------------------

def train_on_source(source_name: str):
    """Train TF-IDF + Logistic Regression on the source dataset."""
    df_source = load_dataset(source_name)
    x_train_text = build_text_field(df_source)
    y_train = df_source["label"].to_numpy()

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        lowercase=True,
        stop_words="english",
    )

    x_train_vec = vectorizer.fit_transform(x_train_text)

    model = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
    )

    model.fit(x_train_vec, y_train)
    return vectorizer, model


def evaluate_on_target(
    vectorizer: TfidfVectorizer,
    model: LogisticRegression,
    source_name: str,
    target_name: str,
) -> list[dict]:
    """Evaluate a trained model on a target dataset across thresholds."""
    df_target = load_dataset(target_name)
    x_test_text = build_text_field(df_target)
    y_true = df_target["label"].to_numpy()

    x_test_vec = vectorizer.transform(x_test_text)
    y_proba = model.predict_proba(x_test_vec)[:, 1]

    roc_auc = "N/A"
    if len(np.unique(y_true)) == 2:
        roc_auc = float(roc_auc_score(y_true, y_proba))

    label_distribution = pd.Series(y_true).value_counts().sort_index().to_dict()

    results = []
    for threshold in THRESHOLDS:
        row = compute_metrics_at_threshold(y_true, y_proba, threshold)
        row.update(
            {
                "train_dataset": source_name,
                "test_dataset": target_name,
                "n_test": int(len(y_true)),
                "roc_auc": roc_auc,
                "label_dist_test": label_distribution,
            }
        )
        results.append(row)

    return results


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main() -> None:
    all_rows = []

    train_sources = ["enron"]
    test_targets = ["enron", "ling", "nazario"]

    print("\n================ CROSS-DOMAIN EVALUATION ================\n")

    for source_name in train_sources:
        print(f"Training source: {source_name}")
        print("-" * 58)

        vectorizer, model = train_on_source(source_name)

        for target_name in test_targets:
            rows = evaluate_on_target(vectorizer, model, source_name, target_name)

            print(f"\nTarget dataset: {target_name} | Trained on: {source_name}")
            print(format_results_compact(rows))

            all_rows.extend(rows)

    results_df = pd.DataFrame(all_rows)

    csv_path = MODEL_DIR / "cross_domain_results.csv"
    json_path = MODEL_DIR / "cross_domain_results.json"

    results_df.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(all_rows, file, indent=2)

    print("\n================ SAVED OUTPUTS ================\n")
    print(f"CSV  -> {csv_path.name}")
    print(f"JSON -> {json_path.name}")
    print("\nDone.\n")


if __name__ == "__main__":
    main()

