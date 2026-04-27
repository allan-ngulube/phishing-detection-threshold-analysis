"""
compare_fixed_vs_dataset_specific_thresholds.py

Compare:
1. Fixed threshold (same across datasets)
2. Dataset-specific thresholds (optimized per dataset)

Author: Allan Ngulube
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATA_DIR = Path("data/clean")
OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "enron": DATA_DIR / "enron_clean.csv",
    "ling": DATA_DIR / "ling_clean.csv",
    "nazario": DATA_DIR / "nazario_clean.csv",
}

TRAIN_DATASET = "enron"
TEST_DATASETS = ["enron", "ling", "nazario"]

FIXED_THRESHOLD = 0.5
THRESHOLD_GRID = np.arange(0.1, 0.95, 0.05)

OPTIMIZE_FOR = "f1"  # can be: f1, precision, recall, accuracy


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_dataset(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATASETS[name])
    df["label"] = df["label"].astype(int)
    return df


def build_text_field(df: pd.DataFrame) -> pd.Series:
    return (df["subject"].fillna("") + " " + df["body"].fillna("")).str.strip()


def train_model(train_df: pd.DataFrame):
    x_train = build_text_field(train_df)
    y_train = train_df["label"].to_numpy()

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        stop_words="english",
    )

    x_train_vec = vectorizer.fit_transform(x_train)

    model = LogisticRegression(max_iter=1000, n_jobs=-1)
    model.fit(x_train_vec, y_train)

    return vectorizer, model


def safe_confusion_matrix(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn, fp, fn, tp


def compute_metrics(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = safe_confusion_matrix(y_true, y_pred)

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) else 0,
        "fnr": fn / (fn + tp) if (fn + tp) else 0,
        "alerts": tp + fp,
        "roc_auc": roc_auc_score(y_true, y_proba)
        if len(np.unique(y_true)) == 2 else np.nan,
    }


def find_best_threshold(y_true, y_proba):
    best_t = FIXED_THRESHOLD
    best_metrics = None
    best_score = -1

    for t in THRESHOLD_GRID:
        metrics = compute_metrics(y_true, y_proba, t)
        score = metrics[OPTIMIZE_FOR]

        if score > best_score:
            best_score = score
            best_t = t
            best_metrics = metrics

    return best_t, best_metrics


def format_output(row):
    return (
        f"{row['dataset']:<8} | "
        f"Fixed({row['fixed_t']:.2f}) → "
        f"F1:{row['fixed_f1']:.3f} "
        f"FPR:{row['fixed_fpr']:.3f} "
        f"FNR:{row['fixed_fnr']:.3f} "
        f"A:{row['fixed_alerts']} || "
        f"Best({row['best_t']:.2f}) → "
        f"F1:{row['best_f1']:.3f} "
        f"FPR:{row['best_fpr']:.3f} "
        f"FNR:{row['best_fnr']:.3f} "
        f"A:{row['best_alerts']}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("\nTraining model...\n")

    train_df = load_dataset(TRAIN_DATASET)
    vectorizer, model = train_model(train_df)

    results = []

    for name in TEST_DATASETS:
        df = load_dataset(name)

        x = build_text_field(df)
        y_true = df["label"].to_numpy()

        x_vec = vectorizer.transform(x)
        y_proba = model.predict_proba(x_vec)[:, 1]

        fixed = compute_metrics(y_true, y_proba, FIXED_THRESHOLD)
        best_t, best = find_best_threshold(y_true, y_proba)

        results.append({
            "dataset": name,

            "fixed_t": FIXED_THRESHOLD,
            "fixed_f1": fixed["f1"],
            "fixed_fpr": fixed["fpr"],
            "fixed_fnr": fixed["fnr"],
            "fixed_alerts": fixed["alerts"],

            "best_t": best_t,
            "best_f1": best["f1"],
            "best_fpr": best["fpr"],
            "best_fnr": best["fnr"],
            "best_alerts": best["alerts"],

            # full metrics (for CSV / analysis)
            "fixed_TP": fixed["TP"],
            "fixed_TN": fixed["TN"],
            "fixed_FP": fixed["FP"],
            "fixed_FN": fixed["FN"],

            "best_TP": best["TP"],
            "best_TN": best["TN"],
            "best_FP": best["FP"],
            "best_FN": best["FN"],

            "roc_auc": fixed["roc_auc"],
        })

    print("=============== THRESHOLD COMPARISON ===============\n")

    for r in results:
        print(format_output(r))

    print("\n====================================================\n")

    df_out = pd.DataFrame(results)
    out_file = OUTPUT_DIR / "threshold_comparison.csv"
    df_out.to_csv(out_file, index=False)

    print(f"Saved results → {out_file}")


if __name__ == "__main__":
    main()