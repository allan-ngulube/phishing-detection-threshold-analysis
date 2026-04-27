"""
prepare_datasets.py

Clean and normalize the Nazario, Ling-Spam, and Enron phishing datasets into a
shared schema:

    subject | body | label

Where:
    label = 1 -> phishing / spam
    label = 0 -> legitimate / ham

Workflow:
1. Load raw CSV files
2. Detect subject, body, and label columns
3. Normalize labels into binary values
4. Standardize missing values and text formatting
5. Print label distributions
6. Save cleaned datasets to disk
"""

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/clean")

NAZARIO_PATH = DATA_DIR / "Nazario.csv"
LING_PATH = DATA_DIR / "Ling.csv"
ENRON_PATH = DATA_DIR / "Enron.csv"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def find_first_existing_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
    dataset_name: str,
    column_type: str,
) -> str:
    """
    Return the first matching column name from the candidate list.

    Args:
        df: Input DataFrame.
        candidates: Possible column names to search for.
        dataset_name: Name of the dataset for error reporting.
        column_type: Logical type of column being searched for
                     (e.g., 'subject', 'body', 'label').

    Raises:
        ValueError: If none of the candidate columns exist.
    """
    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"[{dataset_name}] Could not find a {column_type} column. "
        f"Looked for: {list(candidates)}. "
        f"Available columns: {list(df.columns)}."
    )


def normalize_labels(
    series: pd.Series,
    dataset_name: str,
    explicit_mapping: Optional[dict] = None,
) -> pd.Series:
    """
    Normalize dataset labels into binary values:
        1 = phishing/spam
        0 = legitimate/ham

    Args:
        series: Label column.
        dataset_name: Dataset name for error reporting.
        explicit_mapping: Optional mapping for custom label values.

    Returns:
        A pandas Series of integer labels.

    Raises:
        ValueError: If labels cannot be interpreted safely.
    """
    if explicit_mapping is not None:
        normalized = series.map(explicit_mapping)
        if normalized.isna().any():
            unknown_values = series[normalized.isna()].dropna().unique()
            raise ValueError(
                f"[{dataset_name}] Some labels were not covered by the explicit "
                f"mapping: {list(unknown_values)}"
            )
        return normalized.astype(int)

    if pd.api.types.is_numeric_dtype(series):
        unique_values = set(series.dropna().unique())
        if unique_values.issubset({0, 1}):
            return series.astype(int)

        raise ValueError(
            f"[{dataset_name}] Numeric labels must be 0 or 1. "
            f"Found: {unique_values}"
        )

    normalized_text = series.astype(str).str.strip().str.lower()

    phishing_like = {"phish", "phishing", "spam", "malicious", "1", "bad"}
    legitimate_like = {"legit", "ham", "0", "good", "benign"}

    def map_value(value: str) -> int:
        if value in phishing_like:
            return 1
        if value in legitimate_like:
            return 0
        raise ValueError(
            f"[{dataset_name}] Unrecognized label value: '{value}'. "
            "Provide an explicit mapping for this dataset."
        )

    return normalized_text.apply(map_value).astype(int)


def clean_text_column(series: pd.Series, strip_whitespace: bool = True) -> pd.Series:
    """
    Fill missing values and convert a text column to string.

    Args:
        series: Input text column.
        strip_whitespace: Whether to strip surrounding whitespace.

    Returns:
        Cleaned text series.
    """
    cleaned = series.fillna("").astype(str)
    if strip_whitespace:
        cleaned = cleaned.str.strip()
    return cleaned


def analyze_labels_compact(df: pd.DataFrame, name: str) -> str:
    total = len(df)
    phishing = df["label"].sum()
    legit = total - phishing

    phishing_pct = (phishing / total) * 100
    legit_pct = (legit / total) * 100

    return (
        f"{name:<12} | Total: {total:<6} | "
        f"Phishing: {phishing:<6} ({phishing_pct:>5.1f}%) | "
        f"Legit: {legit:<6} ({legit_pct:>5.1f}%)"
    )


def standardize_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    subject_candidates: Iterable[str],
    body_candidates: Iterable[str],
    label_candidates: Iterable[str],
    explicit_mapping: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Convert a raw dataset into the shared schema:
        subject | body | label
    """
    subject_col = find_first_existing_column(
        df, subject_candidates, dataset_name, "subject"
    )
    body_col = find_first_existing_column(
        df, body_candidates, dataset_name, "body"
    )
    label_col = find_first_existing_column(
        df, label_candidates, dataset_name, "label"
    )

    cleaned_df = df[[subject_col, body_col, label_col]].copy()
    cleaned_df.columns = ["subject", "body", "label"]

    cleaned_df["label"] = normalize_labels(
        cleaned_df["label"], dataset_name, explicit_mapping
    )
    cleaned_df["subject"] = clean_text_column(cleaned_df["subject"], strip_whitespace=True)
    cleaned_df["body"] = clean_text_column(cleaned_df["body"], strip_whitespace=False)

    return cleaned_df


# ---------------------------------------------------------------------------
# Dataset-specific wrappers
# ---------------------------------------------------------------------------

def normalize_nazario(df: pd.DataFrame) -> pd.DataFrame:
    return standardize_dataset(
        df=df,
        dataset_name="Nazario",
        subject_candidates=["subject", "Subject", "SUBJECT"],
        body_candidates=["body", "Body", "BODY", "text", "Text", "message", "Message"],
        label_candidates=["label", "Label", "class", "Class", "category", "Category"],
        explicit_mapping=None,
    )


def normalize_ling(df: pd.DataFrame) -> pd.DataFrame:
    return standardize_dataset(
        df=df,
        dataset_name="Ling-Spam",
        subject_candidates=["subject", "Subject", "SUBJECT"],
        body_candidates=["body", "Body", "BODY", "text", "Text", "message", "Message"],
        label_candidates=["label", "Label", "class", "Class", "is_spam", "spam"],
        explicit_mapping=None,
    )


def normalize_enron(df: pd.DataFrame) -> pd.DataFrame:
    return standardize_dataset(
        df=df,
        dataset_name="Enron",
        subject_candidates=["subject", "Subject", "SUBJECT"],
        body_candidates=[
            "body",
            "Body",
            "BODY",
            "text",
            "Text",
            "message",
            "Message",
            "content",
        ],
        label_candidates=["label", "Label", "class", "Class", "is_phish", "is_spam"],
        explicit_mapping=None,
    )


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw datasets...")
    nazario_raw = pd.read_csv(NAZARIO_PATH)
    ling_raw = pd.read_csv(LING_PATH)
    enron_raw = pd.read_csv(ENRON_PATH)

    print("Normalizing datasets...")
    nazario_clean = normalize_nazario(nazario_raw)
    ling_clean = normalize_ling(ling_raw)
    enron_clean = normalize_enron(enron_raw)

    # -------------------------------
    # Compact Output (One Screen)
    # -------------------------------
    print("\n================ DATASET SUMMARY ================\n")

    print(analyze_labels_compact(nazario_clean, "Nazario"))
    print(analyze_labels_compact(ling_clean, "Ling-Spam"))
    print(analyze_labels_compact(enron_clean, "Enron"))

    # -------------------------------
    # Save files
    # -------------------------------
    nazario_output = OUTPUT_DIR / "nazario_clean.csv"
    ling_output = OUTPUT_DIR / "ling_clean.csv"
    enron_output = OUTPUT_DIR / "enron_clean.csv"

    nazario_clean.to_csv(nazario_output, index=False)
    ling_clean.to_csv(ling_output, index=False)
    enron_clean.to_csv(enron_output, index=False)

    # -------------------------------
    # Output files (clean display)
    # -------------------------------
    print("\n============== OUTPUT FILES ==============\n")
    print(f"Nazario   → {nazario_output.name}")
    print(f"Ling-Spam → {ling_output.name}")
    print(f"Enron     → {enron_output.name}")

    print("\nSchema: subject | body | label (1=phishing, 0=legit)")
    print("==================================================\n")


if __name__ == "__main__":
    main()