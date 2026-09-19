"""
Data loading and validation utilities for the AI4I 2020 Predictive
Maintenance dataset.

This module is intentionally limited to data-engineering concerns:
loading the raw CSV and validating its structure/quality. No
preprocessing, statistical modeling, or visualization is performed
here.
"""

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Resolve paths relative to this file so the module works regardless
# of the current working directory the app/script is launched from.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
DATA_PATH = _PROJECT_ROOT / "data" / "ai4i2020.csv"

# Columns that must be present in the raw dataset.
EXPECTED_COLUMNS: List[str] = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


def load_dataset(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the AI4I 2020 dataset from CSV into a pandas DataFrame.

    Args:
        csv_path: Path to the CSV file. Defaults to
            ``data/ai4i2020.csv`` resolved relative to the project
            root, so this works no matter where the app is launched
            from.

    Returns:
        A pandas DataFrame containing the raw, unmodified dataset.

    Raises:
        FileNotFoundError: If the CSV file does not exist at the
            resolved path.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at expected location: {csv_path}. "
            "Ensure 'ai4i2020.csv' is placed inside the 'data/' folder "
            "at the project root."
        )
    return pd.read_csv(csv_path)


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate the structure and quality of the loaded dataset and
    build a summary report.

    Checks performed:
        - presence of all expected columns
        - row and column counts
        - column names and data types
        - missing values per column and in total
        - duplicate row count
        - Machine failure value counts and failure percentage

    Args:
        df: The DataFrame returned by ``load_dataset()``.

    Returns:
        A dictionary summarizing the validation results.

    Raises:
        ValueError: If any expected column is missing from the
            DataFrame.
    """
    missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            "Dataset is missing expected column(s): "
            f"{missing_columns}. Expected columns are: {EXPECTED_COLUMNS}."
        )

    row_count = len(df)
    column_count = df.shape[1]
    column_names = list(df.columns)
    dtypes = df.dtypes.astype(str).to_dict()

    missing_by_column = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())

    duplicate_row_count = int(df.duplicated().sum())

    failure_counts = df["Machine failure"].value_counts().to_dict()
    failure_percentage = (
        df["Machine failure"].sum() / row_count * 100 if row_count > 0 else 0.0
    )

    summary: Dict[str, Any] = {
        "row_count": row_count,
        "column_count": column_count,
        "column_names": column_names,
        "dtypes": dtypes,
        "missing_by_column": missing_by_column,
        "total_missing": total_missing,
        "duplicate_row_count": duplicate_row_count,
        "machine_failure_counts": failure_counts,
        "machine_failure_percentage": failure_percentage,
    }
    return summary


def print_validation_summary(summary: Dict[str, Any]) -> None:
    """
    Print a human-readable validation summary to the console.

    Args:
        summary: The dictionary returned by ``validate_dataset()``.
    """
    print("=" * 60)
    print("AI4I 2020 DATASET VALIDATION SUMMARY")
    print("=" * 60)

    print(f"\nRow count: {summary['row_count']}")
    print(f"Column count: {summary['column_count']}")

    print("\nColumn names:")
    for name in summary["column_names"]:
        print(f"  - {name}")

    print("\nData types:")
    for col, dtype in summary["dtypes"].items():
        print(f"  - {col}: {dtype}")

    print("\nMissing values by column:")
    for col, count in summary["missing_by_column"].items():
        print(f"  - {col}: {count}")
    print(f"\nTotal missing values: {summary['total_missing']}")

    print(f"\nDuplicate row count: {summary['duplicate_row_count']}")

    print("\nMachine failure value counts:")
    for value, count in summary["machine_failure_counts"].items():
        print(f"  - {value}: {count}")
    print(f"\nMachine failure percentage: {summary['machine_failure_percentage']:.4f}%")

    print("=" * 60)


if __name__ == "__main__":
    dataset = load_dataset()
    validation_summary = validate_dataset(dataset)
    print_validation_summary(validation_summary)
