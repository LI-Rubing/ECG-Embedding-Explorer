import numpy as np
import pandas as pd


def _is_array(value) -> bool:
    return isinstance(value, np.ndarray)


def _array_length(value) -> int | None:
    if not _is_array(value):
        return None
    return int(value.shape[0])


def validate_schema(
    df_embeddings,
    df_ecg,
    embedding_col: str,
    key: str = "ecg_id",
):
    errors: list[str] = []
    warnings: list[str] = []

    if key not in df_embeddings.columns:
        errors.append(f"Join key '{key}' is missing from df_embeddings.")
    if key not in df_ecg.columns:
        errors.append(f"Join key '{key}' is missing from df_ecg.")
    if embedding_col not in df_embeddings.columns:
        errors.append(
            f"Embedding column '{embedding_col}' is missing from df_embeddings."
        )

    if key in df_embeddings.columns and key in df_ecg.columns:
        missing_ids = set(df_embeddings[key]) - set(df_ecg[key])
        if missing_ids:
            errors.append(
                f"{len(missing_ids)} ecg_id values in df_embeddings are missing from df_ecg."
            )

    if embedding_col in df_embeddings.columns:
        lengths = []
        non_arrays = 0
        for value in df_embeddings[embedding_col]:
            length = _array_length(value)
            if length is None:
                non_arrays += 1
            else:
                lengths.append(length)
        if non_arrays:
            errors.append(
                f"Embedding column '{embedding_col}' contains {non_arrays} non-numpy entries."
            )
        if lengths and len(set(lengths)) != 1:
            errors.append(
                f"Embedding column '{embedding_col}' contains arrays of varying lengths."
            )

    if key in df_ecg.columns:
        lead_cols = [col for col in df_ecg.columns if col != key]
        if not lead_cols:
            errors.append("df_ecg has no ECG lead columns.")
        else:
            lead_lengths = []
            non_arrays = 0
            for col in lead_cols:
                for value in df_ecg[col]:
                    length = _array_length(value)
                    if length is None:
                        non_arrays += 1
                    else:
                        lead_lengths.append(length)
            if non_arrays:
                errors.append(
                    f"ECG lead columns contain {non_arrays} non-numpy entries."
                )
            if lead_lengths and len(set(lead_lengths)) != 1:
                errors.append("ECG lead columns contain arrays of varying lengths.")

    ok = len(errors) == 0
    return ok, errors, warnings


def detect_array_columns(df: pd.DataFrame) -> set[str]:
    array_cols: set[str] = set()
    for col in df.columns:
        series = df[col]
        sample = series.dropna()
        if sample.empty:
            continue
        first = sample.iloc[0]
        if isinstance(first, (np.ndarray, list, tuple)):
            array_cols.add(col)
    return array_cols


def embedding_length(value: object) -> int | None:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return None
        return int(value.shape[0])
    if isinstance(value, (list, tuple)):
        return len(value)
    return None


def validate_embedding_length(df: pd.DataFrame, embedding_col: str) -> list[str]:
    errors: list[str] = []
    if embedding_col not in df.columns:
        return errors
    lengths = []
    non_arrays = 0
    for value in df[embedding_col]:
        length = embedding_length(value)
        if length is None:
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                non_arrays += 1
            continue
        lengths.append(int(length))
    if non_arrays:
        errors.append(
            f"Embedding column '{embedding_col}' contains {non_arrays} non-array entries."
        )
    if not lengths:
        errors.append(
            f"Embedding column '{embedding_col}' has no array-like embeddings."
        )
    if lengths and len(set(lengths)) != 1:
        errors.append(
            f"Embedding column '{embedding_col}' contains arrays of varying lengths."
        )
    return errors


def summarize_df(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[col].dtype) for col in df.columns],
            "missing": [int(df[col].isna().sum()) for col in df.columns],
        }
    )
