import os
from functools import lru_cache

import pandas as pd


def load_dataframe(path: str) -> pd.DataFrame:
    extension = path.rsplit(".", 1)[-1].lower()

    if extension == "pkl":
        return pd.read_pickle(path)
    if extension == "csv":
        return pd.read_csv(path)
    if extension == "parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported file extension: .{extension}")


def load_with_fallback(primary_path: str, fallback_path: str) -> pd.DataFrame:
    if os.path.exists(primary_path):
        return load_dataframe(primary_path)
    if os.path.exists(fallback_path):
        return load_dataframe(fallback_path)
    raise FileNotFoundError(
        f"Neither '{primary_path}' nor '{fallback_path}' exists."
    )


def load_ecg_dataframe(ecg_upload, ecg_path: str) -> pd.DataFrame | None:
    if ecg_upload is not None:
        return load_dataframe_from_upload(ecg_upload)
    if not ecg_path:
        return None
    ext = os.path.splitext(ecg_path)[1].lower()
    if ext in {".csv", ".pkl"}:
        return load_dataframe(ecg_path)
    return None


def load_dataframe_from_upload(uploaded_file) -> pd.DataFrame:
    import tempfile

    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    return load_dataframe(tmp_path)


def load_ecg_metadata(ecg_path: str) -> tuple[list[str], int | None]:
    if os.path.exists(ecg_path):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "Missing dependency 'pyarrow' for parquet metadata. "
                "Install it with: pip install pyarrow"
            ) from exc
        parquet_file = pq.ParquetFile(ecg_path)
        num_rows = parquet_file.metadata.num_rows if parquet_file.metadata else None
        return parquet_file.schema.names, num_rows
    raise FileNotFoundError(f"Parquet file not found: '{ecg_path}'")


def load_ecg_row(ecg_id: str, ecg_path: str) -> pd.DataFrame:
    if os.path.exists(ecg_path):
        ext = os.path.splitext(ecg_path)[1].lower()
        if ext == ".parquet":
            return _load_ecg_row_fast(ecg_id, ecg_path)
        try:
            return pd.read_parquet(ecg_path, filters=[("ecg_id", "==", ecg_id)])
        except Exception:
            return _load_ecg_row_fast(ecg_id, ecg_path)
    raise FileNotFoundError(f"Parquet file not found: '{ecg_path}'")


@lru_cache(maxsize=2)
def _get_parquet_file(ecg_path: str):
    import pyarrow.parquet as pq

    return pq.ParquetFile(ecg_path)


@lru_cache(maxsize=2)
def _build_ecg_row_index(ecg_path: str) -> dict[str, tuple[int, int]]:
    parquet_file = _get_parquet_file(ecg_path)
    if "ecg_id" not in parquet_file.schema.names:
        return {}

    index: dict[str, tuple[int, int]] = {}
    for rg_idx in range(parquet_file.num_row_groups):
        ids = parquet_file.read_row_group(rg_idx, columns=["ecg_id"]).column(
            "ecg_id"
        ).to_pylist()
        for row_idx, ecg_id in enumerate(ids):
            ecg_id_str = str(ecg_id)
            if ecg_id_str not in index:
                index[ecg_id_str] = (rg_idx, row_idx)
    return index


@lru_cache(maxsize=128)
def _load_ecg_row_fast(ecg_id: str, ecg_path: str) -> pd.DataFrame:
    parquet_file = _get_parquet_file(ecg_path)
    index = _build_ecg_row_index(ecg_path)
    key = str(ecg_id)
    if key not in index:
        return pd.DataFrame(columns=parquet_file.schema.names)

    rg_idx, row_idx = index[key]
    seen = 0
    for batch in parquet_file.iter_batches(row_groups=[rg_idx], batch_size=256):
        if seen + batch.num_rows <= row_idx:
            seen += batch.num_rows
            continue
        local_row_idx = row_idx - seen
        row_df = batch.slice(local_row_idx, 1).to_pandas()
        if "ecg_id" in row_df.columns and not row_df.empty:
            row_df["ecg_id"] = row_df["ecg_id"].astype(str)
        return row_df

    return pd.DataFrame(columns=parquet_file.schema.names)


def load_ecg_full(ecg_path: str) -> pd.DataFrame:
    if os.path.exists(ecg_path):
        return pd.read_parquet(ecg_path)
    raise FileNotFoundError(f"Parquet file not found: '{ecg_path}'")


def load_ecg_rows_map(
    ecg_ids: list[str], ecg_path: str, batch_size: int = 32
) -> dict[str, pd.DataFrame]:
    if not os.path.exists(ecg_path):
        raise FileNotFoundError(f"Parquet file not found: '{ecg_path}'")
    if not ecg_ids:
        return {}

    target_ids = set(ecg_ids)
    result: dict[str, pd.DataFrame] = {}

    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(ecg_path)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        batch_df = batch.to_pandas()
        if "ecg_id" not in batch_df.columns:
            continue
        matched = batch_df[batch_df["ecg_id"].isin(target_ids)]
        if matched.empty:
            continue

        for _, row in matched.iterrows():
            row_df = row.to_frame().T
            ecg_id = str(row["ecg_id"])
            if ecg_id not in result:
                result[ecg_id] = row_df
        target_ids -= {str(v) for v in matched["ecg_id"].unique().tolist()}
        if not target_ids:
            break

    return result
