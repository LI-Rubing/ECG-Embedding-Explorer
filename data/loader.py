import os

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
        try:
            return pd.read_parquet(ecg_path, filters=[("ecg_id", "==", ecg_id)])
        except Exception:
            return pd.read_parquet(ecg_path)
    raise FileNotFoundError(f"Parquet file not found: '{ecg_path}'")
