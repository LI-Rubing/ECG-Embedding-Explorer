import os

import pandas as pd


def main() -> None:
    input_path = os.environ.get("ECG_PKL_PATH", "data/df_ecg.pkl")
    output_path = os.environ.get("ECG_PARQUET_PATH", "data/df_ecg.parquet")
    compression = os.environ.get("ECG_PARQUET_COMPRESSION", "snappy")
    chunk_size_str = os.environ.get("ECG_PARQUET_CHUNK_SIZE", "2000")
    try:
        chunk_size = int(chunk_size_str)
    except ValueError:
        chunk_size = 2000

    print(f"[convert] reading: {input_path}")
    df = pd.read_pickle(input_path)
    print(f"[convert] loaded: rows={len(df)}, cols={list(df.columns)}")

    print(
        f"[convert] writing: {output_path} "
        f"(compression={compression}, chunk_size={chunk_size})"
    )
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'pyarrow' for parquet writing. "
            "Install it with: pip install pyarrow"
        ) from exc

    schema = None
    writer = None
    try:
        for start in range(0, len(df), chunk_size):
            chunk = df.iloc[start : start + chunk_size]
            table = pa.Table.from_pandas(chunk, schema=schema, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(
                    output_path, schema=schema, compression=compression
                )
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    print("[convert] done.")


if __name__ == "__main__":
    main()
