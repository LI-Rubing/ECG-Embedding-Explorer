# ECG Embedding Explorer

## 1. Project Overview

This project is a Streamlit-based interactive tool that links embedding exploration with ECG waveform inspection in one workflow.

Current implementation covers:
- Multi-format data loading (path input + file upload)
- Embedding reduction and 2D/3D visualization
- On-demand ECG retrieval after point/ID selection
- Large-sample safety controls (sampling, caching, fallback strategy, prefetch limits)

## 2. How to Run

Recommended: Python 3.10+.

```bash
pip install -r requirements.txt
streamlit run ui/app.py
```

ECG is strongly recommended in parquet format for row-level loading:

```bash
ECG_PKL_PATH=data/df_ecg.pkl ECG_PARQUET_PATH=data/df_ecg.parquet python scripts/convert_ecg_to_parquet.py
```

The conversion script supports:
- `ECG_PKL_PATH`
- `ECG_PARQUET_PATH`
- `ECG_PARQUET_COMPRESSION` (default: `snappy`)
- `ECG_PARQUET_CHUNK_SIZE` (default: `2000`)

## 3. Data Requirements and Validation

Supported formats: `parquet / csv / pkl`.

`df_embeddings` requirements:
- Must include `ecg_id`
- Must include at least one array-like embedding column (for example `mean_embedding`, `mean_global_embedding`)
- Embedding lengths should be consistent within the selected embedding column

`df_ecg` requirements:
- Must include `ecg_id`
- All non-`ecg_id` columns are treated as ECG lead waveform columns

Built-in validation includes:
- Join key existence checks
- Embedding-column existence and length consistency
- Optional Full Validation: missing-ID check, per-column missing counts, dataset summary

## 4. Key Implemented Features

- **Loading layer**
  - Sidebar path loading and optional file uploads
  - `df_ecg` can stay unloaded in memory while reading parquet metadata only
  - Optional full ECG table loading (high-RAM mode)

- **Reduction and visualization**
  - Three embedding sources:
    - Precomputed 2D columns (`tsne_mean_embedding_x/y`)
    - Manual selection of numeric X/Y columns
    - High-dimensional embeddings reduced via PCA / t-SNE / UMAP
  - Output dimension switch: 2D or 3D
  - PCA axis mapping via selectable PC indices (X/Y/Z)
  - Plotly `webgl` scatter with hover metadata and color mapping

- **Performance and stability controls**
  - `st.cache_data` used for:
    - Data loading
    - Reduction outputs
    - PCA component transforms
    - ECG row loading and batch prefetch
  - Large-sample protection:
    - `LARGE_EMBEDDING_THRESHOLD=10000`
    - t-SNE/UMAP defaults to PCA fallback on large samples unless explicitly overridden
  - ECG prefetch cap:
    - `MAX_ECG_PREFETCH_POINTS=1200`
    - Prefetch is skipped above the cap to keep the UI responsive

- **Linked ECG panel**
  - Point/selection in scatter updates ECG and metadata on the right panel
  - If ECG is not fully loaded, row-level parquet lookup is used by `ecg_id`
  - Includes `ecg_id Copy Assistant` (substring search + copy)
  - Includes manual `Load ECG by ecg_id` actions (typed ID or dropdown)

## 5. Current Limitations

- Streamlit 3D single-click callbacks can still be unstable; manual `ecg_id` loading is kept as a reliable fallback.
- Click interaction is less stable on low-memory machines at higher point counts; warning/guardrails are present.
- Full missing-value validation can still be expensive on very large datasets.
- There is no comprehensive automated test suite yet; quality is currently verified mainly through interactive checks.

## 6. Suggested Next Steps

- Add unit tests for `data/loader.py` and `embeddings/reducer.py` (edge cases, error paths, version-compat params).
- Add performance benchmarks for PCA/t-SNE/UMAP across dataset scales.
- Improve user-facing progress feedback for long-running validation/reduction tasks.
