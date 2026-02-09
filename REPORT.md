# ECG Embedding Explorer

## 1. Application Overview

This project delivers an interactive Streamlit application for exploring ECG embeddings and visualizing corresponding ECG signals. Users can load embeddings and ECG datasets, run dimensionality reduction (PCA / t-SNE / UMAP), and select points in a 2D scatter plot to view ECG waveforms. The UI supports filtering, coloring by metadata, and lead selection.

## 2. Features Implemented

- Load `df_embeddings` and `df_ecg` from `parquet / csv / pkl`
- Schema validation (join key checks, embedding length consistency, missing IDs)
- Dimensionality reduction (PCA / t-SNE / UMAP) with parameter controls
- 2D visualization with Plotly scatter + color mapping
- Click / selection to display ECG waveform and metadata
- Lead selection (hide/show)

## 3. Difficulties Encountered

- Some provided datasets contain precomputed 2D columns that are fully missing (all NaN), so the UI must detect and guide users to alternative columns or recompute projections.
- When parquet data is large, full missing-value statistics can be expensive; validation must balance robustness and runtime.
- Plotly selection events in Streamlit are not always stable for large point clouds, requiring sampling or fallback interaction.
- With limited local memory, loading large `pkl` files directly is costly; converting to `parquet` is more reliable in practice.
- To reduce memory pressure, ECG visualization uses lazy loading (load only after a point is selected), which can introduce noticeable latency on first draw.
- Under very large datasets, scatter rendering can become unstable or slow due to memory and browser limits.

## 4. Possible Improvements

- Add unit tests for critical functions (ID mapping, schema validation).
- Provide a real progress indicator or clearer computation feedback.
- Automatically detect valid precomputed 2D columns (avoid NaN-only columns).
- Improve click-to-ECG stability with more robust event handling.
- Add stronger memory safeguards (force parquet for large files, limit batch loading).
- Optimize ECG rendering (downsampling, segmented rendering, clearer lazy-loading indicators).
- Enforce stricter sampling limits or warnings for very large datasets.
