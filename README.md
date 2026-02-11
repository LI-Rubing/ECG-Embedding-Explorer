# ECG Embedding Explorer

An interactive Streamlit app for exploring ECG embeddings, visualizing 2D/3D projections, and inspecting corresponding ECG signals.

## 1. Requirements

Recommended: Python 3.10+.

Install dependencies:

```bash
pip install -r requirements.txt
```

Current dependency set:

- `numpy>=1.24`
- `pandas>=2.0`
- `plotly>=5.18`
- `pyarrow>=12.0`
- `scikit-learn>=1.2`
- `streamlit>=1.33`
- `umap-learn>=0.5.5`

## 2. Run

```bash
streamlit run ui/app.py
```

If your ECG data is in `pkl`, convert it to parquet first (recommended):

```bash
python scripts/convert_ecg_to_parquet.py
```

Strong recommendation:
- For ECG data, convert to `parquet` before importing into the app.
- `parquet` enables metadata reading, row-level lazy loading, and faster/stabler interaction on larger datasets.

Conversion command (explicit input/output):

```bash
ECG_PKL_PATH=data/df_ecg.pkl ECG_PARQUET_PATH=data/df_ecg.parquet python scripts/convert_ecg_to_parquet.py
```

## 3. Technical Choices

- **Language**: Python
- **UI Framework**: Streamlit
- **Data**: pandas / numpy
- **Dimensionality Reduction**: scikit-learn (PCA / t-SNE), umap-learn (UMAP)
- **Visualization**: Plotly

## 4. Data Schema Assumptions

Supports `parquet / csv / pkl`.

Example snapshot from a real run:

- `df_embeddings`: `rows=21799`, `cols=14`
- `df_ecg`: `rows=21799`, `cols=13`

### df_embeddings (required)

- Must include: `ecg_id`
- At least one embedding column, e.g.:
  - `mean_embedding`
  - `mean_global_embedding`
- Embedding entries are typically `np.ndarray` or `list/tuple` with consistent length (e.g., 1024).

Optional examples:
- `patient_id`, `device_group`, `sampling_rate`, `label`, `split`, etc.

Observed columns in the provided dataset:
- `mean_embedding`, `mean_global_embedding`, `ecg_id`, `patient_id`, `tor`, `sampling_rate`
- `leads_partition_name`, `leads_array_index`
- `discovery_tsne_raw_embedding_x`, `discovery_tsne_raw_embedding_y`
- `device_name`, `device_model`, `device_name_present`, `device_group`

Note:
- In this dataset, `discovery_tsne_raw_embedding_x/y` are fully missing (`all NaN`), so use computed PCA/t-SNE/UMAP instead of those precomputed columns.

### df_ecg (required)

- Must include: `ecg_id`
- ECG lead columns (all columns except `ecg_id`):
  - Each column stores a single-lead signal (e.g., length 5000)
  - Multiple leads are stored as separate columns per row

Important:
- Repeated lead column names are supported (e.g., multiple columns named `element`).
- The app treats all non-`ecg_id` columns as lead signals.

## 5. Features (Summary)

- Load embeddings and ECG data
- Validation (join key / missing IDs / embedding length consistency)
- PCA / t-SNE / UMAP projection
- Default 2D embedding view; optional 3D view (`Output dims = 3`)
- t-SNE metric selectable from dropdown (not limited to Euclidean)
- UMAP metric selectable from dropdown (no manual typing required)
- Plotly scatter with metadata hover and color mapping
- ECG panel linked to selected point / selected `ecg_id`
- `ecg_id Copy Assistant` for searching and copying long IDs from current plotted data

## 6. Interaction Notes

- On first load, the right ECG panel stays in a waiting state until an `ecg_id` is selected.
- In 3D mode, native Streamlit point-click callbacks can be inconsistent depending on environment.
- Reliable fallback: use `ecg_id Copy Assistant` or `Load ECG by ecg_id` in the right panel.
