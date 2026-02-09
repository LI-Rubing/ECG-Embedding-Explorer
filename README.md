# ECG Embedding Explorer

An interactive Streamlit app for exploring ECG embeddings, visualizing 2D projections, and inspecting corresponding ECG signals.

## 1. Requirements

Recommended: Python 3.10+.

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Run

```bash
streamlit run ui/app.py
```

If your ECG data is in `pkl`, convert it to parquet first (recommended):

```bash
python scripts/convert_ecg_to_parquet.py
```

## 3. Technical Choices

- **Language**: Python
- **UI Framework**: Streamlit
- **Data**: pandas / numpy
- **Dimensionality Reduction**: scikit-learn (PCA / t-SNE), umap-learn (UMAP)
- **Visualization**: Plotly

## 4. Data Schema Assumptions

Supports `parquet / csv / pkl`.

### df_embeddings (required)

- Must include: `ecg_id`
- At least one embedding column, e.g.:
  - `mean_embedding`
  - `mean_global_embedding`
- Embedding entries are typically `np.ndarray` or `list/tuple` with consistent length (e.g., 1024).

Optional examples:
- `patient_id`, `device_group`, `sampling_rate`, `label`, `split`, etc.

### df_ecg (required)

- Must include: `ecg_id`
- ECG lead columns (all columns except `ecg_id`):
  - Each column stores a single-lead signal (e.g., length 5000)
  - Multiple leads are stored as separate columns per row

## 5. Features (Summary)

- Load embeddings and ECG data
- Validation (join key / missing IDs / embedding length consistency)
- PCA / t-SNE / UMAP projection
- 2D scatter visualization with coloring
- Click to display ECG waveform
- Lead selection (hide/show)
