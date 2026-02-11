import os
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
import plotly.graph_objects as go

# Allow running this app from repo root or ui/ directory.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from data.loader import (
    load_dataframe,
    load_dataframe_from_upload,
    load_ecg_dataframe,
    load_ecg_full,
    load_ecg_metadata,
    load_ecg_row,
    load_ecg_rows_map,
)
from data.validator import (
    detect_array_columns,
    embedding_length,
    summarize_df,
    validate_embedding_length,
)
from embeddings.reducer import reduce_embeddings
from visualization.ecg_plot import plot_ecg
from visualization.embedding_plot import get_hover_cols, plot_embedding_scatter


EMBEDDING_COLS = ["mean_embedding", "mean_global_embedding"]
PRECOMPUTED_2D = ("tsne_mean_embedding_x", "tsne_mean_embedding_y")
MAX_ECG_PREFETCH_POINTS = 1200
LARGE_EMBEDDING_THRESHOLD = 10000


@st.cache_data(show_spinner=False)
def _cached_load_dataframe(path: str) -> pd.DataFrame:
    return load_dataframe(path)


@st.cache_data(show_spinner=False)
def _cached_load_ecg_full(path: str) -> pd.DataFrame:
    return load_ecg_full(path)


def _cached_load_ecg_row(path: str, ecg_id: str) -> pd.DataFrame:
    return load_ecg_row(ecg_id, path)


@st.cache_data(show_spinner=False)
def _cached_prefetch_ecg_rows(path: str, ecg_ids: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    return load_ecg_rows_map(list(ecg_ids), path)


@st.cache_data(show_spinner=False)
def _cached_reduce(
    X: np.ndarray, method: str, params: dict[str, Any]
) -> np.ndarray:
    return reduce_embeddings(X, method=method, params=params)


@st.cache_data(show_spinner=False)
def _cached_pca_components(
    X: np.ndarray, n_components: int, whiten: bool
) -> np.ndarray:
    from sklearn.decomposition import PCA

    model = PCA(n_components=int(n_components), whiten=bool(whiten))
    return model.fit_transform(X)


def _load_dataframe_from_upload(uploaded_file) -> pd.DataFrame:
    return load_dataframe_from_upload(uploaded_file)


def _get_session_ecg_row(ecg_id: str) -> pd.DataFrame | None:
    cache = st.session_state.setdefault("ecg_row_cache", {})
    return cache.get(str(ecg_id))


def _set_session_ecg_row(ecg_id: str, row_df: pd.DataFrame, max_items: int = 256) -> None:
    cache = st.session_state.setdefault("ecg_row_cache", {})
    key = str(ecg_id)
    if key in cache:
        cache.pop(key, None)
    cache[key] = row_df
    if len(cache) > max_items:
        oldest_key = next(iter(cache))
        cache.pop(oldest_key, None)
    st.session_state["ecg_row_cache"] = cache


def _apply_filter(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if not column or column == "None":
        return df

    series = df[column]
    if is_numeric_dtype(series) and not is_bool_dtype(series):
        min_val = float(series.min(skipna=True))
        max_val = float(series.max(skipna=True))
        if min_val == max_val:
            st.info("Selected numeric column has a single unique value.")
            return df
        selected = st.sidebar.slider(
            f"Range for {column}",
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val),
        )
        return df[(series >= selected[0]) & (series <= selected[1])]

    if is_datetime64_any_dtype(series):
        min_val = series.min()
        max_val = series.max()
        selected = st.sidebar.date_input(
            f"Date range for {column}", value=(min_val, max_val)
        )
        if isinstance(selected, tuple) and len(selected) == 2:
            start, end = selected
            return df[(series >= pd.to_datetime(start)) & (series <= pd.to_datetime(end))]
        return df

    options = series.dropna().unique().tolist()
    if len(options) > 200:
        options = series.value_counts().head(200).index.tolist()
        st.sidebar.info("Showing top 200 categories by frequency.")
    selected = st.sidebar.multiselect(
        f"Values for {column}", options=options, default=options
    )
    if not selected:
        return df.iloc[0:0]
    return df[series.isin(selected)]


def _extract_point_index(point: dict[str, Any]) -> int | None:
    for key in ("pointIndex", "point_index", "pointNumber", "point_number", "point_inds"):
        if key not in point:
            continue
        value = point[key]
        if value is None:
            continue
        if isinstance(value, (list, tuple, np.ndarray)):
            if not value:
                continue
            value = value[-1]
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _extract_ecg_id_from_point(point: dict[str, Any]) -> str | None:
    ecg_id = point.get("ecg_id")
    if ecg_id is not None:
        return str(ecg_id)
    custom_val = point.get("customdata")
    if isinstance(custom_val, np.ndarray):
        custom_val = custom_val.tolist()
    if isinstance(custom_val, (list, tuple)) and custom_val:
        return str(custom_val[0])
    if custom_val is not None:
        return str(custom_val)
    return None


def _normalize_ecg_id(value: object) -> str:
    return str(value).strip()


def main() -> None:
    st.set_page_config(page_title="ECG Embedding Explorer", layout="wide")
    st.title("ECG Embedding Explorer")
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] .block-container {
            padding-top: 0.6rem;
            padding-bottom: 0.6rem;
            padding-left: 0.7rem;
            padding-right: 0.7rem;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            margin-top: 0.35rem;
            margin-bottom: 0.2rem;
            font-size: 1.0rem;
            line-height: 1.2;
        }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown p {
            font-size: 0.82rem !important;
            line-height: 1.2 !important;
            margin-bottom: 0.1rem !important;
        }
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
            margin-bottom: 0.1rem !important;
        }
        [data-testid="stSidebar"] [data-testid="stTextInput"] input,
        [data-testid="stSidebar"] [data-testid="stNumberInput"] input,
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-testid="stMultiSelect"] > div {
            min-height: 1.9rem !important;
            font-size: 0.82rem !important;
        }
        [data-testid="stSidebar"] [data-testid="stButton"] button {
            min-height: 1.85rem !important;
            font-size: 0.82rem !important;
            padding-top: 0.2rem !important;
            padding-bottom: 0.2rem !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] {
            margin-top: 0.05rem !important;
            margin-bottom: 0.08rem !important;
        }
        [data-testid="stSidebar"] .stSlider,
        [data-testid="stSidebar"] [data-testid="stCheckbox"],
        [data-testid="stSidebar"] [data-testid="stSelectbox"],
        [data-testid="stSidebar"] [data-testid="stNumberInput"],
        [data-testid="stSidebar"] [data-testid="stTextInput"],
        [data-testid="stSidebar"] [data-testid="stMultiSelect"] {
            margin-top: 0.1rem !important;
            margin-bottom: 0.22rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    emb_upload = None
    ecg_upload = None
    with st.sidebar.expander("Data Loading", expanded=False):
        with st.expander("Optional uploads", expanded=False):
            emb_upload = st.file_uploader(
                "Upload df_embeddings (parquet/csv/pkl)", type=["parquet", "csv", "pkl"]
            )
            ecg_upload = st.file_uploader(
                "Upload df_ecg (parquet/csv/pkl)", type=["parquet", "csv", "pkl"]
            )
        emb_path = st.text_input(
            "df_embeddings path", value="data/df_embeddings.pkl"
        )
        ecg_path = st.text_input("df_ecg path", value="data/df_ecg.parquet")
        force_full_ecg_load = st.checkbox(
            "Load full ECG into memory (high RAM)", value=False
        )
        if st.button("Load Data"):
            st.session_state.pop("df_embeddings", None)
            st.session_state.pop("df_ecg", None)
            st.session_state.pop("ecg_row_cache", None)
            st.session_state.pop("current_ecg_id", None)

    if "df_embeddings" not in st.session_state:
        if emb_upload is not None:
            st.session_state["df_embeddings"] = _load_dataframe_from_upload(
                emb_upload
            )
        else:
            st.session_state["df_embeddings"] = _cached_load_dataframe(emb_path)

    df_embeddings = st.session_state["df_embeddings"]
    if "df_ecg" not in st.session_state:
        df_ecg_loaded = load_ecg_dataframe(ecg_upload, ecg_path)
        if df_ecg_loaded is not None:
            st.session_state["df_ecg"] = df_ecg_loaded
        else:
            if (
                force_full_ecg_load
                and ecg_path
                and ecg_path.lower().endswith(".parquet")
            ):
                progress = st.progress(0, text="Loading full ECG parquet...")
                try:
                    progress.progress(10)
                    df_ecg_full = _cached_load_ecg_full(ecg_path)
                    progress.progress(100)
                finally:
                    progress.empty()
                st.session_state["df_ecg"] = df_ecg_full
            else:
                st.session_state["df_ecg"] = None

    st.subheader("Dataset Overview")
    st.write(
        f"df_embeddings: rows={len(df_embeddings)}, cols={len(df_embeddings.columns)}"
    )
    with st.expander("df_embeddings columns", expanded=False):
        st.write(list(df_embeddings.columns))
    df_ecg = st.session_state.get("df_ecg")
    if df_ecg is not None:
        st.write(f"df_ecg: rows={len(df_ecg)}, cols={len(df_ecg.columns)}")
        with st.expander("df_ecg columns", expanded=False):
            st.write(list(df_ecg.columns))
    else:
        try:
            ecg_columns, ecg_rows = load_ecg_metadata(ecg_path)
            if ecg_rows is not None:
                st.write(f"df_ecg: rows={ecg_rows}, cols={len(ecg_columns)}")
            with st.expander("df_ecg columns", expanded=False):
                st.write(list(ecg_columns))
        except Exception as exc:
            st.warning(str(exc))

    array_cols = detect_array_columns(df_embeddings)
    embedding_options = [
        col for col in EMBEDDING_COLS if col in df_embeddings.columns
    ]
    embedding_options += [col for col in array_cols if col not in embedding_options]

    numeric_cols = [
        col
        for col in df_embeddings.columns
        if is_numeric_dtype(df_embeddings[col]) and not is_bool_dtype(df_embeddings[col])
    ]
    has_precomputed = (
        PRECOMPUTED_2D[0] in df_embeddings.columns
        and PRECOMPUTED_2D[1] in df_embeddings.columns
    )
    if has_precomputed:
        embedding_source_options = ["precomputed_tsne_xy"] + embedding_options
    else:
        embedding_source_options = embedding_options
    embedding_source_options += ["select_xy_columns"]

    if not embedding_source_options:
        st.error("No embedding columns found in df_embeddings.")
        st.stop()

    embedding_section = st.sidebar.expander("Embedding Selection", expanded=False)
    embedding_source = embedding_section.selectbox(
        "Embedding source", embedding_source_options
    )
    if embedding_source == "precomputed_tsne_xy":
        embedding_col = None
        xy_mode = "precomputed"
    elif embedding_source == "select_xy_columns":
        embedding_col = None
        xy_mode = "manual"
    else:
        embedding_col = embedding_source
        xy_mode = "embedding"

    xy_x_col = None
    xy_y_col = None
    if xy_mode == "manual":
        if len(numeric_cols) < 2:
            st.error("Need at least two numeric columns to select X and Y.")
            st.stop()
        default_x = None
        default_y = None
        if "discovery_tsne_raw_embedding_x" in df_embeddings.columns:
            default_x = "discovery_tsne_raw_embedding_x"
        if "discovery_tsne_raw_embedding_y" in df_embeddings.columns:
            default_y = "discovery_tsne_raw_embedding_y"
        x_index = numeric_cols.index(default_x) if default_x in numeric_cols else 0
        xy_x_col = embedding_section.selectbox("X column", numeric_cols, index=x_index)
        y_candidates = [c for c in numeric_cols if c != xy_x_col]
        if default_y in y_candidates:
            y_index = y_candidates.index(default_y)
        else:
            y_index = 0 if y_candidates else 0
        xy_y_col = embedding_section.selectbox("Y column", y_candidates, index=y_index)

    validation_embedding_col = embedding_col or (
        embedding_options[0] if embedding_options else None
    )
    if validation_embedding_col is None:
        st.error("No embedding column available for validation.")
        st.stop()

    errors = []
    warnings: list[str] = []
    if "ecg_id" not in df_embeddings.columns:
        errors.append("Join key 'ecg_id' is missing from df_embeddings.")
    if df_ecg is not None:
        if "ecg_id" not in df_ecg.columns:
            errors.append("Join key 'ecg_id' is missing from df_ecg.")
    else:
        try:
            ecg_columns, _ = load_ecg_metadata(ecg_path)
            if "ecg_id" not in ecg_columns:
                errors.append("Join key 'ecg_id' is missing from df_ecg.")
        except Exception as exc:
            warnings.append(f"df_ecg columns check failed: {exc}")
    ok = len(errors) == 0
    if embedding_col:
        errors += validate_embedding_length(df_embeddings, embedding_col)
    if embedding_source == "precomputed_tsne_xy":
        if (
            PRECOMPUTED_2D[0] not in df_embeddings.columns
            or PRECOMPUTED_2D[1] not in df_embeddings.columns
        ):
            errors.append("Precomputed 2D columns are missing from df_embeddings.")

    if errors:
        st.error("Schema validation failed:")
        for message in errors:
            st.write(f"- {message}")
        st.stop()
    if warnings:
        st.warning("Schema checks produced warnings:")
        for message in warnings:
            st.write(f"- {message}")

    with st.expander("2D vs High-D Embeddings", expanded=False):
        if xy_mode == "precomputed":
            st.info(
                "Selected precomputed 2D columns (x, y). "
                "Visualization will be displayed directly without dimensionality reduction."
            )
        elif xy_mode == "manual":
            st.info(
                "Selected custom 2D columns (x, y). "
                "Visualization will be displayed directly without dimensionality reduction."
            )
        else:
            lengths = []
            for value in df_embeddings[embedding_col]:
                length = embedding_length(value)
                if length is not None:
                    lengths.append(int(length))
            n_2d = sum(1 for l in lengths if l == 2)
            n_other = sum(1 for l in lengths if l != 2)
            if n_other == 0 and n_2d > 0:
                st.info(
                    "All embeddings are 2D. Visualization will be displayed directly."
                )
            else:
                st.info(
                    "Embeddings are not 2D. Dimensionality reduction will be applied."
                )
            if lengths:
                unique_lengths = sorted(set(lengths))
                st.write(f"Embedding lengths detected: {unique_lengths}")
            st.write(f"Count 2D: {n_2d}, Count non-2D: {n_other}")

    if len(embedding_options) > 1:
        st.info(
            "Multiple embedding columns detected: "
            + ", ".join(embedding_options)
        )

    with st.sidebar.expander("Validation", expanded=False):
        run_full_validation = st.button("Run Full Validation")
    if run_full_validation:
        st.info("Full validation started...")
        st.write(
            "Running checks: join keys, missing IDs, embedding dimensions, multiple embedding columns."
        )
        validation_messages: list[str] = []
        if "ecg_id" not in df_embeddings.columns:
            validation_messages.append("Missing join key: df_embeddings.ecg_id")
        else:
            validation_messages.append("Join key present: df_embeddings.ecg_id")

        if df_ecg is not None:
            if "ecg_id" not in df_ecg.columns:
                validation_messages.append("Missing join key: df_ecg.ecg_id")
            else:
                validation_messages.append("Join key present: df_ecg.ecg_id")
        else:
            try:
                ecg_columns, _ = load_ecg_metadata(ecg_path)
                if "ecg_id" not in ecg_columns:
                    validation_messages.append("Missing join key: df_ecg.ecg_id")
                else:
                    validation_messages.append("Join key present: df_ecg.ecg_id")
            except Exception as exc:
                validation_messages.append(f"df_ecg columns check failed: {exc}")

        # Embedding dimension consistency check
        for col in embedding_options:
            if col not in df_embeddings.columns:
                validation_messages.append(
                    f"{col}: skipped (column not found in df_embeddings)."
                )
                continue
            errors = validate_embedding_length(df_embeddings, col)
            if errors:
                validation_messages.append(f"{col}: " + "; ".join(errors))
            else:
                lengths = []
                for value in df_embeddings[col]:
                    length = embedding_length(value)
                    if length is not None:
                        lengths.append(int(length))
                if lengths:
                    unique_lengths = sorted(set(lengths))
                    if len(unique_lengths) == 1:
                        validation_messages.append(
                            f"{col}: embedding length consistent (length={unique_lengths[0]})."
                        )
                    else:
                        validation_messages.append(
                            f"{col}: embedding lengths detected {unique_lengths}."
                        )
                else:
                    validation_messages.append(
                        f"{col}: embedding length consistent (no array-like entries found)."
                    )

        # Multiple embedding columns
        if len(embedding_options) > 1:
            validation_messages.append(
                "Multiple embedding columns detected: " + ", ".join(embedding_options)
            )
        else:
            validation_messages.append("Single embedding column detected.")

        # Missing IDs check
        try:
            if "ecg_id" in df_embeddings.columns:
                if df_ecg is not None:
                    if "ecg_id" in df_ecg.columns:
                        missing_ids = set(df_embeddings["ecg_id"]) - set(
                            df_ecg["ecg_id"]
                        )
                        if missing_ids:
                            validation_messages.append(
                                f"Missing IDs in df_ecg: {len(missing_ids)}"
                            )
                        else:
                            validation_messages.append(
                                "All df_embeddings.ecg_id found in df_ecg."
                            )
                    else:
                        validation_messages.append(
                            "Missing IDs check skipped: df_ecg.ecg_id missing."
                        )
                else:
                    import pyarrow.parquet as pq

                    pf = pq.ParquetFile(ecg_path)
                    ecg_ids = set()
                    for batch in pf.iter_batches(columns=["ecg_id"]):
                        ecg_ids.update(batch.column(0).to_pylist())
                    missing_ids = set(df_embeddings["ecg_id"]) - ecg_ids
                    if missing_ids:
                        validation_messages.append(
                            f"Missing IDs in df_ecg: {len(missing_ids)}"
                        )
                    else:
                        validation_messages.append(
                            "All df_embeddings.ecg_id found in df_ecg."
                        )
            else:
                validation_messages.append(
                    "Missing IDs check skipped: df_embeddings.ecg_id missing."
                )
        except Exception as exc:
            validation_messages.append(f"Missing IDs check failed: {exc}")

        with st.expander("Validation Results", expanded=True):
            for msg in validation_messages:
                st.write(f"- {msg}")

        with st.expander("Dataset Summary (Types & Missing)", expanded=False):
            st.write(
                f"df_embeddings: rows={len(df_embeddings)}, cols={len(df_embeddings.columns)}"
            )
            st.dataframe(summarize_df(df_embeddings), use_container_width=True)

            if df_ecg is not None:
                st.write(
                    f"df_ecg: rows={len(df_ecg)}, cols={len(df_ecg.columns)}"
                )
                st.dataframe(summarize_df(df_ecg), use_container_width=True)
            else:
                try:
                    ecg_columns, ecg_rows = load_ecg_metadata(ecg_path)
                    st.write(
                        f"df_ecg: rows={ecg_rows if ecg_rows is not None else 'unknown'}, "
                        f"cols={len(ecg_columns)}"
                    )
                    st.info(
                        "df_ecg not loaded; missing counts require loading the dataset."
                    )
                except Exception as exc:
                    st.warning(f"df_ecg summary failed: {exc}")
        st.success("Full validation completed.")

    with st.sidebar.expander("Filtering", expanded=False):
        filterable_cols = [
            col for col in df_embeddings.columns if col not in array_cols
        ]
        filter_column = st.selectbox(
            "Filter column", ["None"] + filterable_cols
        )
    df_filtered = _apply_filter(df_embeddings, filter_column)
    if df_ecg is not None:
        ecg_array_cols = detect_array_columns(df_ecg)
        ecg_filterable_cols = [
            col
            for col in df_ecg.columns
            if col not in ecg_array_cols and col != "ecg_id"
        ]
        if ecg_filterable_cols:
            with st.sidebar.expander("ECG Filtering", expanded=False):
                ecg_filter_col = st.selectbox(
                    "ECG filter column", ["None"] + ecg_filterable_cols
                )
            df_ecg_filtered = _apply_filter(df_ecg, ecg_filter_col)
            if not df_ecg_filtered.empty and "ecg_id" in df_ecg_filtered.columns:
                df_filtered = df_filtered[
                    df_filtered["ecg_id"].isin(df_ecg_filtered["ecg_id"])
                ]
        else:
            st.sidebar.info("No ECG metadata columns available for filtering.")
    if df_filtered.empty:
        st.warning("Filtered dataframe is empty.")
        st.stop()

    with st.sidebar.expander("Sampling", expanded=False):
        sample_n = st.number_input(
            "Sample size (0=all)",
            min_value=0,
            max_value=50000,
            value=200,
        )
        random_state = st.number_input(
            "Random state", min_value=0, max_value=10_000, value=0
        )
    if sample_n and len(df_filtered) > sample_n:
        df_filtered = df_filtered.sample(
            n=int(sample_n), random_state=int(random_state)
        ).reset_index(drop=True)
    n_points = len(df_filtered)

    summary_filtered = (
        f"Filtered df_embeddings: rows={len(df_filtered)}; "
        "ECG will be loaded on demand."
    )

    with st.sidebar.expander("Coloring", expanded=False):
        colorable_cols = [
            col for col in df_filtered.columns if col not in array_cols
        ]
        color_by = st.selectbox("Color by", ["None"] + colorable_cols)
        point_size = st.slider(
            "Point size", min_value=2.0, max_value=12.0, value=6.0, step=0.5
        )
        point_opacity = st.slider(
            "Point opacity", min_value=0.2, max_value=1.0, value=0.8, step=0.05
        )

    with st.sidebar.expander("Interaction", expanded=False):
        interaction_mode = st.selectbox(
            "Selection mode", ["click (experimental)", "manual"], index=0
        )
        prefetch_click_ecg = st.checkbox(
            "Prefetch ECG for plotted points", value=True
        )
    if interaction_mode.startswith("click") and sample_n and sample_n > 300:
        st.sidebar.warning(
            "Click mode is unstable above ~300 points on low-memory machines. "
            "Capping sample size to 300."
        )
        sample_n = 300

    pca_component_count = 2
    if xy_mode == "embedding" and embedding_col in df_filtered.columns:
        for value in df_filtered[embedding_col]:
            length = embedding_length(value)
            if length is not None and int(length) >= 2:
                pca_component_count = int(length)
                break
    pca_x_pc = 1
    pca_y_pc = 2
    pca_z_pc = 3
    output_dims = 2

    with st.sidebar.expander("Dimensionality Reduction", expanded=False):
        if xy_mode in {"precomputed", "manual"}:
            method = "precomputed"
            params = {}
        else:
            method = st.selectbox("Method", ["pca", "tsne", "umap"], key="dr_method")
            output_dims = int(
                st.selectbox(
                    "Output dims",
                    options=[2, 3],
                    index=0,
                    key="dr_output_dims",
                )
            )
            params: dict[str, Any] = {}
            if method == "pca":
                params["whiten"] = st.checkbox("PCA whiten", value=False)
                pca_x_pc = int(
                    st.number_input(
                        "X PC",
                        min_value=1,
                        max_value=max(2, pca_component_count),
                        value=1,
                        step=1,
                        key="pca_x_pc",
                    )
                )
                pca_y_pc = int(
                    st.number_input(
                        "Y PC",
                        min_value=1,
                        max_value=max(2, pca_component_count),
                        value=2 if pca_component_count >= 2 else 1,
                        step=1,
                        key="pca_y_pc",
                    )
                )
                if output_dims == 3:
                    if pca_component_count < 3:
                        st.warning(
                            "Current embedding has fewer than 3 dimensions for PCA component selection."
                        )
                    pca_z_pc = int(
                        st.number_input(
                            "Z PC",
                            min_value=1,
                            max_value=max(3, pca_component_count),
                            value=3 if pca_component_count >= 3 else 1,
                            step=1,
                            key="pca_z_pc",
                        )
                    )
            elif method == "tsne":
                params["perplexity"] = st.number_input(
                    "t-SNE perplexity", min_value=2.0, max_value=100.0, value=30.0
                )
                params["learning_rate"] = st.number_input(
                    "t-SNE learning_rate", min_value=10.0, max_value=1000.0, value=200.0
                )
                params["n_iter"] = st.number_input(
                    "t-SNE n_iter", min_value=250, max_value=5000, value=1000
                )
                params["metric"] = st.selectbox(
                    "t-SNE metric",
                    options=[
                        "euclidean",
                        "cosine",
                        "manhattan",
                        "chebyshev",
                        "minkowski",
                        "correlation",
                    ],
                    index=0,
                )
            else:
                params["n_neighbors"] = st.number_input(
                    "UMAP n_neighbors", min_value=2, max_value=100, value=15
                )
                params["min_dist"] = st.number_input(
                    "UMAP min_dist", min_value=0.0, max_value=1.0, value=0.1
                )
                params["metric"] = st.selectbox(
                    "UMAP metric",
                    options=[
                        "euclidean",
                        "cosine",
                        "manhattan",
                        "chebyshev",
                        "minkowski",
                        "correlation",
                        "canberra",
                        "braycurtis",
                    ],
                    index=0,
                )

        if xy_mode == "embedding" and n_points >= LARGE_EMBEDDING_THRESHOLD:
            st.warning(
                f"Large sample ({n_points} points): auto-enabling fast settings for stability."
            )
        allow_slow_large_reduction = st.checkbox(
            "Allow slow t-SNE/UMAP on large sample", value=False
        )
        if method == "pca":
            selected_pcs = [pca_x_pc, pca_y_pc] + ([pca_z_pc] if output_dims == 3 else [])
            if len(set(selected_pcs)) != len(selected_pcs):
                st.warning("Selected PCs should be different.")
    compute_now = st.sidebar.button("Compute Embedding")

    def _current_config_key() -> tuple[object, ...]:
        return (
            embedding_source,
            embedding_col,
            xy_mode,
            xy_x_col,
            xy_y_col,
            method,
            tuple(sorted(params.items())),
            output_dims,
            pca_x_pc,
            pca_y_pc,
            pca_z_pc,
            filter_column,
            sample_n,
            random_state,
            len(df_filtered),
        )

    if compute_now or "embedding_cache" not in st.session_state:
        with st.spinner("Computing embeddings..."):
            start = time.perf_counter()
            if xy_mode == "precomputed":
                X_2d = df_filtered[[PRECOMPUTED_2D[0], PRECOMPUTED_2D[1]]].to_numpy()
            elif xy_mode == "manual":
                if xy_x_col is None or xy_y_col is None:
                    st.error("X/Y columns are not set for manual 2D mode.")
                    st.stop()
                xy_df = df_filtered[[xy_x_col, xy_y_col]].copy()
                # Ensure numeric for plotting.
                for col in (xy_x_col, xy_y_col):
                    if not is_numeric_dtype(xy_df[col]):
                        xy_df[col] = pd.to_numeric(xy_df[col], errors="coerce")
                if xy_df[[xy_x_col, xy_y_col]].isna().all().all():
                    st.error(
                        "Selected X/Y columns contain no numeric values to plot."
                    )
                    st.stop()
                X_2d = xy_df[[xy_x_col, xy_y_col]].to_numpy()
            else:
                embeddings_matrix = np.stack(df_filtered[embedding_col].to_numpy())
                # If embeddings already match requested output dims, skip reduction.
                if (
                    embeddings_matrix.ndim == 2
                    and embeddings_matrix.shape[1] == int(output_dims)
                ):
                    X_2d = embeddings_matrix
                elif method == "pca":
                    selected_pcs = [int(pca_x_pc), int(pca_y_pc)] + (
                        [int(pca_z_pc)] if int(output_dims) == 3 else []
                    )
                    if len(set(selected_pcs)) != len(selected_pcs):
                        st.error("Selected PCs must be different.")
                        st.stop()
                    max_pc = max(selected_pcs)
                    if max_pc > embeddings_matrix.shape[1]:
                        st.error(
                            f"Selected PC index exceeds embedding dimension ({embeddings_matrix.shape[1]})."
                        )
                        st.stop()
                    X_pca = _cached_pca_components(
                        embeddings_matrix,
                        n_components=max_pc,
                        whiten=bool(params.get("whiten", False)),
                    )
                    selected_idx = [pc - 1 for pc in selected_pcs]
                    X_2d = X_pca[:, selected_idx]
                else:
                    try:
                        runtime_params = dict(params)
                        runtime_method = method
                        runtime_params["n_components"] = int(output_dims)
                        if (
                            n_points >= LARGE_EMBEDDING_THRESHOLD
                            and method in {"tsne", "umap"}
                            and not allow_slow_large_reduction
                        ):
                            runtime_method = "pca"
                            runtime_params = {}
                            st.info(
                                "Large sample detected: using PCA for speed. "
                                "Enable 'Allow slow t-SNE/UMAP on large sample' to override."
                            )
                        if n_points >= LARGE_EMBEDDING_THRESHOLD and runtime_method == "tsne":
                            runtime_params.setdefault("n_iter", 700)
                            runtime_params.setdefault("init", "pca")
                            runtime_params.setdefault("method", "barnes_hut")
                            runtime_params.setdefault("angle", 0.5)
                        if n_points >= LARGE_EMBEDDING_THRESHOLD and runtime_method == "umap":
                            runtime_params.setdefault("low_memory", True)
                            runtime_params.setdefault("n_epochs", 200)
                        X_2d = _cached_reduce(
                            embeddings_matrix, runtime_method, runtime_params
                        )
                    except Exception as exc:
                        st.error(str(exc))
                        st.stop()

            reduction_time = time.perf_counter() - start
        base_cols = [col for col in df_filtered.columns if col not in array_cols]
        if "ecg_id" in df_filtered.columns and "ecg_id" not in base_cols:
            base_cols.append("ecg_id")
        df_plot_base = df_filtered[base_cols].reset_index(drop=True).copy()
        st.session_state["embedding_cache"] = {
            "key": _current_config_key(),
            "X_2d": X_2d,
            "df_plot_base": df_plot_base,
            "reduction_time": reduction_time,
            "method": method,
            "params": params,
        }
        st.success("Embedding cache ready.")
    else:
        cache = st.session_state["embedding_cache"]
        if cache["key"] != _current_config_key():
            st.info("Parameters changed. Click 'Compute Embedding' to update.")
        X_2d = cache["X_2d"]
        df_plot_base = cache["df_plot_base"].copy()
        reduction_time = cache["reduction_time"]
        method = cache["method"]
        params = cache["params"]

    with st.expander("Computation Summary", expanded=False):
        st.write(summary_filtered)
        st.write(f"Dimensionality reduction time: {reduction_time:.4f}s")
    if embedding_source != "precomputed_tsne_xy" and method == "pca":
        with st.expander("PCA Details", expanded=False):
            show_pca_details = st.checkbox("Show PCA details", value=False)
            if not show_pca_details:
                st.info("Enable to compute PCA loadings and variance ratio.")
            else:
                try:
                    from sklearn.decomposition import PCA
                except Exception:
                    st.warning(
                        "PCA details require scikit-learn. "
                        "Install it with: pip install scikit-learn"
                    )
                else:
                    embeddings_matrix = np.stack(df_filtered[embedding_col].to_numpy())
                    selected_pcs = sorted(
                        {int(pca_x_pc), int(pca_y_pc)}
                        | ({int(pca_z_pc)} if int(output_dims) == 3 else set())
                    )
                    max_selected_pc = max(selected_pcs)
                    pca_model = PCA(
                        n_components=max_selected_pc, whiten=params.get("whiten", False)
                    )
                    pca_model.fit(embeddings_matrix)
                    var_ratio = pca_model.explained_variance_ratio_
                    var_labels = [f"PC{pc}" for pc in selected_pcs]
                    var_values = [var_ratio[pc - 1] for pc in selected_pcs]
                    fig_var = go.Figure(
                        data=[
                            go.Bar(
                                x=var_labels,
                                y=var_values,
                                text=[f"{v:.3f}" for v in var_values],
                                textposition="auto",
                            )
                        ]
                    )
                    fig_var.update_layout(
                        title="Explained Variance Ratio",
                        yaxis_title="Variance Ratio",
                        xaxis_title="Component",
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    st.plotly_chart(fig_var, use_container_width=True)

                    # Cumulative explained variance (Top K PCs)
                    top_k = st.number_input(
                        "Top K PCs (cumulative variance)",
                        min_value=2,
                        max_value=50,
                        value=10,
                    )
                    pca_full = PCA(
                        n_components=min(int(top_k), embeddings_matrix.shape[1]),
                        whiten=params.get("whiten", False),
                    )
                    pca_full.fit(embeddings_matrix)
                    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
                    fig_cum = go.Figure(
                        data=[
                            go.Scatter(
                                x=list(range(1, len(cum_var) + 1)),
                                y=cum_var,
                                mode="lines+markers",
                            )
                        ]
                    )
                    fig_cum.update_layout(
                        title="Cumulative Explained Variance (Top K PCs)",
                        xaxis_title="Number of PCs",
                        yaxis_title="Cumulative Variance",
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    st.plotly_chart(fig_cum, use_container_width=True)

                    # Top K PCs summary table
                    st.subheader("Top K PCs Summary")
                    pc_labels = [f"PC{i}" for i in range(1, len(cum_var) + 1)]
                    pc_var = pca_full.explained_variance_ratio_
                    summary_df = pd.DataFrame(
                        {
                            "pc": pc_labels,
                            "variance_ratio": pc_var,
                            "cumulative_variance": cum_var,
                        }
                    )
                    st.dataframe(summary_df)

                    def _top_loadings(pc_idx_1based: int, top_k: int = 10):
                        comp = pca_model.components_[pc_idx_1based - 1]
                        idx = np.argsort(np.abs(comp))[::-1][:top_k]
                        return pd.DataFrame(
                            {
                                "feature_index": idx,
                                "loading": comp[idx],
                                "abs_loading": np.abs(comp[idx]),
                            }
                        )

                    for pc in [int(pca_x_pc), int(pca_y_pc)] + (
                        [int(pca_z_pc)] if int(output_dims) == 3 else []
                    ):
                        st.write(f"Top loadings for PC{pc}:")
                        st.dataframe(_top_loadings(pc))

    if "df_plot_base" not in locals():
        base_cols = [col for col in df_filtered.columns if col not in array_cols]
        if "ecg_id" in df_filtered.columns and "ecg_id" not in base_cols:
            base_cols.append("ecg_id")
        df_plot_base = df_filtered[base_cols].reset_index(drop=True).copy()
    df_plot = df_plot_base.copy()
    df_plot["x"] = X_2d[:, 0]
    df_plot["y"] = X_2d[:, 1]
    if X_2d.shape[1] >= 3:
        df_plot["z"] = X_2d[:, 2]

    if (
        prefetch_click_ecg
        and len(df_plot) <= MAX_ECG_PREFETCH_POINTS
        and df_ecg is None
        and ecg_path
        and ecg_path.lower().endswith(".parquet")
    ):
        cache = st.session_state.get("embedding_cache", {})
        prefetch_ids = tuple(str(v) for v in df_plot["ecg_id"].tolist())
        if cache.get("prefetch_ids") != prefetch_ids:
            with st.spinner("Prefetching ECG rows for plotted points..."):
                cache["prefetched_ecg_rows"] = _cached_prefetch_ecg_rows(
                    ecg_path, prefetch_ids
                )
                cache["prefetch_ids"] = prefetch_ids
                st.session_state["embedding_cache"] = cache
    elif prefetch_click_ecg and len(df_plot) > MAX_ECG_PREFETCH_POINTS:
        st.info(
            f"Skipping ECG prefetch for {len(df_plot)} points (> {MAX_ECG_PREFETCH_POINTS}) "
            "to keep embedding plot responsive."
        )

    hover_cols = get_hover_cols(df_plot, filterable_cols)
    plot_cols = set(hover_cols + ["x", "y"])
    if "z" in df_plot.columns:
        plot_cols.add("z")
    if color_by != "None":
        plot_cols.add(color_by)
    df_plot_view = df_plot[[col for col in plot_cols if col in df_plot.columns]].copy()

    fig = plot_embedding_scatter(
        X_2d,
        df_plot_view,
        color_by=None if color_by == "None" else color_by,
        marker_size=point_size,
        marker_opacity=point_opacity,
        show=False,
        hover_cols=hover_cols,
        render_mode="webgl",
    )
    fig.update_layout(
        height=760,
        margin=dict(l=10, r=10, t=40, b=10),
        clickmode="event+select",
        dragmode="pan",
    )
    if X_2d.shape[1] == 2:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    else:
        fig.update_layout(
            scene=dict(
                xaxis_title="Component 1",
                yaxis_title="Component 2",
                zaxis_title="Component 3",
                dragmode="orbit",
            )
        )

    left_col, right_col = st.columns([1.35, 1.2], gap="medium")
    selected_points = []
    selected_point_indices: list[int] = []

    with left_col:
        st.subheader("Embedding Scatter")
        if X_2d.shape[1] == 3:
            st.caption("3D embedding is enabled: rotate and zoom to inspect points.")
        chart_key = "embedding_scatter_chart_3d" if X_2d.shape[1] == 3 else "embedding_scatter_chart_2d"
        selection = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode=("points",),
            key=chart_key,
            config={
                "displaylogo": False,
                "modeBarButtonsToAdd": ["select2d", "lasso2d"],
            },
        )
        try:
            sel = None
            if selection and hasattr(selection, "selection"):
                sel = selection.selection
            elif isinstance(selection, dict):
                sel = selection.get("selection")
            if not sel:
                chart_state = st.session_state.get(chart_key)
                if isinstance(chart_state, dict):
                    sel = chart_state.get("selection")
            if isinstance(sel, dict):
                if sel.get("points"):
                    selected_points = sel["points"]
                    for point in selected_points:
                        point_index = _extract_point_index(point)
                        if point_index is not None:
                            selected_point_indices.append(point_index)
                if isinstance(sel.get("point_indices"), list):
                    selected_point_indices = [
                        int(i) for i in sel["point_indices"] if i is not None
                    ]
                elif isinstance(sel.get("indices"), list):
                    selected_point_indices = [
                        int(i) for i in sel["indices"] if i is not None
                    ]
        except Exception:
            selected_points = []
            selected_point_indices = []

        if not selected_points and not selected_point_indices:
            st.caption("Click a point to update ECG on the right panel.")
            if X_2d.shape[1] == 3:
                st.caption(
                    "Native Streamlit 3D single-click callbacks are not stable. "
                    "Use the ecg_id controls on the right to load ECG reliably."
                )

    selected_ecg_ids_from_plot: list[str] = []
    if selected_point_indices:
        dedup_idx_for_copy: list[int] = []
        seen_idx_for_copy: set[int] = set()
        for idx in selected_point_indices:
            int_idx = int(idx)
            if int_idx not in seen_idx_for_copy and 0 <= int_idx < len(df_plot):
                dedup_idx_for_copy.append(int_idx)
                seen_idx_for_copy.add(int_idx)
        selected_ecg_ids_from_plot = [
            str(df_plot.iloc[idx]["ecg_id"]) for idx in dedup_idx_for_copy
        ]
    elif selected_points:
        seen_ecg_ids: set[str] = set()
        for point in selected_points:
            point_ecg_id = _extract_ecg_id_from_point(point)
            if point_ecg_id is None:
                point_index = _extract_point_index(point)
                if point_index is not None and 0 <= int(point_index) < len(df_plot):
                    point_ecg_id = str(df_plot.iloc[int(point_index)]["ecg_id"])
            if point_ecg_id:
                point_ecg_id = str(point_ecg_id)
                if point_ecg_id not in seen_ecg_ids:
                    selected_ecg_ids_from_plot.append(point_ecg_id)
                    seen_ecg_ids.add(point_ecg_id)

    with left_col:
        if selected_ecg_ids_from_plot:
            st.caption("Selected ecg_id(s) from plot (copyable)")
            st.code("\n".join(selected_ecg_ids_from_plot))

    if selected_points or selected_point_indices:
        selected_ecg_id = None
        selected_idx = None
        if selected_point_indices:
            dedup_indices: list[int] = []
            seen_indices: set[int] = set()
            for idx in selected_point_indices:
                if idx not in seen_indices:
                    dedup_indices.append(int(idx))
                    seen_indices.add(int(idx))
            prev_idx = st.session_state.get("selected_point_index")
            for idx in reversed(dedup_indices):
                if prev_idx is None or int(idx) != int(prev_idx):
                    selected_idx = int(idx)
                    break
            if selected_idx is None:
                selected_idx = int(dedup_indices[-1])
            selected_ecg_id = df_plot.iloc[selected_idx]["ecg_id"]
        elif selected_points:
            point = selected_points[-1]
            selected_ecg_id = _extract_ecg_id_from_point(point)
            if selected_ecg_id is None:
                point_index = _extract_point_index(point)
                if point_index is not None:
                    selected_idx = int(point_index)
                    selected_ecg_id = df_plot.iloc[selected_idx]["ecg_id"]
        if selected_ecg_id is None:
            st.warning("Point selection received but ecg_id could not be resolved.")
            st.stop()
        st.session_state["selected_ecg_id"] = str(selected_ecg_id)
        if selected_idx is not None:
            st.session_state["selected_point_index"] = int(selected_idx)
        st.session_state["current_ecg_id"] = str(selected_ecg_id)

    with right_col:
        st.subheader("ECG Panel")
        ecg_id_options = df_plot["ecg_id"].astype(str).tolist()
        if not ecg_id_options:
            st.info("No ecg_id available in current plotted data.")
            return
        current_id_raw = st.session_state.get("current_ecg_id")
        current_id = (
            _normalize_ecg_id(current_id_raw) if current_id_raw is not None else ""
        )
        selected_ecg_id = current_id if current_id in ecg_id_options else None

        # Keep picker synced to the current selected id, but do not auto-load a default id.
        if "ecg_id_picker" not in st.session_state:
            st.session_state["ecg_id_picker"] = (
                selected_ecg_id if selected_ecg_id is not None else ecg_id_options[0]
            )
        elif selected_ecg_id is not None and st.session_state.get("ecg_id_picker") != selected_ecg_id:
            st.session_state["ecg_id_picker"] = selected_ecg_id

        selected_row = None
        df_ecg_row = None
        if selected_ecg_id is not None:
            matches = df_plot[df_plot["ecg_id"].astype(str) == selected_ecg_id]
            if not matches.empty:
                selected_row = matches.iloc[0]

            if df_ecg is not None:
                df_ecg_row = df_ecg[df_ecg["ecg_id"].astype(str) == selected_ecg_id]
            else:
                prefetched_rows = st.session_state.get("embedding_cache", {}).get(
                    "prefetched_ecg_rows", {}
                )
                df_ecg_row = prefetched_rows.get(str(selected_ecg_id))
                if df_ecg_row is None:
                    df_ecg_row = _get_session_ecg_row(selected_ecg_id)
                if df_ecg_row is None:
                    df_ecg_row = _cached_load_ecg_row(ecg_path, selected_ecg_id)
                    if df_ecg_row is not None and not df_ecg_row.empty:
                        _set_session_ecg_row(selected_ecg_id, df_ecg_row)

        if selected_ecg_id is None:
            st.info("No ECG selected yet. Click a point on the left or load an ecg_id below.")
        elif df_ecg_row is None or df_ecg_row.empty:
            st.error(f"ecg_id '{selected_ecg_id}' not found in df_ecg.")
        else:
            st.session_state["current_ecg_id"] = selected_ecg_id
            selected_leads = sorted([c for c in df_ecg_row.columns if c != "ecg_id"])
            leads = np.stack([df_ecg_row.iloc[0][col] for col in selected_leads])
            ecg_title = f"ECG {selected_ecg_id}"

            ecg_fig_small = plot_ecg(
                leads, title=ecg_title, lead_names=selected_leads, show=False
            )
            ecg_fig_small.update_layout(height=min(620, max(420, 135 * len(selected_leads))))
            st.plotly_chart(ecg_fig_small, use_container_width=True)
            st.caption("Current ecg_id")
            st.code(selected_ecg_id)
            st.caption("Use the copy icon in the code block.")

        st.subheader("Embedding Metadata")
        if selected_ecg_id is None:
            st.info("No embedding selected yet.")
        elif selected_row is not None:
            drop_cols = []
            for col in selected_row.index:
                if isinstance(selected_row[col], np.ndarray):
                    drop_cols.append(col)
            meta_row = selected_row.drop(labels=drop_cols, errors="ignore")
            st.dataframe(meta_row.to_frame(name="value"), use_container_width=True)
        else:
            st.info("Metadata not available for this ecg_id in the current view.")

        with st.expander("ecg_id Copy Assistant", expanded=False):
            st.caption(
                "3D tooltip text is usually not directly copyable; "
                "search and copy ecg_id here."
            )
            copy_query = st.text_input(
                "Search ecg_id by substring",
                value="",
                key="ecg_id_copy_query",
                placeholder="Type any fragment, e.g. fed2abc or a suffix",
            ).strip()
            if copy_query:
                matched_ids = [
                    eid for eid in ecg_id_options if copy_query.lower() in eid.lower()
                ]
            else:
                matched_ids = ecg_id_options
            max_show = 500
            shown_ids = matched_ids[:max_show]
            if not shown_ids:
                st.info("No matching ecg_id found.")
            else:
                if len(matched_ids) > max_show:
                    st.caption(f"{len(matched_ids)} matches found; showing first {max_show}.")
                helper_pick = st.selectbox(
                    "Matching ecg_id",
                    options=shown_ids,
                    key="ecg_id_copy_picker",
                )
                st.code(helper_pick)
                st.caption("Use the copy icon in the code block.")
                if st.button("Load this ecg_id", key="load_helper_ecg_id"):
                    st.session_state["current_ecg_id"] = helper_pick
                    st.session_state["ecg_id_picker"] = helper_pick
                    st.rerun()

        with st.expander("Load ECG by ecg_id", expanded=True):
            typed_ecg_id = st.text_input(
                "Type or paste ecg_id",
                value=selected_ecg_id or "",
                key="ecg_id_search_input",
            )
            load_typed = st.button("Load typed ecg_id", key="load_typed_ecg_id")
            if load_typed:
                typed = _normalize_ecg_id(typed_ecg_id)
                if not typed:
                    st.warning("Please enter an ecg_id.")
                elif typed not in set(ecg_id_options):
                    st.warning(
                        "This ecg_id is not in the current plotted sample. "
                        "Adjust filtering/sampling or pick from the dropdown."
                    )
                else:
                    st.session_state["current_ecg_id"] = typed
                    st.session_state["ecg_id_picker"] = typed
                    st.rerun()

            picked_ecg_id = st.selectbox(
                "Or choose ecg_id from current plot",
                options=ecg_id_options,
                key="ecg_id_picker",
            )
            if st.button("Load dropdown ecg_id", key="load_dropdown_ecg_id"):
                picked = _normalize_ecg_id(picked_ecg_id)
                if picked:
                    st.session_state["current_ecg_id"] = picked
                    st.rerun()


if __name__ == "__main__":
    main()
