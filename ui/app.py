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
    load_ecg_metadata,
    load_ecg_row,
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


@st.cache_data(show_spinner=False)
def _cached_load_dataframe(path: str) -> pd.DataFrame:
    return load_dataframe(path)


@st.cache_data(show_spinner=False)
def _cached_reduce(
    X: np.ndarray, method: str, params: dict[str, Any]
) -> np.ndarray:
    return reduce_embeddings(X, method=method, params=params)


def _load_dataframe_from_upload(uploaded_file) -> pd.DataFrame:
    return load_dataframe_from_upload(uploaded_file)


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


def main() -> None:
    st.set_page_config(page_title="ECG Embedding Explorer", layout="wide")
    st.title("ECG Embedding Explorer")

    st.sidebar.header("Data Loading")
    emb_upload = None
    ecg_upload = None
    with st.sidebar.expander("Optional uploads", expanded=False):
        emb_upload = st.file_uploader(
            "Upload df_embeddings (parquet/csv/pkl)", type=["parquet", "csv", "pkl"]
        )
        ecg_upload = st.file_uploader(
            "Upload df_ecg (parquet/csv/pkl)", type=["parquet", "csv", "pkl"]
        )
    emb_path = st.sidebar.text_input(
        "df_embeddings path", value="data/df_embeddings.pkl"
    )
    ecg_path = st.sidebar.text_input("df_ecg path", value="data/df_ecg.parquet")

    if st.sidebar.button("Load Data"):
        st.session_state.pop("df_embeddings", None)
        st.session_state.pop("df_ecg", None)

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

    st.sidebar.header("Embedding Selection")
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

    embedding_source = st.sidebar.selectbox(
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
        xy_x_col = st.sidebar.selectbox("X column", numeric_cols, index=x_index)
        y_candidates = [c for c in numeric_cols if c != xy_x_col]
        if default_y in y_candidates:
            y_index = y_candidates.index(default_y)
        else:
            y_index = 0 if y_candidates else 0
        xy_y_col = st.sidebar.selectbox("Y column", y_candidates, index=y_index)

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

    st.sidebar.header("Validation")
    if st.sidebar.button("Run Full Validation"):
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

    st.sidebar.header("Filtering")
    filterable_cols = [
        col for col in df_embeddings.columns if col not in array_cols
    ]
    filter_column = st.sidebar.selectbox(
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
            st.sidebar.header("ECG Filtering")
            ecg_filter_col = st.sidebar.selectbox(
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

    st.sidebar.header("Sampling")
    sample_n = st.sidebar.number_input(
        "Sample size (0=all)",
        min_value=0,
        max_value=50000,
        value=200,
    )
    random_state = st.sidebar.number_input(
        "Random state", min_value=0, max_value=10_000, value=0
    )
    if sample_n and len(df_filtered) > sample_n:
        df_filtered = df_filtered.sample(
            n=int(sample_n), random_state=int(random_state)
        ).reset_index(drop=True)

    summary_filtered = (
        f"Filtered df_embeddings: rows={len(df_filtered)}; "
        "ECG will be loaded on demand."
    )

    st.sidebar.header("Coloring")
    colorable_cols = [
        col for col in df_filtered.columns if col not in array_cols
    ]
    color_by = st.sidebar.selectbox("Color by", ["None"] + colorable_cols)

    st.sidebar.header("Interaction")
    interaction_mode = st.sidebar.selectbox(
        "Selection mode", ["click (experimental)", "manual"], index=1
    )
    if interaction_mode.startswith("click") and sample_n and sample_n > 300:
        st.sidebar.warning(
            "Click mode is unstable above ~300 points on low-memory machines. "
            "Capping sample size to 300."
        )
        sample_n = 300

    st.sidebar.header("Dimensionality Reduction")
    if xy_mode in {"precomputed", "manual"}:
        method = "precomputed"
        params = {}
    else:
        method = st.sidebar.selectbox("Method", ["pca", "tsne", "umap"])
        params: dict[str, Any] = {}
        if method == "pca":
            params["whiten"] = st.sidebar.checkbox("PCA whiten", value=False)
        elif method == "tsne":
            params["perplexity"] = st.sidebar.number_input(
                "t-SNE perplexity", min_value=2.0, max_value=100.0, value=30.0
            )
            params["learning_rate"] = st.sidebar.number_input(
                "t-SNE learning_rate", min_value=10.0, max_value=1000.0, value=200.0
            )
            params["n_iter"] = st.sidebar.number_input(
                "t-SNE n_iter", min_value=250, max_value=5000, value=1000
            )
        else:
            params["n_neighbors"] = st.sidebar.number_input(
                "UMAP n_neighbors", min_value=2, max_value=100, value=15
            )
            params["min_dist"] = st.sidebar.number_input(
                "UMAP min_dist", min_value=0.0, max_value=1.0, value=0.1
            )
            params["metric"] = st.sidebar.text_input(
                "UMAP metric", value="euclidean"
            )

    def _current_config_key() -> tuple[object, ...]:
        return (
            embedding_source,
            embedding_col,
            xy_mode,
            xy_x_col,
            xy_y_col,
            method,
            tuple(sorted(params.items())),
            filter_column,
            sample_n,
            random_state,
            len(df_filtered),
        )

    compute_now = st.sidebar.button("Compute Embedding")
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
                # If embeddings are already 2D, skip reduction.
                if embeddings_matrix.ndim == 2 and embeddings_matrix.shape[1] == 2:
                    X_2d = embeddings_matrix
                else:
                    try:
                        X_2d = _cached_reduce(embeddings_matrix, method, params)
                    except Exception as exc:
                        st.error(str(exc))
                        st.stop()

            reduction_time = time.perf_counter() - start
        st.session_state["embedding_cache"] = {
            "key": _current_config_key(),
            "X_2d": X_2d,
            "df_plot": df_filtered.reset_index(drop=True).copy(),
            "reduction_time": reduction_time,
            "method": method,
            "params": params,
            "fig": None,
            "df_plot_view": None,
        }
        st.success("Embedding cache ready.")
    else:
        cache = st.session_state["embedding_cache"]
        if cache["key"] != _current_config_key():
            st.info("Parameters changed. Click 'Compute Embedding' to update.")
        X_2d = cache["X_2d"]
        df_filtered = cache["df_plot"].copy()
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
                    pca_model = PCA(n_components=2, whiten=params.get("whiten", False))
                    pca_model.fit(embeddings_matrix)
                    var_ratio = pca_model.explained_variance_ratio_
                    fig_var = go.Figure(
                        data=[
                            go.Bar(
                                x=["PC1", "PC2"],
                                y=var_ratio,
                                text=[f"{v:.3f}" for v in var_ratio],
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

                    def _top_loadings(component_idx: int, top_k: int = 10):
                        comp = pca_model.components_[component_idx]
                        idx = np.argsort(np.abs(comp))[::-1][:top_k]
                        return pd.DataFrame(
                            {
                                "feature_index": idx,
                                "loading": comp[idx],
                                "abs_loading": np.abs(comp[idx]),
                            }
                        )

                    st.write("Top loadings for PC1:")
                    st.dataframe(_top_loadings(0))
                    st.write("Top loadings for PC2:")
                    st.dataframe(_top_loadings(1))

    df_plot = df_filtered.reset_index(drop=True).copy()
    df_plot["x"] = X_2d[:, 0]
    df_plot["y"] = X_2d[:, 1]

    hover_cols = get_hover_cols(df_plot, filterable_cols)
    plot_cols = set(hover_cols + ["x", "y"])
    if color_by != "None":
        plot_cols.add(color_by)
    df_plot_view = df_plot[[col for col in plot_cols if col in df_plot.columns]].copy()

    st.subheader("Embedding Scatter")
    cache = st.session_state.get("embedding_cache", {})
    if cache.get("fig") is None or cache.get("df_plot_view") is None:
        fig = plot_embedding_scatter(
            X_2d,
            df_plot_view,
            color_by=None if color_by == "None" else color_by,
            show=False,
            hover_cols=hover_cols,
            render_mode="webgl",
        )
        cache["fig"] = fig
        cache["df_plot_view"] = df_plot_view
        st.session_state["embedding_cache"] = cache
    else:
        fig = cache["fig"]

    selected_points = []
    selected_point_index = st.session_state.get("selected_point_index")

    def _with_highlight(base_fig, point_index: int | None):
        if point_index is None:
            return base_fig
        fig_hl = go.Figure(base_fig)
        fig_hl.add_trace(
            go.Scatter(
                x=[df_plot.iloc[point_index]["x"]],
                y=[df_plot.iloc[point_index]["y"]],
                mode="markers",
                marker=dict(size=12, color="red", symbol="x"),
                name="selected",
                showlegend=False,
            )
        )
        return fig_hl

    chart_placeholder = st.empty()

    selection = chart_placeholder.plotly_chart(
        _with_highlight(fig, selected_point_index),
        use_container_width=True,
        on_select="rerun",
    )
    try:
        if selection and hasattr(selection, "selection"):
            sel = selection.selection
            if isinstance(sel, dict) and sel.get("points"):
                selected_points = sel["points"]
    except Exception:
        selected_points = []

    if not selected_points:
        st.info("Click a point to capture ecg_id.")

    if selected_points:
        point = selected_points[0]
        selected_ecg_id = point.get("ecg_id")
        if selected_ecg_id is None:
            point_index = None
            for key in ("pointIndex", "point_index", "pointNumber", "point_number"):
                if key in point:
                    point_index = int(point[key])
                    break
            if point_index is None:
                st.error(f"Unsupported selection format: {point}")
                st.stop()
            selected_row = df_plot.iloc[point_index]
            selected_ecg_id = selected_row["ecg_id"]
            selected_point_index = point_index
        else:
            selected_row = df_plot[df_plot["ecg_id"] == selected_ecg_id].iloc[0]
            selected_point_index = int(
                df_plot.index[df_plot["ecg_id"] == selected_ecg_id][0]
            )
        selected_ecg_id = selected_row["ecg_id"]
        st.subheader("Selected ECG")
        st.session_state["selected_point_index"] = selected_point_index
        chart_placeholder.plotly_chart(
            _with_highlight(fig, selected_point_index),
            use_container_width=True,
        )
        st.subheader("Copy ecg_id")
        st.code(selected_ecg_id)
        st.session_state["selected_ecg_id"] = selected_ecg_id
        if interaction_mode.startswith("click"):
            st.session_state["pending_ecg_id"] = selected_ecg_id

    if not interaction_mode.startswith("click"):
        manual_id = st.text_input(
            "Paste ecg_id to load", value=st.session_state.get("selected_ecg_id", "")
        )
        if st.button("Load ECG"):
            if manual_id:
                st.session_state["pending_ecg_id"] = manual_id

    pending_ecg_id = st.session_state.get("pending_ecg_id")
    if not pending_ecg_id:
        return
    selected_ecg_id = pending_ecg_id
    status = st.status("Plotting ECG...", state="running")
    meta_lines = [f"ecg_id: {selected_ecg_id}"]
    selected_row = None
    if "df_plot" in locals():
        matches = df_plot[df_plot["ecg_id"] == selected_ecg_id]
        if not matches.empty:
            selected_row = matches.iloc[0]
    if selected_row is not None:
        if "patient_id" in selected_row:
            meta_lines.append(f"patient_id: {selected_row['patient_id']}")
        if "device_group" in selected_row:
            meta_lines.append(f"device_group: {selected_row['device_group']}")
        if "sampling_rate" in selected_row:
            meta_lines.append(f"sampling_rate: {selected_row['sampling_rate']}")
    st.write(" | ".join(meta_lines))

    st.subheader("Embedding Metadata (All Labels)")
    if selected_row is not None:
        drop_cols = []
        for col in selected_row.index:
            if isinstance(selected_row[col], np.ndarray):
                drop_cols.append(col)
        meta_row = selected_row.drop(labels=drop_cols, errors="ignore")
        st.dataframe(meta_row.to_frame(name="value"))
    else:
        st.info("Metadata not available for this ecg_id in the current view.")

    if df_ecg is not None:
        df_ecg_row = df_ecg[df_ecg["ecg_id"] == selected_ecg_id]
    else:
        df_ecg_row = load_ecg_row(selected_ecg_id, ecg_path)
    if df_ecg_row.empty:
        st.error(f"ecg_id '{selected_ecg_id}' not found in df_ecg.")
        st.stop()
    lead_cols = sorted([c for c in df_ecg_row.columns if c != "ecg_id"])
    st.subheader("Lead Selection")
    col_all, col_none = st.columns(2)
    if col_all.button("Select all leads"):
        st.session_state["selected_leads"] = lead_cols
    if col_none.button("Select none"):
        st.session_state["selected_leads"] = []
    selected_leads = st.multiselect(
        "Visible leads",
        options=lead_cols,
        default=st.session_state.get("selected_leads", lead_cols),
    )
    st.session_state["selected_leads"] = selected_leads
    if not selected_leads:
        st.info("No leads selected.")
        return
    leads = np.stack([df_ecg_row.iloc[0][col] for col in selected_leads])

    ecg_title = f"ECG {selected_ecg_id}"
    ecg_fig = plot_ecg(
        leads, title=ecg_title, lead_names=selected_leads, show=False
    )
    st.plotly_chart(ecg_fig, use_container_width=True)
    status.update(label="Plotting succeeded", state="complete")


if __name__ == "__main__":
    main()
