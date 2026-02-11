import numpy as np
import pandas as pd
import plotly.express as px
from pandas.api.types import is_bool_dtype, is_numeric_dtype, is_object_dtype


def _compact_legend_layout() -> dict:
    return {
        "legend": dict(
            x=1.0,
            y=1.0,
            xanchor="right",
            yanchor="top",
            font=dict(size=10),
            itemsizing="constant",
            itemwidth=30,
            bgcolor="rgba(255,255,255,0.6)",
        )
    }


def plot_embedding_scatter(
    X_2d: np.ndarray,
    df_embeddings: pd.DataFrame,
    color_by: str | None = None,
    marker_size: float = 6.0,
    marker_opacity: float = 0.8,
    show: bool = True,
    hover_cols: list[str] | None = None,
    render_mode: str = "webgl",
) -> "plotly.graph_objects.Figure":
    if X_2d.ndim != 2 or X_2d.shape[1] not in (2, 3):
        raise ValueError("X_2d must have shape (N, 2) or (N, 3).")
    if len(df_embeddings) != X_2d.shape[0]:
        raise ValueError(
            "df_embeddings must have the same number of rows as X_2d has points."
        )
    is_3d = X_2d.shape[1] == 3

    # Build a minimal plotting frame to avoid copying full metadata for large N.
    data: dict[str, object] = {
        "__x__": X_2d[:, 0].astype("float64", copy=False),
        "__y__": X_2d[:, 1].astype("float64", copy=False),
    }
    if is_3d:
        data["__z__"] = X_2d[:, 2].astype("float64", copy=False)

    if hover_cols is None:
        hover_cols = ["ecg_id"]
        for col in ("patient_id", "device_group", "sampling_rate"):
            if col in df_embeddings.columns and col not in hover_cols:
                hover_cols.append(col)
    for col in hover_cols:
        if col in df_embeddings.columns:
            data[col] = df_embeddings[col].to_numpy(copy=False)
    custom_data_cols = ["ecg_id"] if "ecg_id" in df_embeddings.columns else None

    if color_by is None:
        df_plot = pd.DataFrame(data)
        if is_3d:
            fig = px.scatter_3d(
                df_plot,
                x="__x__",
                y="__y__",
                z="__z__",
                hover_data=hover_cols,
                custom_data=custom_data_cols,
            )
        else:
            fig = px.scatter(
                df_plot,
                x="__x__",
                y="__y__",
                hover_data=hover_cols,
                custom_data=custom_data_cols,
                render_mode=render_mode,
            )
        fig.update_traces(marker=dict(size=float(marker_size), opacity=float(marker_opacity)))
        fig.update_layout(
            title="Embedding Scatter",
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            margin=dict(l=10, r=10, t=40, b=10),
            **_compact_legend_layout(),
        )
        if show:
            fig.show()
        return fig

    if color_by not in df_embeddings.columns:
        raise ValueError(f"color_by '{color_by}' not found in df_embeddings columns.")

    series = df_embeddings[color_by]

    # Decide categorical vs numerical coloring based on dtype.
    if is_numeric_dtype(series) and not is_bool_dtype(series):
        data[color_by] = series.to_numpy(copy=False)
        df_plot = pd.DataFrame(data)
        if is_3d:
            fig = px.scatter_3d(
                df_plot,
                x="__x__",
                y="__y__",
                z="__z__",
                color=color_by,
                hover_data=hover_cols,
                custom_data=custom_data_cols,
            )
        else:
            fig = px.scatter(
                df_plot,
                x="__x__",
                y="__y__",
                color=color_by,
                hover_data=hover_cols,
                custom_data=custom_data_cols,
                render_mode=render_mode,
            )
        fig.update_traces(marker=dict(size=float(marker_size), opacity=float(marker_opacity)))
        fig.update_layout(
            title=f"Embedding Scatter colored by {color_by}",
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            margin=dict(l=10, r=10, t=40, b=10),
            coloraxis_colorbar=dict(
                thickness=10,
                x=1.01,
                xanchor="left",
                y=0.5,
                yanchor="middle",
            ),
            **_compact_legend_layout(),
        )
        if show:
            fig.show()
        return fig

    # Treat as categorical by default. Convert unhashable values to strings.
    def _safe_label(value: object) -> object:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return np.nan
        if isinstance(value, np.ndarray):
            return np.array2string(value, separator=",")
        if isinstance(value, (list, tuple, dict, set)):
            return str(value)
        return value

    if is_object_dtype(series):
        series = series.map(_safe_label)

    # Prevent trace explosion for very high-cardinality categorical colors.
    max_categories = 60
    if len(df_embeddings) >= 5000:
        value_counts = series.value_counts(dropna=False)
        if len(value_counts) > max_categories:
            top_values = set(value_counts.head(max_categories - 1).index.tolist())
            series = series.where(series.isin(top_values), other="Other")

    data[color_by] = series.astype("category")
    df_plot = pd.DataFrame(data)
    if is_3d:
        fig = px.scatter_3d(
            df_plot,
            x="__x__",
            y="__y__",
            z="__z__",
            color=color_by,
            hover_data=hover_cols,
            custom_data=custom_data_cols,
        )
    else:
        fig = px.scatter(
            df_plot,
            x="__x__",
            y="__y__",
            color=color_by,
            hover_data=hover_cols,
            custom_data=custom_data_cols,
            render_mode=render_mode,
        )
    fig.update_traces(marker=dict(size=float(marker_size), opacity=float(marker_opacity)))
    fig.update_layout(
        title=f"Embedding Scatter colored by {color_by}",
        xaxis_title="Component 1",
        yaxis_title="Component 2",
        margin=dict(l=10, r=10, t=40, b=10),
        **_compact_legend_layout(),
    )
    if show:
        fig.show()
    return fig


def get_hover_cols(df_plot: pd.DataFrame, candidate_cols: list[str]) -> list[str]:
    hover_cols = ["ecg_id"]
    preferred = ["patient_id", "device_group", "sampling_rate"]
    for col in preferred:
        if col in candidate_cols and col in df_plot.columns and col not in hover_cols:
            hover_cols.append(col)
    max_hover_cols = 8
    for col in candidate_cols:
        if col in df_plot.columns and col not in hover_cols:
            hover_cols.append(col)
        if len(hover_cols) >= max_hover_cols:
            break
    return hover_cols
