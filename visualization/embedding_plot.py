import numpy as np
import pandas as pd
import plotly.express as px
from pandas.api.types import is_bool_dtype, is_numeric_dtype, is_object_dtype


def plot_embedding_scatter(
    X_2d: np.ndarray,
    df_embeddings: pd.DataFrame,
    color_by: str | None = None,
    show: bool = True,
    hover_cols: list[str] | None = None,
    render_mode: str = "webgl",
) -> "plotly.graph_objects.Figure":
    if X_2d.ndim != 2 or X_2d.shape[1] != 2:
        raise ValueError("X_2d must be a 2D array with shape (N, 2).")
    if len(df_embeddings) != X_2d.shape[0]:
        raise ValueError(
            "df_embeddings must have the same number of rows as X_2d has points."
        )

    df_plot = df_embeddings.copy()
    df_plot["__x__"] = X_2d[:, 0].astype("float64", copy=False)
    df_plot["__y__"] = X_2d[:, 1].astype("float64", copy=False)

    if hover_cols is None:
        hover_cols = ["ecg_id"]
        for col in ("patient_id", "device_group", "sampling_rate"):
            if col in df_embeddings.columns and col not in hover_cols:
                hover_cols.append(col)

    if color_by is None:
        fig = px.scatter(
            df_plot,
            x="__x__",
            y="__y__",
            hover_data=hover_cols,
            render_mode=render_mode,
        )
        fig.update_traces(marker=dict(size=6, opacity=0.8))
        fig.update_layout(
            title="Embedding Scatter",
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        if show:
            fig.show()
        return fig

    if color_by not in df_embeddings.columns:
        raise ValueError(f"color_by '{color_by}' not found in df_embeddings columns.")

    series = df_embeddings[color_by]

    # Decide categorical vs numerical coloring based on dtype.
    if is_numeric_dtype(series) and not is_bool_dtype(series):
        df_plot[color_by] = series.to_numpy()
        fig = px.scatter(
            df_plot,
            x="__x__",
            y="__y__",
            color=color_by,
            hover_data=hover_cols,
            render_mode=render_mode,
        )
        fig.update_traces(marker=dict(size=6, opacity=0.8))
        fig.update_layout(
            title=f"Embedding Scatter colored by {color_by}",
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            margin=dict(l=10, r=10, t=40, b=10),
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

    df_plot[color_by] = series.astype("category")
    fig = px.scatter(
        df_plot,
        x="__x__",
        y="__y__",
        color=color_by,
        hover_data=hover_cols,
        render_mode=render_mode,
    )
    fig.update_traces(marker=dict(size=6, opacity=0.8))
    fig.update_layout(
        title=f"Embedding Scatter colored by {color_by}",
        xaxis_title="Component 1",
        yaxis_title="Component 2",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    if show:
        fig.show()
    return fig


def get_hover_cols(df_plot: pd.DataFrame, candidate_cols: list[str]) -> list[str]:
    hover_cols = ["ecg_id"]
    for col in candidate_cols:
        if col in df_plot.columns and col not in hover_cols:
            hover_cols.append(col)
    return hover_cols
