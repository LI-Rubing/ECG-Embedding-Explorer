import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_ecg(
    signal: np.ndarray,
    title: str | None = None,
    lead_names: list[str] | None = None,
    show: bool = True,
) -> go.Figure:
    n_leads = signal.shape[0]
    if lead_names is None:
        lead_names = [f"Lead {idx}" for idx in range(1, n_leads + 1)]

    x = np.arange(signal.shape[1])
    fig = make_subplots(
        rows=n_leads,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=[
            lead_names[idx] if idx < len(lead_names) else f"Lead {idx + 1}"
            for idx in range(n_leads)
        ],
    )
    for idx in range(n_leads):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=signal[idx],
                mode="lines",
                name=lead_names[idx] if idx < len(lead_names) else f"Lead {idx + 1}",
                showlegend=True,
            ),
            row=idx + 1,
            col=1,
        )

    fig.update_layout(
        title=title or "ECG",
        height=180 * n_leads,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_xaxes(title_text="Time", row=n_leads, col=1)
    for r in range(1, n_leads + 1):
        fig.update_yaxes(title_text="Amplitude", row=r, col=1)

    if show:
        fig.show()
    return fig
