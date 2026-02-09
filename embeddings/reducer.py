import inspect
from typing import Any

import numpy as np


def reduce_embeddings(X: np.ndarray, method: str, params: dict[str, Any]) -> np.ndarray:
    """
    Compute 2D embedding using PCA / t-SNE / UMAP.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Input embedding matrix.
    method : str
        One of {"pca", "tsne", "umap"}.
    **params :
        Method-specific parameters.

    Returns
    -------
    X_2d : np.ndarray, shape (n_samples, 2)
        2D projected embeddings.
    """
    global _LAST_STATS
    method = method.lower()
    params = params or {}

    if method == "pca":
        try:
            from sklearn.decomposition import PCA
        except ImportError as exc:
            raise ImportError(
                "Missing dependency 'scikit-learn'. "
                "Install it with: pip install scikit-learn"
            ) from exc

        model = PCA(n_components=2, **params)
        X_2d = model.fit_transform(X)

    elif method == "tsne":
        try:
            from sklearn.manifold import TSNE
        except ImportError as exc:
            raise ImportError(
                "Missing dependency 'scikit-learn'. "
                "Install it with: pip install scikit-learn"
            ) from exc

        # Handle sklearn API differences: some versions use max_iter instead of n_iter.
        tsne_params = dict(params)
        tsne_sig = inspect.signature(TSNE.__init__)
        has_n_iter = "n_iter" in tsne_sig.parameters
        has_max_iter = "max_iter" in tsne_sig.parameters
        if "n_iter" in tsne_params and has_max_iter and not has_n_iter:
            tsne_params["max_iter"] = tsne_params.pop("n_iter")
        if "max_iter" in tsne_params and has_n_iter and not has_max_iter:
            tsne_params["n_iter"] = tsne_params.pop("max_iter")

        model = TSNE(n_components=2, **tsne_params)
        X_2d = model.fit_transform(X)

    elif method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise ImportError(
                "Missing dependency 'umap-learn'. "
                "Install it with: pip install umap-learn"
            ) from exc

        model = umap.UMAP(n_components=2, **params)
        X_2d = model.fit_transform(X)

    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            "Choose from {'pca', 'tsne', 'umap'}."
        )

    return X_2d
