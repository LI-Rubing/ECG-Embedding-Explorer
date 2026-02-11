import inspect
import os
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

        pca_params = dict(params)
        n_components = int(pca_params.pop("n_components", 2))
        model = PCA(n_components=n_components, **pca_params)
        X_2d = model.fit_transform(X)

    elif method == "tsne":
        try:
            from sklearn.manifold import TSNE
        except ImportError as exc:
            raise ImportError(
                "Missing dependency 'scikit-learn'. "
                "Install it with: pip install scikit-learn"
            ) from exc

        X_work = np.asarray(X, dtype=np.float32, order="C")

        # Handle sklearn API differences: some versions use max_iter instead of n_iter.
        tsne_params = dict(params)
        tsne_sig = inspect.signature(TSNE.__init__)
        tsne_sig_params = set(tsne_sig.parameters.keys())
        has_n_iter = "n_iter" in tsne_sig.parameters
        has_max_iter = "max_iter" in tsne_sig.parameters
        if "learning_rate" in tsne_sig.parameters and "learning_rate" not in tsne_params:
            tsne_params["learning_rate"] = "auto"
        if "method" in tsne_sig.parameters and "method" not in tsne_params:
            tsne_params["method"] = "barnes_hut"
        if "angle" in tsne_sig.parameters and "angle" not in tsne_params:
            tsne_params["angle"] = 0.5
        if "n_jobs" in tsne_sig.parameters and "n_jobs" not in tsne_params:
            tsne_params["n_jobs"] = -1

        # Guard invalid perplexity for different sample sizes.
        if "perplexity" in tsne_params:
            max_perplexity = max(5.0, (X_work.shape[0] - 1) / 3.0)
            tsne_params["perplexity"] = float(
                min(float(tsne_params["perplexity"]), max_perplexity)
            )

        if "n_iter" in tsne_params and has_max_iter and not has_n_iter:
            tsne_params["max_iter"] = tsne_params.pop("n_iter")
        if "max_iter" in tsne_params and has_n_iter and not has_max_iter:
            tsne_params["n_iter"] = tsne_params.pop("max_iter")

        # Drop params unsupported by the installed sklearn version.
        for key in list(tsne_params.keys()):
            if key not in tsne_sig_params:
                tsne_params.pop(key, None)

        n_components = int(tsne_params.pop("n_components", 2))
        model = TSNE(n_components=n_components, **tsne_params)
        X_2d = model.fit_transform(X_work)

    elif method == "umap":
        try:
            os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
            import umap
        except ImportError as exc:
            raise ImportError(
                "Missing dependency 'umap-learn'. "
                "Install it with: pip install umap-learn"
            ) from exc

        X_work = np.asarray(X, dtype=np.float32, order="C")

        umap_params = dict(params)
        umap_sig = inspect.signature(umap.UMAP.__init__)
        if "low_memory" in umap_sig.parameters and "low_memory" not in umap_params:
            umap_params["low_memory"] = True
        if "n_jobs" in umap_sig.parameters and "n_jobs" not in umap_params:
            umap_params["n_jobs"] = -1

        n_components = int(umap_params.pop("n_components", 2))
        model = umap.UMAP(n_components=n_components, **umap_params)
        X_2d = model.fit_transform(X_work)

    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            "Choose from {'pca', 'tsne', 'umap'}."
        )

    return X_2d
