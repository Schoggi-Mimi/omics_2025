"""Dimensionality reduction algorithms for proteomics data."""

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def apply_pca(
    data: Union[np.ndarray, pd.DataFrame],
    n_components: Union[int, float] = 2,
    return_model: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, PCA]]:
    """Apply Principal Component Analysis (PCA) for dimensionality reduction.
    
    PCA is a linear dimensionality reduction technique that identifies
    orthogonal directions of maximum variance in the data.
    
    Args:
        data: Input data matrix (samples x features)
        n_components: Number of components to keep. If int, exact number.
                     If float (0.0-1.0), select components to retain that
                     much variance.
        return_model: If True, also return the fitted PCA model
        
    Returns:
        Transformed data (samples x n_components), or tuple with model if return_model=True
        
    Examples:
        >>> import numpy as np
        >>> data = np.random.randn(100, 50)
        >>> reduced = apply_pca(data, n_components=2)
        >>> print(reduced.shape)  # (100, 2)
    """
    # Convert DataFrame to numpy if needed
    is_dataframe = isinstance(data, pd.DataFrame)
    if is_dataframe:
        sample_names = data.index
        data_array = data.values
    else:
        data_array = data
        sample_names = None
    
    # Apply PCA
    pca = PCA(n_components=n_components, random_state=42)
    transformed = pca.fit_transform(data_array)
    
    # Optionally convert back to DataFrame
    if is_dataframe and sample_names is not None:
        component_names = [f"PC{i+1}" for i in range(transformed.shape[1])]
        transformed = pd.DataFrame(
            transformed,
            index=sample_names,
            columns=component_names
        )
    
    if return_model:
        return transformed, pca
    return transformed


def apply_umap(
    data: Union[np.ndarray, pd.DataFrame],
    n_components: int = 2,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    metric: str = "cosine", # common for proteomics
    random_state: Optional[int] = 42,
    return_model: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, object]]:
    """Apply UMAP (Uniform Manifold Approximation and Projection).
    
    UMAP is a non-linear dimensionality reduction technique that preserves
    both local and global data structure, often revealing complex patterns.
    
    Args:
        data: Input data matrix (samples x features)
        n_components: Number of dimensions in the embedding space
        n_neighbors: Size of local neighborhood for manifold approximation
                    (larger = more global structure, smaller = more local)
        min_dist: Minimum distance between points in embedding space
                 (smaller = more clustered, larger = more dispersed)
        metric: Distance metric to use (euclidean, manhattan, cosine, etc.)
        random_state: Random seed for reproducibility
        return_model: If True, also return the fitted UMAP model
        
    Returns:
        Transformed data (samples x n_components), or tuple with model if return_model=True
        
    Raises:
        ImportError: If umap-learn is not installed
        
    Examples:
        >>> import numpy as np
        >>> data = np.random.randn(100, 50)
        >>> reduced = apply_umap(data, n_components=2)
        >>> print(reduced.shape)  # (100, 2)
    """
    if not UMAP_AVAILABLE:
        raise ImportError(
            "umap-learn is not installed. "
            "Install it with: pip install umap-learn"
        )
    np.random.seed(random_state)
    # Convert DataFrame to numpy if needed
    is_dataframe = isinstance(data, pd.DataFrame)
    if is_dataframe:
        sample_names = data.index
        data_array = data.values
    else:
        data_array = data
        sample_names = None
    
    # Apply UMAP
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        low_memory=True # for large datasets
    )
    transformed = reducer.fit_transform(data_array)
    
    # Optionally convert back to DataFrame
    if is_dataframe and sample_names is not None:
        component_names = [f"UMAP{i+1}" for i in range(transformed.shape[1])]
        transformed = pd.DataFrame(
            transformed,
            index=sample_names,
            columns=component_names
        )
    
    if return_model:
        return transformed, reducer
    return transformed



def pca_elbow(explained_variance_ratio):
    """
    Detect elbow in PCA scree plot using the maximum distance to chord method.
    This method is specifically for PCA, not clustering inertia.

    Args:
        explained_variance_ratio: array-like list of variance explained per PC.

    Returns:
        int: elbow point (1-based PC index)
    """
    y = np.array(explained_variance_ratio)
    x = np.arange(1, len(y) + 1)

    # Normalize
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    # Line from first to last PC
    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])
    line_vec = p2 - p1

    # Distances
    distances = np.array([
        np.abs(np.cross(line_vec, np.array([x_norm[i], y_norm[i]]) - p1)) /
        np.linalg.norm(line_vec)
        for i in range(len(x_norm))
    ])

    elbow_idx = int(np.argmax(distances))
    return elbow_idx + 1

def pc_names(df: pd.DataFrame, prefix: str = "PC") -> pd.DataFrame:
    """Ensure PCA columns are named PC1..PCn (without changing values/order)."""
    out = df.copy()
    out.columns = [f"{prefix}{i+1}" for i in range(out.shape[1])]
    return out