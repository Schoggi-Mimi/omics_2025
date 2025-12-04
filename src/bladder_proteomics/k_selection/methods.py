"""Methods for determining optimal number of clusters."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from typing import Union, Dict, List, Optional, Tuple


def silhouette_analysis(
    data: Union[np.ndarray, pd.DataFrame],
    k_range: Optional[List[int]] = None,
    random_state: Optional[int] = 42,
    return_samples: bool = False
) -> Union[Dict[int, float], Tuple[Dict[int, float], Dict[int, np.ndarray]]]:
    """Perform silhouette analysis to find optimal number of clusters.
    
    The silhouette score measures how similar a sample is to its own cluster
    compared to other clusters. Values range from -1 to 1, where higher is better.
    
    Args:
        data: Input data matrix (samples x features)
        k_range: Range of k values to test (default: [2, 3, 4, 5, 6, 7, 8, 9, 10])
        random_state: Random seed for reproducibility
        return_samples: If True, also return per-sample silhouette scores
        
    Returns:
        Dictionary mapping k to average silhouette score,
        optionally with per-sample scores
        
    Examples:
        >>> import numpy as np
        >>> data = np.random.randn(100, 50)
        >>> scores = silhouette_analysis(data)
        >>> best_k = max(scores, key=scores.get)
        >>> print(f"Best k: {best_k}, Score: {scores[best_k]:.3f}")
    """
    if k_range is None:
        k_range = list(range(2, 11))
    
    # Convert DataFrame to numpy if needed
    data_array = data.values if isinstance(data, pd.DataFrame) else data
    
    scores = {}
    sample_scores = {}
    
    for k in k_range:
        # Perform clustering
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(data_array)
        
        # Calculate silhouette score
        avg_score = silhouette_score(data_array, labels)
        scores[k] = avg_score
        
        if return_samples:
            sample_scores[k] = silhouette_samples(data_array, labels)
    
    if return_samples:
        return scores, sample_scores
    return scores


def elbow_method(
    data: Union[np.ndarray, pd.DataFrame],
    k_range: Optional[List[int]] = None,
    random_state: Optional[int] = 42,
    return_models: bool = False
) -> Union[Dict[int, float], Tuple[Dict[int, float], Dict[int, KMeans]]]:
    """Apply the elbow method to find optimal number of clusters.
    
    The elbow method plots the within-cluster sum of squares (inertia)
    against the number of clusters. The "elbow" point suggests optimal k.
    
    Args:
        data: Input data matrix (samples x features)
        k_range: Range of k values to test (default: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        random_state: Random seed for reproducibility
        return_models: If True, also return fitted models
        
    Returns:
        Dictionary mapping k to inertia (within-cluster sum of squares),
        optionally with fitted models
        
    Examples:
        >>> import numpy as np
        >>> data = np.random.randn(100, 50)
        >>> inertias = elbow_method(data)
        >>> # Plot to find elbow visually
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(list(inertias.keys()), list(inertias.values()))
        >>> plt.xlabel('Number of clusters')
        >>> plt.ylabel('Inertia')
        >>> plt.show()
    """
    if k_range is None:
        k_range = list(range(1, 11))
    
    # Convert DataFrame to numpy if needed
    data_array = data.values if isinstance(data, pd.DataFrame) else data
    
    inertias = {}
    models = {}
    
    for k in k_range:
        # Perform clustering
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(data_array)
        
        # Store inertia
        inertias[k] = kmeans.inertia_
        
        if return_models:
            models[k] = kmeans
    
    if return_models:
        return inertias, models
    return inertias


def calculate_elbow_point(inertias: Dict[int, float]) -> int:
    """Calculate the elbow point from inertia values.
    
    Uses the "knee" detection method to automatically identify
    the elbow point in the inertia curve.
    
    Args:
        inertias: Dictionary mapping k to inertia values
        
    Returns:
        Optimal k value (elbow point)
        
    Examples:
        >>> inertias = {1: 1000, 2: 500, 3: 300, 4: 250, 5: 230}
        >>> optimal_k = calculate_elbow_point(inertias)
    """
    if len(inertias) < 3:
        raise ValueError("Need at least 3 points to calculate elbow")
    
    # Sort by k
    k_values = sorted(inertias.keys())
    inertia_values = [inertias[k] for k in k_values]
    
    # Normalize to 0-1 range
    k_norm = np.array(k_values)
    k_norm = (k_norm - k_norm.min()) / (k_norm.max() - k_norm.min())
    
    inertia_norm = np.array(inertia_values)
    inertia_norm = (inertia_norm - inertia_norm.min()) / (inertia_norm.max() - inertia_norm.min())
    
    # Calculate distance from each point to the line between first and last points
    # Line from (k_norm[0], inertia_norm[0]) to (k_norm[-1], inertia_norm[-1])
    line_vec = np.array([k_norm[-1] - k_norm[0], inertia_norm[-1] - inertia_norm[0]])
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    
    max_dist = 0
    elbow_idx = 0
    
    for i in range(1, len(k_values) - 1):
        point_vec = np.array([k_norm[i] - k_norm[0], inertia_norm[i] - inertia_norm[0]])
        # Distance from point to line
        dist = np.abs(np.cross(line_vec_norm, point_vec))
        
        if dist > max_dist:
            max_dist = dist
            elbow_idx = i
    
    return k_values[elbow_idx]
