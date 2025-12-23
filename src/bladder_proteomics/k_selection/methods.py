"""Methods for determining optimal number of clusters."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, silhouette_samples,
                             silhouette_score)


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



def _fit_kmeans(X: np.ndarray, k: int, seed: int = 42, n_init: int = 50) -> Tuple[KMeans, np.ndarray]:
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
    labels = km.fit_predict(X)
    return km, labels


def align_labels_by_pc1(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Deterministic label alignment: sort clusters by mean PC1 and relabel to 0..k-1.
    Helpful to keep plots/tables consistent across runs.
    """
    unique = np.unique(labels)
    means = {lab: float(np.mean(X[labels == lab, 0])) for lab in unique}
    order = sorted(unique, key=lambda lab: means[lab])
    mapping = {old: new for new, old in enumerate(order)}
    return np.vectorize(mapping.get)(labels)


def compute_inertia_curve(X: np.ndarray, k_range: Iterable[int], seed: int = 42, n_init: int = 50) -> Dict[int, float]:
    inertias = {}
    for k in k_range:
        km, _ = _fit_kmeans(X, k=k, seed=seed, n_init=n_init)
        inertias[int(k)] = float(km.inertia_)
    return inertias


def compute_silhouette_curve(X: np.ndarray, k_range: Iterable[int], seed: int = 42, n_init: int = 50,
                            align_labels: bool = False) -> Dict[int, float]:
    sil = {}
    for k in k_range:
        _, labels = _fit_kmeans(X, k=int(k), seed=seed, n_init=n_init)
        if align_labels:
            labels = align_labels_by_pc1(X, labels)
        sil[int(k)] = float(silhouette_score(X, labels))
    return sil


def cluster_size_dict(labels: np.ndarray) -> Dict[int, int]:
    vc = pd.Series(labels).value_counts().sort_index()
    return {int(k): int(v) for k, v in vc.items()}

def passes_min_cluster_size(labels: np.ndarray, min_size: int = 10) -> bool:
    counts = pd.Series(labels).value_counts()
    return bool((counts >= min_size).all())


def ari_stability_init(
        X: pd.DataFrame,
        k: int,
        seeds: Iterable[int] = range(10),
        n_init: int = 1,   # IMPORTANT: initialization sensitivity -> use n_init=1
    ) -> Tuple[float, float]:
    Xv = X.values if hasattr(X, "values") else X
    all_labels = []
    for sd in seeds:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=sd)
        all_labels.append(km.fit_predict(Xv))

    aris = []
    for i in range(len(all_labels)):
        for j in range(i + 1, len(all_labels)):
            aris.append(adjusted_rand_score(all_labels[i], all_labels[j]))

    return float(np.mean(aris)), float(np.std(aris))


def evaluate_k_selection_one(X_df: pd.DataFrame, k_max: int = 20) -> Dict:
    """
    Returns inertia curve (k=1..k_max), silhouette curve (k=2..k_max),
    elbow_k (optional), best_k_sil, and sizes at best_k_sil.
    """
    X = X_df.values

    sil = silhouette_analysis(X, range(2, k_max + 1))
    inertia = elbow_method(X, range(1, k_max + 1))
    elbow = calculate_elbow_point(inertia)
    best_k = max(sil, key=sil.get)

    return {
        "inertia": inertia,
        "silhouette": sil,
        "elbow_k": elbow,
        "best_k_sil": int(best_k),
        "best_sil": float(sil[best_k]),
    }

def pc_sensitivity_best_k(pca_df: pd.DataFrame, pc_grid: List[int], k_max: int = 20,
                          seed: int = 42, n_init: int = 50, align_labels: bool = True) -> pd.DataFrame:
    rows = []
    for npc in pc_grid:
        X = pca_df.iloc[:, :npc].values
        sil = compute_silhouette_curve(X, range(2, k_max + 1), seed=seed, n_init=n_init, align_labels=align_labels)
        best_k = max(sil, key=sil.get)

        _, lab = _fit_kmeans(X, k=int(best_k), seed=seed, n_init=n_init)
        if align_labels:
            lab = align_labels_by_pc1(X, lab)

        rows.append({
            "n_pc": int(npc),
            "best_k_sil": int(best_k),
            "best_sil": float(sil[best_k]),
            "sizes": cluster_size_dict(lab),
        })
    return pd.DataFrame(rows)

def gap_statistic(X, k_max=20, B=50, random_state=42, n_init=10):
    """
    Compute Gap Statistic for k=1..k_max and return optimal k.
    Args:
        X: DataFrame or array-like, shape (n_samples, n_features)
        k_max: Maximum number of clusters to evaluate
        B: Number of reference datasets to generate
        random_state: Random seed for reproducibility
        n_init: Number of initializations for KMeans
    Returns:
        gap: Array of gap values for each k
        se: Standard error of gap values
        wk: Within-cluster dispersion for each k
        optimal_k: Optimal number of clusters determined by the gap statistic
    """
    rng = np.random.default_rng(random_state)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    wk = np.zeros(k_max)
    gap = np.zeros(k_max)
    se = np.zeros(k_max)

    for k in range(1, k_max + 1):
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        km.fit(X)
        wk[k - 1] = km.inertia_

        ref_wk = np.zeros(B)
        for b in range(B):
            X_ref = rng.uniform(mins, maxs, size=X.shape)
            km_ref = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
            km_ref.fit(X_ref)
            ref_wk[b] = km_ref.inertia_

        gap[k - 1] = ref_wk.mean() - wk[k - 1]
        se[k - 1] = ref_wk.std(ddof=1) / np.sqrt(B)

    optimal_k = k_max
    for k in range(1, k_max):
        if gap[k - 1] >= gap[k] - se[k]:
            optimal_k = k
            break

    return gap, se, wk, optimal_k

