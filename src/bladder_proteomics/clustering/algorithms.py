"""Clustering algorithms for proteomics data."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from typing import Union, Tuple, Optional


def kmeans_cluster(
    data: Union[np.ndarray, pd.DataFrame],
    n_clusters: int = 3,
    random_state: Optional[int] = 42,
    n_init: int = 10,
    max_iter: int = 300,
    return_model: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, KMeans]]:
    """Apply KMeans clustering algorithm.
    
    KMeans partitions data into k clusters by minimizing within-cluster
    sum of squared distances to cluster centers.
    
    Args:
        data: Input data matrix (samples x features)
        n_clusters: Number of clusters to form
        random_state: Random seed for reproducibility
        n_init: Number of times the algorithm will run with different centroid seeds
        max_iter: Maximum number of iterations per run
        return_model: If True, also return the fitted model
        
    Returns:
        Cluster labels for each sample, or tuple with model if return_model=True
        
    Examples:
        >>> import numpy as np
        >>> data = np.random.randn(100, 50)
        >>> labels = kmeans_cluster(data, n_clusters=3)
        >>> print(labels.shape)  # (100,)
    """
    # Convert DataFrame to numpy if needed
    data_array = data.values if isinstance(data, pd.DataFrame) else data
    
    # Apply KMeans
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter
    )
    labels = kmeans.fit_predict(data_array)
    
    if return_model:
        return labels, kmeans
    return labels


def gmm_cluster(
    data: Union[np.ndarray, pd.DataFrame],
    n_components: int = 3,
    covariance_type: str = "full",
    random_state: Optional[int] = 42,
    n_init: int = 5,
    max_iter: int = 100,
    return_model: bool = False,
    return_probabilities: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, GaussianMixture], Tuple[np.ndarray, np.ndarray]]:
    """Apply Gaussian Mixture Model (GMM) clustering.
    
    GMM is a probabilistic model that assumes data is generated from
    a mixture of Gaussian distributions with unknown parameters.
    
    Args:
        data: Input data matrix (samples x features)
        n_components: Number of mixture components (clusters)
        covariance_type: Type of covariance parameters
                        ('full', 'tied', 'diag', 'spherical')
        random_state: Random seed for reproducibility
        n_init: Number of initializations to perform
        max_iter: Maximum number of EM iterations
        return_model: If True, also return the fitted model
        return_probabilities: If True, return probability matrix instead of hard labels
        
    Returns:
        Cluster labels (or probabilities), optionally with model
        
    Examples:
        >>> import numpy as np
        >>> data = np.random.randn(100, 50)
        >>> labels = gmm_cluster(data, n_components=3)
        >>> print(labels.shape)  # (100,)
    """
    # Convert DataFrame to numpy if needed
    data_array = data.values if isinstance(data, pd.DataFrame) else data
    
    # Apply GMM
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter
    )
    gmm.fit(data_array)
    
    if return_probabilities:
        result = gmm.predict_proba(data_array)
    else:
        result = gmm.predict(data_array)
    
    if return_model:
        return result, gmm
    return result


def agglomerative_cluster(
    data: Union[np.ndarray, pd.DataFrame],
    n_clusters: int = 3,
    linkage: str = "ward",
    metric: str = "euclidean",
    return_model: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, AgglomerativeClustering]]:
    """Apply Agglomerative (Hierarchical) clustering.
    
    Hierarchical clustering builds a tree of clusters by iteratively
    merging or splitting clusters based on a linkage criterion.
    
    Args:
        data: Input data matrix (samples x features)
        n_clusters: Number of clusters to form
        linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
        metric: Distance metric to use (e.g., 'euclidean', 'manhattan', 'cosine')
                Note: 'ward' linkage only works with 'euclidean' metric
        return_model: If True, also return the fitted model
        
    Returns:
        Cluster labels for each sample, or tuple with model if return_model=True
        
    Examples:
        >>> import numpy as np
        >>> data = np.random.randn(100, 50)
        >>> labels = agglomerative_cluster(data, n_clusters=3)
        >>> print(labels.shape)  # (100,)
    """
    # Convert DataFrame to numpy if needed
    data_array = data.values if isinstance(data, pd.DataFrame) else data
    
    # Validate linkage and metric combination
    if linkage == "ward" and metric != "euclidean":
        raise ValueError("Ward linkage only works with euclidean metric")
    
    # Apply Agglomerative Clustering
    agg = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
        metric=metric if linkage != "ward" else "euclidean"
    )
    labels = agg.fit_predict(data_array)
    
    if return_model:
        return labels, agg
    return labels

def compute_cluster_table(pca_df, n_pc, k):
    """Compute cluster counts and percentages using the same logic as the user's code."""
    
    X = pca_df.iloc[:, :n_pc]
    labels = kmeans_cluster(X, n_clusters=k)  # unchanged logic

    cluster_ids, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    df_cluster = pd.DataFrame({
        "cluster": cluster_ids,
        "count": counts,
        "percentage": counts / total * 100
    }).sort_values("count", ascending=False)

    return df_cluster, labels
