"""Visualization functions for proteomics data."""

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure


def plot_pca_variance(pca_model, max_components=20, figsize=(6,4)):
    """
    Plot explained variance for each PCA component.
    
    Args:
        pca_model: Fitted PCA model
        max_components: Number of components to display
        figsize: Figure size
    """
    var_ratio = pca_model.explained_variance_ratio_[:max_components]
    components = np.arange(1, len(var_ratio) + 1)

    plt.figure(figsize=figsize)
    plt.bar(components, var_ratio * 100, color='steelblue')
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance (%)")
    plt.title("PCA Explained Variance")
    plt.show()

def plot_pca_cumulative_variance(pca_model, threshold=0.80, figsize=(6,4)):
    """
    Plot cumulative explained variance and threshold line.
    
    Args:
        pca_model: Fitted PCA model
        threshold: Cumulative variance threshold (default 0.80 = 80%)
        figsize: Figure size
    """
    var_ratio = pca_model.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio) * 100
    components = np.arange(1, len(cum_var) + 1)

    plt.figure(figsize=figsize)
    plt.plot(components, cum_var, marker='o')
    plt.axhline(threshold * 100, color='red', linestyle='--')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance (%)")
    plt.title("Cumulative PCA Explained Variance")
    plt.grid(True)
    plt.show()

def plot_silhouette_scores(scores: dict):
    ks = list(scores.keys())
    vals = list(scores.values())

    plt.figure(figsize=(6, 4))
    plt.plot(ks, vals, marker='o', linewidth=2, markersize=8)
    plt.xticks(ks)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Analysis for Optimal k")
    plt.grid(alpha=0.3)
    plt.show()

def plot_elbow(inertias: dict):
    ks = list(inertias.keys())
    vals = list(inertias.values())

    plt.figure(figsize=(6, 4))
    plt.plot(ks, vals, marker='o', linewidth=2, markersize=8)
    plt.xticks(ks)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia (WCSS)")
    plt.title("Elbow Method for Optimal k")
    plt.grid(alpha=0.3)
    plt.show()

def _choose_cmap(labels):
    n_clusters = len(np.unique(labels))
    if n_clusters == 2:
        return ListedColormap(["#1f77b4", "#ff7f0e"])  # blue & orange
    elif n_clusters == 3:
        return ListedColormap(["#1f77b4", "#ff7f0e", "#2ca02c"])  # blue/orange/green
    else:
        return "tab10"
    
def plot_pca(
    pca_data: Union[np.ndarray, pd.DataFrame],
    labels: Optional[np.ndarray] = None,
    components: Tuple[int, int] = (0, 1),
    title: str = "PCA Plot",
    figsize: Tuple[float, float] = (10, 8),
    alpha: float = 0.7,
    cmap: str = "tab10",
    save_path: Optional[str] = None,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """Plot PCA results in 2D.
    
    Args:
        pca_data: PCA-transformed data (samples x components)
        labels: Optional cluster labels or categories for coloring points
        components: Which components to plot (default: first two)
        title: Plot title
        figsize: Figure size (width, height)
        alpha: Point transparency
        cmap: Colormap for labels
        save_path: If provided, save figure to this path
        ax: Optional matplotlib axes to plot on
        
    Returns:
        Tuple of (figure, axes)
        
    Examples:
        >>> import numpy as np
        >>> from bladder_proteomics.dimensionality_reduction import apply_pca
        >>> data = np.random.randn(100, 50)
        >>> pca_result = apply_pca(data, n_components=2)
        >>> fig, ax = plot_pca(pca_result)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Extract components
    if isinstance(pca_data, pd.DataFrame):
        x = pca_data.iloc[:, components[0]].values
        y = pca_data.iloc[:, components[1]].values
        xlabel = pca_data.columns[components[0]]
        ylabel = pca_data.columns[components[1]]
    else:
        x = pca_data[:, components[0]]
        y = pca_data[:, components[1]]
        xlabel = f"PC{components[0] + 1}"
        ylabel = f"PC{components[1] + 1}"
    
    # Plot
    if labels is not None:
        cmap_used = _choose_cmap(labels)
        scatter = ax.scatter(x, y, c=labels, alpha=alpha, cmap=cmap_used, s=50)
        plt.colorbar(scatter, ax=ax, label="Cluster")
    else:
        ax.scatter(x, y, alpha=alpha, s=50)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig, ax


def plot_umap(
    umap_data: Union[np.ndarray, pd.DataFrame],
    labels: Optional[np.ndarray] = None,
    components: Tuple[int, int] = (0, 1),
    title: str = "UMAP Plot",
    figsize: Tuple[float, float] = (10, 8),
    alpha: float = 0.7,
    cmap: str = "tab10",
    save_path: Optional[str] = None,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """Plot UMAP results in 2D.
    
    Args:
        umap_data: UMAP-transformed data (samples x components)
        labels: Optional cluster labels or categories for coloring points
        components: Which components to plot (default: first two)
        title: Plot title
        figsize: Figure size (width, height)
        alpha: Point transparency
        cmap: Colormap for labels
        save_path: If provided, save figure to this path
        ax: Optional matplotlib axes to plot on
        
    Returns:
        Tuple of (figure, axes)
        
    Examples:
        >>> import numpy as np
        >>> from bladder_proteomics.dimensionality_reduction import apply_umap
        >>> data = np.random.randn(100, 50)
        >>> umap_result = apply_umap(data, n_components=2)
        >>> fig, ax = plot_umap(umap_result)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Extract components
    if isinstance(umap_data, pd.DataFrame):
        x = umap_data.iloc[:, components[0]].values
        y = umap_data.iloc[:, components[1]].values
        xlabel = umap_data.columns[components[0]]
        ylabel = umap_data.columns[components[1]]
    else:
        x = umap_data[:, components[0]]
        y = umap_data[:, components[1]]
        xlabel = f"UMAP{components[0] + 1}"
        ylabel = f"UMAP{components[1] + 1}"
    
    # Plot
    if labels is not None:
        cmap_used = _choose_cmap(labels)
        scatter = ax.scatter(x, y, c=labels, alpha=alpha, cmap=cmap_used, s=50)
        plt.colorbar(scatter, ax=ax, label="Cluster")
    else:
        ax.scatter(x, y, alpha=alpha, s=50)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig, ax


def plot_heatmap(
    data: Union[np.ndarray, pd.DataFrame],
    row_labels: Optional[List[str]] = None,
    col_labels: Optional[List[str]] = None,
    title: str = "Heatmap",
    figsize: Tuple[float, float] = (12, 10),
    cmap: str = "RdBu_r",
    center: Optional[float] = 0,
    robust: bool = True,
    cbar_label: str = "Expression",
    save_path: Optional[str] = None,
    cluster_rows: bool = False,
    cluster_cols: bool = False,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """Create a heatmap of proteomics data.
    
    Args:
        data: Data matrix to visualize
        row_labels: Labels for rows (samples)
        col_labels: Labels for columns (features)
        title: Plot title
        figsize: Figure size (width, height)
        cmap: Colormap
        center: Value to center colormap at
        robust: If True, use robust quantiles for colormap limits
        cbar_label: Label for colorbar
        save_path: If provided, save figure to this path
        cluster_rows: If True, perform hierarchical clustering on rows
        cluster_cols: If True, perform hierarchical clustering on columns
        ax: Optional matplotlib axes to plot on
        
    Returns:
        Tuple of (figure, axes)
        
    Examples:
        >>> import numpy as np
        >>> data = np.random.randn(20, 50)
        >>> fig, ax = plot_heatmap(data, cluster_rows=True, cluster_cols=True)
    """
    if isinstance(data, pd.DataFrame):
        data_array = data.values
        if row_labels is None:
            row_labels = data.index
        if col_labels is None:
            col_labels = data.columns
    else:
        data_array = data
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Create heatmap
    if cluster_rows or cluster_cols:
        # Use seaborn's clustermap for hierarchical clustering
        # This creates its own figure, so we need to handle it differently
        g = sns.clustermap(
            data_array,
            cmap=cmap,
            center=center,
            robust=robust,
            figsize=figsize,
            row_cluster=cluster_rows,
            col_cluster=cluster_cols,
            cbar_kws={"label": cbar_label}
        )
        g.fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        
        if save_path:
            g.savefig(save_path, dpi=300, bbox_inches="tight")
        
        return g.fig, g.ax_heatmap
    else:
        # Regular heatmap
        # Apply center and robust parameters
        vmin, vmax = None, None
        if robust:
            vmin, vmax = np.percentile(data_array, [2, 98])
        
        im = ax.imshow(data_array, cmap=cmap, aspect="auto", 
                      vmin=vmin, vmax=vmax)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(cbar_label, rotation=270, labelpad=20)
        
        # Labels
        if row_labels is not None:
            ax.set_yticks(np.arange(len(row_labels)))
            ax.set_yticklabels(row_labels)
        
        if col_labels is not None:
            ax.set_xticks(np.arange(len(col_labels)))
            ax.set_xticklabels(col_labels, rotation=90)
        
        ax.set_title(title, fontsize=14, fontweight="bold")
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        
        return fig, ax


def plot_clusters(
    data: Union[np.ndarray, pd.DataFrame],
    labels: np.ndarray,
    method: str = "pca",
    title: str = "Cluster Visualization",
    figsize: Tuple[float, float] = (10, 8),
    alpha: float = 0.7,
    cmap: str = "tab10",
    save_path: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """Visualize clusters in reduced dimensional space.
    
    Args:
        data: Original data matrix (samples x features)
        labels: Cluster labels for each sample
        method: Dimensionality reduction method ('pca' or 'umap')
        title: Plot title
        figsize: Figure size (width, height)
        alpha: Point transparency
        cmap: Colormap for clusters
        save_path: If provided, save figure to this path
        
    Returns:
        Tuple of (figure, axes)
        
    Examples:
        >>> import numpy as np
        >>> from bladder_proteomics.clustering import kmeans_cluster
        >>> data = np.random.randn(100, 50)
        >>> labels = kmeans_cluster(data, n_clusters=3)
        >>> fig, ax = plot_clusters(data, labels, method='pca')
    """
    # Import here to avoid circular imports
    from ..dimensionality_reduction import apply_pca, apply_umap

    # Reduce dimensionality
    if method.lower() == "pca":
        reduced = apply_pca(data, n_components=2)
        plot_func = plot_pca
    elif method.lower() == "umap":
        reduced = apply_umap(data, n_components=2)
        plot_func = plot_umap
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca' or 'umap'")
    
    # Plot
    fig, ax = plot_func(
        reduced,
        labels=labels,
        title=title,
        figsize=figsize,
        alpha=alpha,
        cmap=cmap,
        save_path=save_path
    )
    
    return fig, ax
