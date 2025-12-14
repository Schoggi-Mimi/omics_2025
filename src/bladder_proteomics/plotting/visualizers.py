"""Visualization functions for proteomics data."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min, silhouette_score

from ..clustering import compute_cluster_table


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

def plot_silhouette_scores(scores: dict, ax: Optional[Axes] = None, title: str = "Silhouette Analysis for Optimal k"):
    ks = list(scores.keys())
    vals = list(scores.values())

    # If no axis is provided, create a new figure and axis
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        created_fig = True

    ax.plot(ks, vals, marker='o', linewidth=2, markersize=8)
    ax.set_xticks(ks)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title(title)
    ax.grid(alpha=0.3)

    # Only show if we created the figure inside
    if created_fig:
        plt.show()

def plot_elbow(inertias: dict, ax: Optional[Axes] = None, title: str = "Elbow Method for Optimal k"):
    ks = list(inertias.keys())
    vals = list(inertias.values())

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        created_fig = True

    ax.plot(ks, vals, marker='o', linewidth=2, markersize=8)
    ax.set_xticks(ks)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia (WCSS)")
    ax.set_title(title)
    ax.grid(alpha=0.3)

    if created_fig:
        plt.show()


# def _choose_cmap(labels):
#     n_clusters = len(np.unique(labels))

#     # For small number of clusters, use high-quality colors.
#     if n_clusters <= 10:
#         cmap = plt.cm.get_cmap("tab10", n_clusters)
#     elif n_clusters <= 20:
#         cmap = plt.cm.get_cmap("tab20", n_clusters)
#     else:
#         # fallback: continuous color spectrum
#         cmap = plt.cm.get_cmap("hsv", n_clusters)

#     return cmap

CLUSTER_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]

def _choose_fixed_cmap(labels):
    n_clusters = len(np.unique(labels))
    return ListedColormap(CLUSTER_COLORS[:n_clusters])

    
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
        cmap_used = _choose_fixed_cmap(labels)
        scatter = ax.scatter(x, y, c=labels, cmap=cmap_used, alpha=alpha, s=50, vmin=0, vmax=len(np.unique(labels)) - 1)
        legend_elements = [
            Line2D(
                [0], [0],
                marker='o',
                color='w',
                label=f'Cluster {i}',
                markerfacecolor=CLUSTER_COLORS[i],
                markersize=8
            )
            for i in np.unique(labels)
        ]
        ax.legend(handles=legend_elements, title="Cluster")
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
        # cmap_used = _choose_cmap(labels)
        cmap_used = _choose_fixed_cmap(labels)
        scatter = ax.scatter(x, y, c=labels, cmap=cmap_used, alpha=alpha, s=50, vmin=0, vmax=len(np.unique(labels)) - 1)
        legend_elements = [
            Line2D(
                [0], [0],
                marker='o',
                color='w',
                label=f'Cluster {i}',
                markerfacecolor=CLUSTER_COLORS[i],
                markersize=8
            )
            for i in np.unique(labels)
        ]
        ax.legend(handles=legend_elements, title="Cluster")
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


def plot_cluster_distributions(
    pca_df,
    n_pc_list,
    k,
    figsize=(5, 5),
    cmap="tab10"
):
    """
    Plot cluster-size distributions for one or multiple PCA dimensionalities.
    X-axis = rank (clusters sorted by size).
    Each rank shows side-by-side bars for different n_pc values.

    Args:
        pca_df: PCA-transformed dataframe (samples × PCs)
        n_pc_list: int or list of ints (e.g. 4 or [4,5] or [3,4,5])
        k: number of clusters for k-means
        figsize: size of the output figure
        cmap: matplotlib colormap (must have >= len(n_pc_list) colors)
    """

    # -----------------------------
    # Ensure list form
    # -----------------------------
    if isinstance(n_pc_list, int):
        n_pc_list = [n_pc_list]

    n_versions = len(n_pc_list)

    # Color map for different PC choices
    colors = plt.cm.get_cmap(cmap, n_versions)

    # -----------------------------
    # Compute cluster distributions
    # -----------------------------
    tables = {}
    max_pct = 0
    max_clusters = 0

    for idx, npc in enumerate(n_pc_list):
        df_cluster, _ = compute_cluster_table(pca_df, npc, k)
        df_cluster = df_cluster.sort_values("percentage", ascending=False).reset_index(drop=True)
        df_cluster["rank"] = np.arange(1, len(df_cluster) + 1)

        tables[npc] = df_cluster
        max_pct = max(max_pct, df_cluster["percentage"].max())
        max_clusters = max(max_clusters, len(df_cluster))

    # -----------------------------
    # Prepare x-axis positions
    # -----------------------------
    ranks = np.arange(1, max_clusters + 1)
    x = np.arange(len(ranks))

    bar_width = 0.8 / n_versions  # ensure bars fit within 1 unit width

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=figsize)

    for i, npc in enumerate(n_pc_list):
        df_cluster = tables[npc]

        # Align percentages by rank (pad with zeros if needed)
        y = np.zeros(max_clusters)
        y[: len(df_cluster)] = df_cluster["percentage"]

        plt.bar(
            x + (i - (n_versions - 1)/2) * bar_width,
            y,
            width=bar_width,
            label=f"PC={npc}",
            alpha=0.85,
            color=colors(i)
        )

    # -----------------------------
    # Decorations
    # -----------------------------
    plt.xticks(x, ranks)
    plt.xlabel("Cluster Rank (largest → smallest)")
    plt.ylabel("Percentage (%)")
    plt.title(f"Cluster Size Distribution (k={k})")
    plt.ylim(0, max_pct * 1.15)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(title="n_pc")

    plt.tight_layout()
    plt.show()


def _palette_for_labels(labels: pd.Series) -> dict:
    """Map each cluster label to a fixed color."""
    unique = sorted(pd.unique(labels))
    return {lab: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, lab in enumerate(unique)}

def plot_2d_embedding(
    emb2d: pd.DataFrame,
    labels: pd.Series,
    title: str,
    ax: plt.Axes | None = None,
    x: str | None = None,
    y: str | None = None,
) -> plt.Axes:
    """2D scatter with legend (no colorbar), using fixed colors per cluster label."""
    if ax is None:
        ax = plt.gca()
    x = x or emb2d.columns[0]
    y = y or emb2d.columns[1]

    dfp = emb2d[[x, y]].copy()
    dfp["Cluster"] = labels.astype(int).values

    palette = _palette_for_labels(dfp["Cluster"])
    sns.scatterplot(
        data=dfp,
        x=x,
        y=y,
        hue="Cluster",
        palette=palette,
        s=35,
        linewidth=0.2,
        edgecolor="white",
        ax=ax,
    )
    ax.set_title(title)
    ax.legend(title="Cluster", loc="best", frameon=True)
    return ax

def plot_cluster_sizes(labels, title="Cluster sizes", ax=None, sort_by="label"):
    """
    Horizontal bar chart with count and percentage.
    sort_by: 'label' or 'size'
    """
    if ax is None:
        ax = plt.gca()

    counts = labels.value_counts()
    if sort_by == "label":
        counts = counts.sort_index()
    elif sort_by == "size":
        counts = counts.sort_values(ascending=True)

    palette_map = _palette_for_labels(labels)
    colors = [palette_map[k] for k in counts.index]

    y = np.arange(len(counts))
    ax.barh(y, counts.values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels([str(k) for k in counts.index])
    ax.set_xlabel("Number of patients")
    ax.set_title(title)

    total = counts.sum()
    for i, (lab, val) in enumerate(counts.items()):
        ax.text(val + max(1, 0.01 * total), i, f"{val} ({val/total:.1%})", va="center")

    ax.set_xlim(0, counts.max() * 1.15)
    return ax


# ---------- Global style / palette ----------

PALETTE = {
    "blue": "#4C72B0",
    "orange": "#DD8452",
    "green": "#55A868",
    "red": "#C44E52",
    "gray": "#7A7A7A",
    "black": "#000000",
}

DEFAULT_CLUSTER_PALETTE = sns.color_palette("Set2")


def set_plot_style(dpi: int = 120) -> None:
    """
    Call once per notebook (early).
    Ensures consistent seaborn theme and matplotlib defaults across all plots.
    """
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 10

def get_cluster_colors(labels: Sequence[int]) -> Dict[int, str]:
    """Map cluster labels -> consistent colors (Set2)."""
    uniq = np.sort(pd.unique(pd.Series(labels)))
    colors = {}
    for i, lab in enumerate(uniq):
        colors[int(lab)] = DEFAULT_CLUSTER_PALETTE[i % len(DEFAULT_CLUSTER_PALETTE)]
    return colors


def plot_log10_raw_distribution(
    df_raw: pd.DataFrame,
    bins: int = 60,
    ax: Optional[plt.Axes] = None,
    title: str = "log10(raw abundance) distribution",
) -> plt.Axes:
    """
    Histogram of log10(raw values) pooled across all proteins and patients.
    Diagnostic only (downstream uses log2 transform).
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 3))

    flat = df_raw.to_numpy().ravel()
    flat = flat[np.isfinite(flat)]
    sns.histplot(np.log10(flat), bins=bins, ax=ax, color=PALETTE["blue"])
    ax.set_title(title)
    ax.set_xlabel("log10(value)")
    ax.set_ylabel("Count")
    return ax


def plot_median_centering_diagnostics(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Returns:
      fig1: histogram of per-patient medians before vs after centering
    """
    med_before = df_before.median(axis=1)
    med_after = df_after.median(axis=1)

    fig1, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    sns.histplot(med_before, bins=25, kde=True, ax=axes[0], color=PALETTE["blue"])
    axes[0].set_title("Per-patient median (before centering)")
    axes[0].set_xlabel("Median log2 abundance")
    axes[0].set_ylabel("Count")

    sns.histplot(med_after, bins=25, kde=True, ax=axes[1], color=PALETTE["green"])
    axes[1].set_title("Per-patient median (after centering)")
    axes[1].set_xlabel("Median log2 abundance")

    fig1.tight_layout()
    return fig1


# ---------- PCA plots ----------

def plot_pca_scatter_with_outliers(
    pca_df_2d: pd.DataFrame,
    is_outlier: pd.Series,
    ax: Optional[plt.Axes] = None,
    title: str = "PCA (PC1 vs PC2) with outlier flags",
) -> plt.Axes:
    """
    Scatter of PC1 vs PC2 colored by outlier flag.
    Expects pca_df_2d to contain columns ['PC1', 'PC2'].
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5.5, 4))

    plot_df = pca_df_2d.copy()
    plot_df["outlier_flag"] = is_outlier.reindex(plot_df.index).map({False: "No", True: "Yes"})

    sns.scatterplot(
        data=plot_df, x="PC1", y="PC2",
        hue="outlier_flag",
        palette={"No": PALETTE["blue"], "Yes": PALETTE["red"]},
        s=60, alpha=0.85, ax=ax
    )
    ax.set_title(title)
    ax.legend(title="Outlier flagged", loc="best")
    return ax


def plot_scree_and_cumulative(
    explained_full: np.ndarray,
    explained_clean: Optional[np.ndarray] = None,
    elbow_full: Optional[int] = None,
    elbow_clean: Optional[int] = None,
    var_line: float = 0.80,
    plot_n: int = 20,
    int_xticks: bool = True,
) -> plt.Figure:
    """
    Consistent scree + cumulative plot (FULL vs CLEAN).
    """
    explained_full = np.asarray(explained_full)[:plot_n]
    cum_full = np.cumsum(explained_full)

    if explained_clean is not None:
        explained_clean = np.asarray(explained_clean)[:plot_n]
        cum_clean = np.cumsum(explained_clean)

    xs = np.arange(1, len(explained_full) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Scree
    axes[0].plot(xs, explained_full, marker="o", label="Full", color=PALETTE["blue"])
    if explained_clean is not None:
        axes[0].plot(xs, explained_clean, marker="o", label="Clean", color=PALETTE["orange"])
    if elbow_full is not None:
        axes[0].axvline(elbow_full, linestyle="--", color=PALETTE["black"], linewidth=1, label=f"Elbow(full)={elbow_full}")
    if elbow_clean is not None:
        axes[0].axvline(elbow_clean, linestyle="--", color=PALETTE["gray"], linewidth=1, label=f"Elbow(clean)={elbow_clean}")
    axes[0].set_title("Scree plot (explained variance ratio)")
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("Explained variance ratio")
    axes[0].legend()

    # Cumulative
    axes[1].plot(xs, cum_full, marker="o", label="Full", color=PALETTE["blue"])
    if explained_clean is not None:
        axes[1].plot(xs, cum_clean, marker="o", label="Clean", color=PALETTE["orange"])
    axes[1].axhline(var_line, linestyle="--", color=PALETTE["black"], linewidth=1, label=f"{int(var_line*100)}% variance")
    axes[1].set_title("Cumulative explained variance")
    axes[1].set_xlabel("Number of PCs")
    axes[1].set_ylabel("Cumulative explained variance")
    axes[1].legend()

    if int_xticks:
        for ax in axes:
            ax.set_xticks(xs)

    fig.tight_layout()
    return fig


def plot_k_selection_grid(res_full, res_clean, n_pc: int, figsize=(12, 7)):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    plot_elbow(res_full["inertia"], ax=axes[0, 0], title=f"Elbow (FULL, n_PC={n_pc})")
    plot_silhouette_scores(res_full["silhouette"], ax=axes[0, 1], title=f"Silhouette (FULL, n_PC={n_pc})")
    plot_elbow(res_clean["inertia"], ax=axes[1, 0], title=f"Elbow (CLEAN, n_PC={n_pc})")
    plot_silhouette_scores(res_clean["silhouette"], ax=axes[1, 1], title=f"Silhouette (CLEAN, n_PC={n_pc})")
    fig.tight_layout()
    return fig, axes


def get_cluster_palette(k: int = 2) -> Sequence[str]:
    """
    Consistent colors across the whole report.
    Uses seaborn 'colorblind' palette by default.
    """
    return sns.color_palette("colorblind", n_colors=k)


def overlay_flagged_outliers_on_clean_pca(
    df_scaled_clean: pd.DataFrame,
    df_scaled_outliers: pd.DataFrame,
    labels_clean: pd.Series,
    n_pc: int,
    title: str = "Clean clusters with flagged outliers (same PCA space)",
    random_state: int = 42,
    fig_size: Tuple[float, float] = (6.8, 4.4),
) -> plt.Figure:
    """
    Supplementary plot:
    Fit PCA on CLEAN only (n_pc), transform OUTLIERS with same PCA model,
    then show PC1/PC2 with clean clusters colored + outliers as black X.
    """
    # Fit PCA on clean, transform both
    pca = PCA(n_components=n_pc, random_state=random_state)
    Z_clean = pca.fit_transform(df_scaled_clean.values)
    Z_out = pca.transform(df_scaled_outliers.values)

    clean_df = pd.DataFrame(Z_clean[:, :2], index=df_scaled_clean.index, columns=["PC1", "PC2"])
    out_df = pd.DataFrame(Z_out[:, :2], index=df_scaled_outliers.index, columns=["PC1", "PC2"])

    clean_df["Cluster"] = labels_clean.loc[clean_df.index].astype(int).values
    palette = get_cluster_palette(k=len(np.unique(labels_clean)))

    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    sns.scatterplot(
        data=clean_df,
        x="PC1",
        y="PC2",
        hue="Cluster",
        palette=palette,
        s=55,
        alpha=0.85,
        ax=ax,
    )

    sns.scatterplot(
        data=out_df,
        x="PC1",
        y="PC2",
        color="black",
        marker="X",
        s=90,
        label="Flagged outlier",
        ax=ax,
    )

    ax.set_title(title)
    ax.legend(loc="best", title=None)
    plt.tight_layout()
    return fig

