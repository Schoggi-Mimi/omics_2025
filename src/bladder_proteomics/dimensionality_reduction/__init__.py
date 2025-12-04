"""Dimensionality reduction module.

Provides functions for reducing high-dimensional proteomics data including:
- PCA (Principal Component Analysis)
- UMAP (Uniform Manifold Approximation and Projection)
"""

from .reducers import apply_pca, apply_umap

__all__ = [
    "apply_pca",
    "apply_umap",
]
