"""Dimensionality reduction module.

Provides functions for:
- PCA (Principal Component Analysis)
- UMAP (Uniform Manifold Approximation and Projection)
- PCA elbow detection (pca_elbow)
"""

from .reducers import apply_pca, apply_umap, pca_elbow

__all__ = [
    "apply_pca",
    "apply_umap",
    "pca_elbow",
]
