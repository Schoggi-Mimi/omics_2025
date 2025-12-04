"""Bladder Proteomics Analysis Package.

A comprehensive Python package for bladder cancer proteomics analysis,
providing tools for preprocessing, dimensionality reduction, clustering,
feature selection, and visualization of proteomics data (3120×140 matrix + metadata).
"""

__version__ = "0.1.0"

from . import preprocessing
from . import dimensionality_reduction
from . import clustering
from . import k_selection
from . import feature_selection
from . import plotting
from . import utils

__all__ = [
    "preprocessing",
    "dimensionality_reduction",
    "clustering",
    "k_selection",
    "feature_selection",
    "plotting",
    "utils",
]
