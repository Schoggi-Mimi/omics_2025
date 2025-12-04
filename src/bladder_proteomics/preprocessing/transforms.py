"""Data transformation and filtering functions for proteomics data."""

from typing import Tuple, Union

import numpy as np
import pandas as pd


def log1p_transform(data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
    """Apply log(1+x) transformation to the data.
    
    This transformation is useful for stabilizing variance and handling
    data with wide dynamic range, common in proteomics experiments.
    
    Args:
        data: Input data matrix (samples x features) or DataFrame
        
    Returns:
        Transformed data in the same format as input
        
    Examples:
        >>> import numpy as np
        >>> data = np.array([[1, 2, 3], [4, 5, 6]])
        >>> transformed = log1p_transform(data)
    """
    if isinstance(data, pd.DataFrame):
        return data.apply(np.log1p)
    return np.log1p(data)


def zscore_normalize(
    data: Union[np.ndarray, pd.DataFrame],
    axis: int = 0,
    with_mean: bool = True,
    with_std: bool = True
) -> Union[np.ndarray, pd.DataFrame]:
    """Apply z-score normalization to the data.
    
    Standardizes features by removing the mean and scaling to unit variance.
    Formula: z = (x - mean) / std
    
    Args:
        data: Input data matrix (samples x features) or DataFrame
        axis: Axis along which to normalize (0=columns/features, 1=rows/samples)
        with_mean: If True, center the data before scaling
        with_std: If True, scale the data to unit variance
        
    Returns:
        Normalized data in the same format as input
        
    Examples:
        >>> import numpy as np
        >>> data = np.array([[1, 2, 3], [4, 5, 6]])
        >>> normalized = zscore_normalize(data, axis=0)
    """
    # if isinstance(data, pd.DataFrame):
        # result = data.copy()
        # if with_mean:
        #     result = result - result.mean(axis=axis)
        # if with_std:
        #     #result = result / result.std(axis=axis) # if axis=1, this may produce wrong results
        #     result = (result - result.mean(axis=axis))  
        #     result = result.divide(result.std(axis=axis), axis=1-axis)
        # return result
    # else:
    #     result = data.copy().astype(float)
    #     if with_mean:
    #         result = result - np.mean(result, axis=axis, keepdims=True)
    #     if with_std:
    #         std = np.std(result, axis=axis, keepdims=True)
    #         # Avoid division by zero with small epsilon
    #         std = np.where(std == 0, 1e-8, std)
    #         result = result / std
    #     return result
    if isinstance(data, pd.DataFrame):
        result = data.copy()
        if axis == 0: # normalize columns/features
            means = result.mean(axis=0)
            stds = result.std(axis=0)
            stds = stds.replace(0, 0.00000001) # Avoid division by zero
            if with_mean and with_std:
                result = (result - means) / stds
            elif with_mean:
                result = result - means
            elif with_std:
                result = result / stds
        elif axis == 1: # normalize rows/samples
            means = result.mean(axis=1)
            stds = result.std(axis=1)
            stds = stds.replace(0, 0.00000001) # Avoid division by zero
            if with_mean and with_std:
                result = result.sub(means, axis=0).div(stds, axis=0)
            elif with_mean:
                result = result.sub(means, axis=0)
            elif with_std:
                result = result.div(stds, axis=0)
        else:
            raise ValueError("Axis must be 0 (columns) or 1 (rows).")
        return result
    else:
        result = data.copy().astype(float)
        mean = result.mean(axis=axis, keepdims=True)
        std = result.std(axis=axis, keepdims=True)
        std = np.where(std == 0, 0.00000001, std)
        if with_mean and with_std:
            return (result - mean) / std
        elif with_mean:
            return result - mean
        elif with_std:
            return result / std
        return result


def variance_filter(
    data: Union[np.ndarray, pd.DataFrame],
    threshold: float = 0.0,
    percentile: float = None
) -> Tuple[Union[np.ndarray, pd.DataFrame], np.ndarray]:
    """Filter features based on variance threshold.
    
    Removes low-variance features that are likely uninformative.
    Can use either an absolute threshold or a percentile-based threshold.
    
    Args:
        data: Input data matrix (samples x features) or DataFrame
        threshold: Minimum variance threshold (default: 0.0, removes constant features)
        percentile: If provided, keep features above this variance percentile (0-100)
        
    Returns:
        Tuple of (filtered_data, mask) where mask indicates kept features
        
    Examples:
        >>> import numpy as np
        >>> data = np.array([[1, 2, 0], [1, 3, 0], [1, 4, 0]])
        >>> filtered, mask = variance_filter(data, threshold=0.1)
    """
    if isinstance(data, pd.DataFrame):
        variances = data.var(axis=0)
        
        if percentile is not None:
            threshold = np.percentile(variances, percentile)
        
        mask = variances >= threshold
        filtered_data = data.loc[:, mask]
        return filtered_data, mask # mask.values can misalign later
    data = np.asarray(data) # force numpy
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {data.shape}")
    variances = np.var(data, axis=0)
    
    if percentile is not None:
        threshold = np.percentile(variances, percentile)
    
    mask = variances >= threshold
    filtered_data = data[:, mask]
    return filtered_data, mask
