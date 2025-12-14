"""Data loading and validation utilities for proteomics data."""

import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd


def load_data(
    data_path: Union[str, Path],
    metadata_path: Optional[Union[str, Path]] = None,
    transpose: bool = False,
    index_col: Optional[int] = 0,
    sep: str = "\t"
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Load proteomics data and optional metadata.
    
    Supports various file formats including CSV, TSV, and Excel files.
    Expected data format: 140 patients x 3121 features (proteins).
    
    Args:
        data_path: Path to the main data file
        metadata_path: Optional path to metadata file
        transpose: If True, transpose the data (useful if features are rows)
        index_col: Column to use as row index (default: first column)
        sep: Delimiter for text files (default: tab)
        
    Returns:
        DataFrame with data, or tuple of (data, metadata) if metadata_path provided
        
    Examples:
        >>> data = load_data("proteomics_data.csv")
        >>> data, metadata = load_data("proteomics_data.csv", "metadata.csv")
    """
    data_path = Path(data_path)
    
    # Determine file type and load data
    if data_path.suffix.lower() in [".xlsx", ".xls"]:
        data = pd.read_excel(data_path, index_col=index_col)
    elif data_path.suffix.lower() == ".csv":
        data = pd.read_csv(data_path, index_col=index_col)
    elif data_path.suffix.lower() in [".tsv", ".txt"]:
        data = pd.read_csv(data_path, index_col=index_col, sep=sep)
    else:
        # Try to load as CSV by default
        try:
            data = pd.read_csv(data_path, index_col=index_col)
        except Exception as e:
            raise ValueError(f"Unsupported file format: {data_path.suffix}. Error: {e}")
    
    # Transpose if requested
    if transpose:
        data = data.T

    # Remove duplicate columns if any
    n_before = data.shape[1]
    data = data.loc[:, ~data.T.duplicated()] # protein features removed
    n_removed_profiles = n_before - data.shape[1]
    print(f"Removed proteins with identical profiles: {n_removed_profiles}")
    # Load metadata if provided
    if metadata_path is not None:
        metadata_path = Path(metadata_path)
        
        if metadata_path.suffix.lower() in [".xlsx", ".xls"]:
            metadata = pd.read_excel(metadata_path, index_col=index_col)
        elif metadata_path.suffix.lower() == ".csv":
            metadata = pd.read_csv(metadata_path, index_col=index_col)
        elif metadata_path.suffix.lower() in [".tsv", ".txt"]:
            metadata = pd.read_csv(metadata_path, index_col=index_col, sep=sep)
        else:
            try:
                metadata = pd.read_csv(metadata_path, index_col=index_col)
            except Exception as e:
                raise ValueError(f"Unsupported metadata file format: {metadata_path.suffix}. Error: {e}")
        
        return data, metadata
    
    return data


def validate_data(
    data: pd.DataFrame,
    check_missing: bool = True,
    check_duplicates: bool = True,
) -> Dict[str, Any]:
    """Lightweight validation without mutating the data."""
    report: Dict[str, Any] = {
        "shape": data.shape,
        "n_samples": data.shape[0],
        "n_features": data.shape[1],
    }

    if check_missing:
        report["missing_values"] = int(data.isna().sum().sum())

    if check_duplicates:
        report["duplicate_rows"] = int(data.duplicated().sum())
        report["duplicate_cols"] = int(data.T.duplicated().sum())

    # Print short summary
    msg = [
        f"Data shape (samples × features): {report['shape']}",
    ]
    if check_missing:
        msg.append(f"Missing values: {report['missing_values']}")
    if check_duplicates:
        msg.append(f"Duplicate rows: {report['duplicate_rows']} | Duplicate columns: {report['duplicate_cols']}")

    print("\n".join(msg))
    print("=" * 60)
    return report


def save_results(
    data: Union[pd.DataFrame, np.ndarray],
    output_path: Union[str, Path],
    index: bool = True,
    **kwargs
) -> None:
    """Save analysis results to file.
    
    Supports CSV, TSV, and Excel formats based on file extension.
    
    Args:
        data: Data to save (DataFrame or array)
        output_path: Path where to save the file
        index: Whether to write row index (for DataFrames)
        **kwargs: Additional arguments passed to pandas save methods
        
    Examples:
        >>> results = pd.DataFrame({"feature": ["A", "B"], "score": [0.5, 0.8]})
        >>> save_results(results, "feature_scores.csv")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame if needed
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    
    # Save based on file extension
    if output_path.suffix.lower() in [".xlsx", ".xls"]:
        data.to_excel(output_path, index=index, **kwargs)
    elif output_path.suffix.lower() == ".csv":
        data.to_csv(output_path, index=index, **kwargs)
    elif output_path.suffix.lower() in [".tsv", ".txt"]:
        data.to_csv(output_path, index=index, sep="\t", **kwargs)
    else:
        # Default to CSV
        warnings.warn(f"Unknown extension {output_path.suffix}, saving as CSV")
        data.to_csv(output_path, index=index, **kwargs)
    
    print(f"Results saved to: {output_path}")
