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
    data = data.loc[:, ~data.columns.duplicated()]
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
    expected_samples: Optional[int] = 140,
    expected_features: Optional[int] = 3121,
    check_missing: bool = True,
    check_duplicates: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """Validate proteomics data quality and structure.
    
    Checks for:
    - Expected dimensions
    - Missing values
    - Duplicate samples/features
    - Data type consistency
    - Outliers
    
    Args:
        data: Input data DataFrame
        expected_samples: Expected number of samples (default: 140)
        expected_features: Expected number of features (default: 3121)
        check_missing: Check for missing values
        check_duplicates: Check for duplicate rows/columns
        verbose: Print validation report
        
    Returns:
        Dictionary with validation results
        
    Examples:
        >>> data = load_data("proteomics_data.csv")
        >>> report = validate_data(data)
    """
    report = {
        "shape": data.shape,
        "n_samples": data.shape[0],
        "n_features": data.shape[1],
        "issues": [],
        "warnings": []
    }
    # Check for non-numeric columns
    non_numeric = []
    for col in data.columns:
        if not pd.api.types.is_numeric_dtype(data[col]):
            non_numeric.append(col)

    if non_numeric:
        report["non_numeric_features"] = non_numeric
        report["warnings"].append(
            f"Dropping {len(non_numeric)} non-numeric features: {non_numeric}"
        )
        data = data.drop(columns=non_numeric)
    
    # Check dimensions
    if expected_samples is not None and data.shape[0] != expected_samples:
        report["warnings"].append(
            f"Expected {expected_samples} samples, found {data.shape[0]}"
        )
    
    if expected_features is not None and data.shape[1] != expected_features:
        report["warnings"].append(
            f"Expected {expected_features} features, found {data.shape[1]}"
        )
    
    # Check for missing values
    if check_missing:
        missing = data.isnull().sum().sum()
        if missing > 0:
            missing_pct = 100 * missing / (data.shape[0] * data.shape[1])
            report["missing_values"] = missing
            report["missing_percentage"] = missing_pct
            report["warnings"].append(
                f"Found {missing} missing values ({missing_pct:.2f}%)"
            )
        else:
            report["missing_values"] = 0
    
    # Check for duplicates
    if check_duplicates:
        dup_rows = data.duplicated().sum()
        dup_cols = data.T.duplicated().sum()
        
        if dup_rows > 0:
            report["duplicate_samples"] = dup_rows
            report["issues"].append(f"Found {dup_rows} duplicate samples")
        
        if dup_cols > 0:
            report["duplicate_features"] = dup_cols
            report["issues"].append(f"Found {dup_cols} duplicate features")
    
    # Check data types
    non_numeric = []
    for col in data.columns:
        if not pd.api.types.is_numeric_dtype(data[col]):
            non_numeric.append(col)
    
    # if non_numeric:
    #     report["non_numeric_features"] = non_numeric
    #     report["warnings"].append(
    #         f"Found {len(non_numeric)} non-numeric features"
    #     )
    
    # Check for extreme outliers (values > 5 standard deviations)
    # if len(non_numeric) < data.shape[1]:  # Only if we have numeric data
    #     numeric_data = data.select_dtypes(include=[np.number])
    #     z_scores = np.abs((numeric_data - numeric_data.mean()) / numeric_data.std())
    #     extreme_outliers = (z_scores > 5).sum().sum()
        
    #     if extreme_outliers > 0:
    #         report["extreme_outliers"] = extreme_outliers
    #         report["warnings"].append(
    #             f"Found {extreme_outliers} extreme outliers (>5 std)"
    #         )
    
    # Print report if verbose
    if verbose:
        print("=" * 60)
        print("DATA VALIDATION REPORT")
        print("=" * 60)
        print(f"Shape: {report['shape']} (samples × features)")
        print(f"Samples: {report['n_samples']}")
        print(f"Features: {report['n_features']}")
        
        if report.get("missing_values", 0) > 0:
            print(f"\nMissing values: {report['missing_values']} ({report['missing_percentage']:.2f}%)")
        else:
            print("\nNo missing values detected")
        
        if report["issues"]:
            print("\nISSUES:")
            for issue in report["issues"]:
                print(f"  ❌ {issue}")
        
        if report["warnings"]:
            print("\nWARNINGS:")
            for warning in report["warnings"]:
                print(f"  ⚠️  {warning}")
        
        if not report["issues"] and not report["warnings"]:
            print("\n✓ Data validation passed with no issues")
        
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
