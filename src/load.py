"""
Load module for TMDB Movie Data Analysis Pipeline.

This module handles data persistence:
- Saving raw data to CSV
- Saving processed/cleaned data to CSV
- Loading previously saved data

Example:
    >>> from src.load import save_raw, save_processed, load_processed
    >>> save_raw(df_raw)
    >>> save_processed(df_clean)
    >>> df_loaded = load_processed()
"""

import os
import pandas as pd
from typing import Optional

# Import configuration
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, 
    RAW_DATA_FILE, PROCESSED_DATA_FILE
)
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger("load")


def ensure_directory_exists(directory: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory: Path to directory
    
    Example:
        >>> ensure_directory_exists("data/raw")
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.debug(f"Created directory: {directory}")


def save_raw(df: pd.DataFrame, filename: Optional[str] = None) -> str:
    """
    Save raw data to CSV file.
    
    Args:
        df: DataFrame containing raw data
        filename: Optional custom filename (default: movies_raw.csv)
    
    Returns:
        Path to saved file
    
    Example:
        >>> from src.extract import extract
        >>> df_raw = extract()
        >>> filepath = save_raw(df_raw)
        >>> print(f"Saved to: {filepath}")
        Saved to: data/raw/movies_raw.csv
    """
    ensure_directory_exists(RAW_DATA_DIR)
    
    if filename is None:
        filename = RAW_DATA_FILE
    
    filepath = os.path.join(RAW_DATA_DIR, filename)
    
    logger.info(f"Saving raw data to {filepath}...")
    df.to_csv(filepath, index=False, encoding='utf-8')
    logger.info(f" Saved {len(df)} rows to {filepath}")
    
    return filepath


def save_processed(df: pd.DataFrame, filename: Optional[str] = None) -> str:
    """
    Save processed/cleaned data to CSV file.
    
    Args:
        df: DataFrame containing cleaned data
        filename: Optional custom filename (default: movies_clean.csv)
    
    Returns:
        Path to saved file
    
    Example:
        >>> from src.transform import transform
        >>> df_clean = transform(df_raw)
        >>> filepath = save_processed(df_clean)
        >>> print(f"Saved to: {filepath}")
        Saved to: data/processed/movies_clean.csv
    """
    ensure_directory_exists(PROCESSED_DATA_DIR)
    
    if filename is None:
        filename = PROCESSED_DATA_FILE
    
    filepath = os.path.join(PROCESSED_DATA_DIR, filename)
    
    logger.info(f"Saving processed data to {filepath}...")
    df.to_csv(filepath, index=False, encoding='utf-8')
    logger.info(f" Saved {len(df)} rows to {filepath}")
    
    return filepath


def load_raw(filename: Optional[str] = None) -> pd.DataFrame:
    """
    Load previously saved raw data.
    
    Args:
        filename: Optional custom filename (default: movies_raw.csv)
    
    Returns:
        DataFrame containing raw data
    
    Raises:
        FileNotFoundError: If the file doesn't exist
    
    Example:
        >>> df_raw = load_raw()
        >>> print(f"Loaded {len(df_raw)} rows")
    """
    if filename is None:
        filename = RAW_DATA_FILE
    
    filepath = os.path.join(RAW_DATA_DIR, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found: {filepath}")
    
    logger.info(f"Loading raw data from {filepath}...")
    df = pd.read_csv(filepath, encoding='utf-8')
    logger.info(f" Loaded {len(df)} rows from {filepath}")
    
    return df


def load_processed(filename: Optional[str] = None) -> pd.DataFrame:
    """
    Load previously saved processed/cleaned data.
    
    Args:
        filename: Optional custom filename (default: movies_clean.csv)
    
    Returns:
        DataFrame containing cleaned data
    
    Raises:
        FileNotFoundError: If the file doesn't exist
    
    Example:
        >>> df_clean = load_processed()
        >>> print(f"Loaded {len(df_clean)} movies")
    """
    if filename is None:
        filename = PROCESSED_DATA_FILE
    
    filepath = os.path.join(PROCESSED_DATA_DIR, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Processed data file not found: {filepath}")
    
    logger.info(f"Loading processed data from {filepath}...")
    df = pd.read_csv(filepath, encoding='utf-8', parse_dates=['release_date'])
    logger.info(f" Loaded {len(df)} rows from {filepath}")
    
    return df


def save_analysis_results(results: dict, filename: str = "analysis_results.csv") -> str:
    """
    Save analysis results (e.g., rankings, comparisons) to CSV.
    
    Args:
        results: Dictionary where keys are category names and values are DataFrames
        filename: Output filename
    
    Returns:
        Path to saved file
    
    Example:
        >>> results = {'top_revenue': df_top, 'top_rated': df_rated}
        >>> save_analysis_results(results)
    """
    ensure_directory_exists(PROCESSED_DATA_DIR)
    filepath = os.path.join(PROCESSED_DATA_DIR, filename)
    
    logger.info(f"Saving analysis results to {filepath}...")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for category, data in results.items():
            f.write(f"\n=== {category.upper()} ===\n")
            if isinstance(data, pd.DataFrame):
                data.to_csv(f, index=False)
            else:
                f.write(str(data) + "\n")
    
    logger.info(f"✓ Saved analysis results to {filepath}")
    return filepath


if __name__ == "__main__":
    # Test load module
    print("Testing load module...")
    
    # Create sample data
    sample_df = pd.DataFrame({
        'id': [1, 2, 3],
        'title': ['Movie A', 'Movie B', 'Movie C'],
        'release_date': ['2023-01-15', '2023-06-20', '2023-12-01'],
        'budget_musd': [100, 50, 200]
    })
    
    # Test save functions
    raw_path = save_raw(sample_df)
    processed_path = save_processed(sample_df)
    
    print(f"\nSaved raw to: {raw_path}")
    print(f"Saved processed to: {processed_path}")
    
    # Test load functions
    df_loaded = load_processed()
    print(f"\nLoaded {len(df_loaded)} rows")
