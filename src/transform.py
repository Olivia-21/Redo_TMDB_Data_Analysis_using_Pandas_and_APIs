"""
Transform module for TMDB Movie Data Analysis Pipeline.

This module handles all data cleaning and preprocessing:
- Dropping irrelevant columns
- Parsing JSON-like columns
- Converting data types
- Handling missing values
- Filtering and reordering data

Example:
    >>> from src.transform import transform
    >>> df_clean = transform(df_raw)
    >>> print(df_clean.shape)
"""

import pandas as pd
import numpy as np
from typing import Any, Optional, List
import ast

# Import configuration
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    COLUMNS_TO_DROP, JSON_COLUMNS, FINAL_COLUMN_ORDER
)
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger("transform")


def safe_parse_json(value: Any) -> Any:
    """
    Safely parse a JSON-like string value.
    
    Args:
        value: Value to parse (string, dict, list, or None)
    
    Returns:
        Parsed value or None if parsing fails
    
    Example:
        >>> safe_parse_json("[{'name': 'Action'}]")
        [{'name': 'Action'}]
    """
    # Handle None and NaN values
    if value is None:
        return None
    
    # Handle already parsed dict/list (from JSON API response)
    if isinstance(value, (dict, list)):
        return value
    
    # Handle pandas NA/NaN (scalar check only)
    try:
        if pd.isna(value):
            return None
    except (ValueError, TypeError):
        # pd.isna fails on arrays, which means it's already a valid structure
        pass
    
    # Try to parse string representation
    try:
        return ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return None


def extract_names_from_json(json_data: Any, separator: str = "|") -> Optional[str]:
    """
    Extract 'name' fields from JSON-like data and join with separator.
    
    Args:
        json_data: List of dictionaries with 'name' keys
        separator: String to join names (default: "|")
    
    Returns:
        Pipe-separated string of names, or None if no data
    
    Example:
        >>> data = [{'name': 'Action'}, {'name': 'Adventure'}]
        >>> extract_names_from_json(data)
        'Action|Adventure'
    """
    if not json_data or not isinstance(json_data, list):
        return None
    
    names = [
        item.get("name") or item.get("english_name", "")
        for item in json_data 
        if isinstance(item, dict)
    ]
    names = [name for name in names if name]
    
    return separator.join(names) if names else None


def extract_collection_name(json_data: Any) -> Optional[str]:
    """
    Extract collection name from belongs_to_collection field.
    
    Args:
        json_data: Dictionary containing collection data
    
    Returns:
        Collection name or None
    
    Example:
        >>> data = {'name': 'Avengers Collection', 'id': 86311}
        >>> extract_collection_name(data)
        'Avengers Collection'
    """
    if not json_data or not isinstance(json_data, dict):
        return None
    
    return json_data.get("name")


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are not needed for analysis.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with irrelevant columns removed
    
    Example:
        >>> df_clean = drop_irrelevant_columns(df_raw)
        >>> 'adult' in df_clean.columns
        False
    """
    cols_to_drop = [col for col in COLUMNS_TO_DROP if col in df.columns]
    
    if cols_to_drop:
        logger.info(f"Dropping columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    
    return df


def parse_json_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse and extract data from JSON-like columns.
    
    Transforms:
    - belongs_to_collection → collection name (string)
    - genres → pipe-separated genre names
    - production_companies → pipe-separated company names
    - production_countries → pipe-separated country names
    - spoken_languages → pipe-separated language names
    
    Args:
        df: Input DataFrame with JSON-like columns
    
    Returns:
        DataFrame with parsed columns
    
    Example:
        >>> df_parsed = parse_json_columns(df_raw)
        >>> print(df_parsed['genres'].iloc[0])
        'Action|Adventure|Science Fiction'
    """
    logger.info("Parsing JSON-like columns...")
    df = df.copy()
    
    # Parse belongs_to_collection
    if 'belongs_to_collection' in df.columns:
        df['belongs_to_collection'] = df['belongs_to_collection'].apply(
            lambda x: extract_collection_name(safe_parse_json(x))
        )
        logger.debug("Parsed belongs_to_collection")
    
    # Parse genres
    if 'genres' in df.columns:
        df['genres'] = df['genres'].apply(
            lambda x: extract_names_from_json(safe_parse_json(x))
        )
        logger.debug("Parsed genres")
    
    # Parse production_companies
    if 'production_companies' in df.columns:
        df['production_companies'] = df['production_companies'].apply(
            lambda x: extract_names_from_json(safe_parse_json(x))
        )
        logger.debug("Parsed production_companies")
    
    # Parse production_countries
    if 'production_countries' in df.columns:
        df['production_countries'] = df['production_countries'].apply(
            lambda x: extract_names_from_json(safe_parse_json(x))
        )
        logger.debug("Parsed production_countries")
    
    # Parse spoken_languages
    if 'spoken_languages' in df.columns:
        df['spoken_languages'] = df['spoken_languages'].apply(
            lambda x: extract_names_from_json(safe_parse_json(x))
        )
        logger.debug("Parsed spoken_languages")
    
    logger.info("JSON column parsing complete")
    return df


def convert_datatypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to appropriate data types.
    
    Conversions:
    - budget, revenue → numeric (converted to millions USD)
    - release_date → datetime
    - popularity, vote_average, vote_count, runtime → numeric
    - id → integer
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with converted data types
    
    Example:
        >>> df_typed = convert_datatypes(df_raw)
        >>> df_typed['budget_musd'].dtype
        dtype('float64')
    """
    logger.info("Converting data types...")
    df = df.copy()
    
    # Convert numeric columns
    numeric_cols = ['budget', 'revenue', 'popularity', 'vote_average', 'vote_count', 'runtime', 'id']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert budget and revenue to millions USD
    if 'budget' in df.columns:
        df['budget_musd'] = df['budget'] / 1_000_000
        df = df.drop(columns=['budget'])
        logger.debug("Converted budget to millions USD")
    
    if 'revenue' in df.columns:
        df['revenue_musd'] = df['revenue'] / 1_000_000
        df = df.drop(columns=['revenue'])
        logger.debug("Converted revenue to millions USD")
    
    # Convert release_date to datetime
    if 'release_date' in df.columns:
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        logger.debug("Converted release_date to datetime")
    
    # Ensure id is integer
    if 'id' in df.columns:
        df['id'] = df['id'].astype('Int64')  # Nullable integer
    
    logger.info("Data type conversion complete")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing and invalid values in the dataset.
    
    Actions:
    - Replace 0 values in budget/revenue/runtime with NaN
    - Replace placeholder text in overview/tagline with NaN
    - Handle vote_count = 0 cases
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with handled missing values
    
    Example:
        >>> df_clean = handle_missing_values(df_raw)
        >>> (df_clean['budget_musd'] == 0).sum()
        0
    """
    logger.info("Handling missing values...")
    df = df.copy()
    
    # Replace 0 with NaN in budget, revenue, runtime
    zero_to_nan_cols = ['budget_musd', 'revenue_musd', 'runtime']
    for col in zero_to_nan_cols:
        if col in df.columns:
            zero_count = (df[col] == 0).sum()
            if zero_count > 0:
                df[col] = df[col].replace(0, np.nan)
                logger.debug(f"Replaced {zero_count} zeros with NaN in {col}")
    
    # Replace placeholder text with NaN in text columns
    placeholder_patterns = ['No Data', 'N/A', 'NA', 'None', '', 'Unknown']
    text_cols = ['overview', 'tagline']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].replace(placeholder_patterns, np.nan)
            # Also handle empty strings
            df[col] = df[col].apply(lambda x: np.nan if pd.isna(x) or str(x).strip() == '' else x)
    
    logger.info("Missing value handling complete")
    return df


def remove_duplicates_and_invalid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows and rows with invalid/missing id or title.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with duplicates and invalid rows removed
    
    Example:
        >>> df_clean = remove_duplicates_and_invalid(df_raw)
        >>> df_clean.duplicated().sum()
        0
    """
    logger.info("Removing duplicates and invalid rows...")
    initial_count = len(df)
    
    # Remove duplicates based on id
    if 'id' in df.columns:
        df = df.drop_duplicates(subset=['id'], keep='first')
    
    # Remove rows with missing id or title
    if 'id' in df.columns:
        df = df[df['id'].notna()]
    if 'title' in df.columns:
        df = df[df['title'].notna()]
    
    removed = initial_count - len(df)
    if removed > 0:
        logger.info(f"Removed {removed} duplicate/invalid rows")
    
    return df


def filter_by_min_columns(df: pd.DataFrame, min_columns: int = 10) -> pd.DataFrame:
    """
    Keep only rows where at least a minimum number of columns have non-NaN values.
    
    Args:
        df: Input DataFrame
        min_columns: Minimum number of non-NaN columns required
    
    Returns:
        DataFrame with filtered rows
    
    Example:
        >>> df_filtered = filter_by_min_columns(df_raw, min_columns=10)
    """
    logger.info(f"Filtering rows with at least {min_columns} non-NaN values...")
    initial_count = len(df)
    
    # Count non-NaN values per row
    non_nan_counts = df.notna().sum(axis=1)
    df = df[non_nan_counts >= min_columns]
    
    removed = initial_count - len(df)
    if removed > 0:
        logger.info(f"Removed {removed} rows with too many missing values")
    
    return df


def filter_released_movies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only movies with 'Released' status and drop the status column.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with only released movies
    
    Example:
        >>> df_released = filter_released_movies(df_raw)
        >>> 'status' in df_released.columns
        False
    """
    logger.info("Filtering to released movies only...")
    
    if 'status' in df.columns:
        initial_count = len(df)
        df = df[df['status'] == 'Released']
        df = df.drop(columns=['status'])
        removed = initial_count - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} non-released movies")
    
    return df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder columns to match the final column order specification.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with reordered columns
    
    Example:
        >>> df_ordered = reorder_columns(df_clean)
        >>> list(df_ordered.columns)[:3]
        ['id', 'title', 'tagline']
    """
    logger.info("Reordering columns...")
    
    # Get columns that exist in both the desired order and the dataframe
    available_cols = [col for col in FINAL_COLUMN_ORDER if col in df.columns]
    
    # Add any remaining columns not in the desired order
    remaining_cols = [col for col in df.columns if col not in available_cols]
    
    final_order = available_cols + remaining_cols
    df = df[final_order]
    
    logger.debug(f"Final column order: {list(df.columns)}")
    return df


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main transformation function - orchestrates all cleaning steps.
    
    Pipeline steps:
    1. Drop irrelevant columns
    2. Parse JSON-like columns
    3. Convert data types
    4. Handle missing values
    5. Remove duplicates and invalid rows
    6. Filter by minimum column count
    7. Filter to released movies
    8. Reorder columns
    9. Reset index
    
    Args:
        df: Raw DataFrame from extraction
    
    Returns:
        Cleaned and transformed DataFrame
    
    Example:
        >>> from src.extract import extract
        >>> from src.transform import transform
        >>> df_raw = extract()
        >>> df_clean = transform(df_raw)
        >>> print(f"Cleaned: {len(df_clean)} movies")
    """
    logger.info(f"Starting transformation on {len(df)} rows...")
    
    # Step 1: Drop irrelevant columns
    df = drop_irrelevant_columns(df)
    
    # Step 2: Parse JSON-like columns
    df = parse_json_columns(df)
    
    # Step 3: Convert data types
    df = convert_datatypes(df)
    
    # Step 4: Handle missing values
    df = handle_missing_values(df)
    
    # Step 5: Remove duplicates and invalid rows
    df = remove_duplicates_and_invalid(df)
    
    # Step 6: Filter by minimum column count
    df = filter_by_min_columns(df, min_columns=10)
    
    # Step 7: Filter to released movies
    df = filter_released_movies(df)
    
    # Step 8: Reorder columns
    df = reorder_columns(df)
    
    # Step 9: Reset index
    df = df.reset_index(drop=True)
    
    logger.info(f"Transformation complete! Final shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    return df


if __name__ == "__main__":
    # Test with sample data
    print("Testing transform module...")
    
    # Create sample data
    sample_data = {
        'id': [1, 2],
        'title': ['Test Movie 1', 'Test Movie 2'],
        'adult': [False, False],
        'genres': [[{'name': 'Action'}, {'name': 'Drama'}], [{'name': 'Comedy'}]],
        'budget': [100000000, 50000000],
        'revenue': [500000000, 150000000],
        'release_date': ['2023-01-15', '2023-06-20'],
        'status': ['Released', 'Released'],
        'popularity': [100.5, 75.3],
        'vote_average': [8.1, 7.2],
        'vote_count': [1000, 500],
        'runtime': [120, 95],
        'overview': ['A test movie', 'Another test movie'],
        'tagline': ['Test tagline', ''],
        'original_language': ['en', 'en'],
        'belongs_to_collection': [None, {'name': 'Test Collection'}],
        'production_companies': [[{'name': 'Studio A'}], [{'name': 'Studio B'}]],
        'production_countries': [[{'name': 'USA'}], [{'name': 'UK'}]],
        'spoken_languages': [[{'name': 'English'}], [{'name': 'English'}]],
        'poster_path': ['/path1.jpg', '/path2.jpg'],
        'cast': ['Actor A|Actor B', 'Actor C'],
        'cast_size': [10, 5],
        'director': ['Director A', 'Director B'],
        'crew_size': [50, 30]
    }
    
    df_test = pd.DataFrame(sample_data)
    df_clean = transform(df_test)
    
    print(f"\nTest DataFrame shape: {df_clean.shape}")
    print(f"Columns: {list(df_clean.columns)}")
