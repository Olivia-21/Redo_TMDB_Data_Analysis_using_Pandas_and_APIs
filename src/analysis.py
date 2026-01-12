"""
Analysis module for TMDB Movie Data Analysis Pipeline.

This module provides KPI calculations and advanced queries:
- Movie rankings (revenue, budget, profit, ROI, ratings, popularity)
- Advanced search and filtering
- Franchise vs standalone comparison
- Top franchises and directors analysis

Example:
    >>> from src.analysis import get_best_worst_movies, top_directors
    >>> rankings = get_best_worst_movies(df_clean)
    >>> directors = top_directors(df_clean)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable

# Import configuration
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import MIN_BUDGET_FOR_ROI, MIN_VOTES_FOR_RATING, DEFAULT_TOP_N
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger("analysis")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_profit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add profit column to DataFrame (revenue - budget).
    
    Args:
        df: DataFrame with budget_musd and revenue_musd columns
    
    Returns:
        DataFrame with profit_musd column added
    
    Example:
        >>> df = calculate_profit(df)
        >>> print(df['profit_musd'].head())
    """
    df = df.copy()
    df['profit_musd'] = df['revenue_musd'] - df['budget_musd']
    logger.debug("Calculated profit column")
    return df


def calculate_roi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ROI (Return on Investment) column to DataFrame.
    
    ROI = (Revenue - Budget) / Budget * 100
    
    Args:
        df: DataFrame with budget_musd and revenue_musd columns
    
    Returns:
        DataFrame with roi_percent column added
    
    Example:
        >>> df = calculate_roi(df)
        >>> print(df['roi_percent'].head())
    """
    df = df.copy()
    # Avoid division by zero
    df['roi_percent'] = np.where(
        df['budget_musd'] > 0,
        ((df['revenue_musd'] - df['budget_musd']) / df['budget_musd']) * 100,
        np.nan
    )
    logger.debug("Calculated ROI column")
    return df


def rank_movies(
    df: pd.DataFrame, 
    column: str, 
    ascending: bool = False, 
    top_n: int = DEFAULT_TOP_N,
    filter_condition: Optional[Callable] = None,
    display_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Generic User-Defined Function (UDF) for ranking movies.
    
    Args:
        df: Input DataFrame
        column: Column to rank by
        ascending: If True, rank smallest first (default: False = largest first)
        top_n: Number of top results to return
        filter_condition: Optional function to filter DataFrame before ranking
        display_columns: Optional list of columns to display in result
    
    Returns:
        DataFrame with top N ranked movies
    
    Example:
        >>> # Get top 5 movies by revenue
        >>> top_revenue = rank_movies(df, 'revenue_musd', top_n=5)
        
        >>> # Get lowest rated movies with at least 10 votes
        >>> filter_fn = lambda x: x['vote_count'] >= 10
        >>> worst_rated = rank_movies(df, 'vote_average', ascending=True, 
        ...                           filter_condition=filter_fn, top_n=5)
    """
    df_ranked = df.copy()
    
    # Apply filter if provided
    if filter_condition is not None:
        df_ranked = df_ranked[filter_condition(df_ranked)]
    
    # Remove NaN values in ranking column
    df_ranked = df_ranked[df_ranked[column].notna()]
    
    # Sort and get top N
    df_ranked = df_ranked.sort_values(column, ascending=ascending).head(top_n)
    
    # Select display columns if specified
    if display_columns:
        available_cols = [col for col in display_columns if col in df_ranked.columns]
        df_ranked = df_ranked[available_cols]
    
    return df_ranked.reset_index(drop=True)


# =============================================================================
# BEST/WORST MOVIE RANKINGS
# =============================================================================

def get_best_worst_movies(df: pd.DataFrame, top_n: int = DEFAULT_TOP_N) -> Dict[str, pd.DataFrame]:
    """
    Get comprehensive rankings for best and worst performing movies.
    
    Categories:
    1. Highest/Lowest Revenue
    2. Highest/Lowest Budget
    3. Highest/Lowest Profit
    4. Highest/Lowest ROI (budget >= $10M)
    5. Most Voted Movies
    6. Highest/Lowest Rated (>= 10 votes)
    7. Most Popular Movies
    
    Args:
        df: Cleaned DataFrame
        top_n: Number of results per category
    
    Returns:
        Dictionary with category names as keys and DataFrames as values
    
    Example:
        >>> rankings = get_best_worst_movies(df_clean)
        >>> print(rankings['highest_revenue'])
        >>> print(rankings['highest_roi'])
    """
    logger.info("Calculating best/worst movie rankings...")
    
    # Add calculated columns
    df = calculate_profit(df)
    df = calculate_roi(df)
    
    # Display columns for rankings
    display_cols = ['title', 'release_date', 'budget_musd', 'revenue_musd', 
                    'profit_musd', 'roi_percent', 'vote_average', 'vote_count', 'popularity']
    
    rankings = {}
    
    # 1. Highest Revenue
    rankings['highest_revenue'] = rank_movies(
        df, 'revenue_musd', ascending=False, top_n=top_n, display_columns=display_cols
    )
    logger.debug("Calculated highest revenue ranking")
    
    # 2. Lowest Revenue (non-zero)
    rankings['lowest_revenue'] = rank_movies(
        df, 'revenue_musd', ascending=True, top_n=top_n,
        filter_condition=lambda x: x['revenue_musd'] > 0,
        display_columns=display_cols
    )
    
    # 3. Highest Budget
    rankings['highest_budget'] = rank_movies(
        df, 'budget_musd', ascending=False, top_n=top_n, display_columns=display_cols
    )
    
    # 4. Lowest Budget (non-zero)
    rankings['lowest_budget'] = rank_movies(
        df, 'budget_musd', ascending=True, top_n=top_n,
        filter_condition=lambda x: x['budget_musd'] > 0,
        display_columns=display_cols
    )
    
    # 5. Highest Profit
    rankings['highest_profit'] = rank_movies(
        df, 'profit_musd', ascending=False, top_n=top_n, display_columns=display_cols
    )
    
    # 6. Lowest Profit
    rankings['lowest_profit'] = rank_movies(
        df, 'profit_musd', ascending=True, top_n=top_n, display_columns=display_cols
    )
    
    # 7. Highest ROI (budget >= $10M)
    rankings['highest_roi'] = rank_movies(
        df, 'roi_percent', ascending=False, top_n=top_n,
        filter_condition=lambda x: x['budget_musd'] >= MIN_BUDGET_FOR_ROI,
        display_columns=display_cols
    )
    
    # 8. Lowest ROI (budget >= $10M)
    rankings['lowest_roi'] = rank_movies(
        df, 'roi_percent', ascending=True, top_n=top_n,
        filter_condition=lambda x: x['budget_musd'] >= MIN_BUDGET_FOR_ROI,
        display_columns=display_cols
    )
    
    # 9. Most Voted
    rankings['most_voted'] = rank_movies(
        df, 'vote_count', ascending=False, top_n=top_n, display_columns=display_cols
    )
    
    # 10. Highest Rated (>= 10 votes)
    rankings['highest_rated'] = rank_movies(
        df, 'vote_average', ascending=False, top_n=top_n,
        filter_condition=lambda x: x['vote_count'] >= MIN_VOTES_FOR_RATING,
        display_columns=display_cols
    )
    
    # 11. Lowest Rated (>= 10 votes)
    rankings['lowest_rated'] = rank_movies(
        df, 'vote_average', ascending=True, top_n=top_n,
        filter_condition=lambda x: x['vote_count'] >= MIN_VOTES_FOR_RATING,
        display_columns=display_cols
    )
    
    # 12. Most Popular
    rankings['most_popular'] = rank_movies(
        df, 'popularity', ascending=False, top_n=top_n, display_columns=display_cols
    )
    
    logger.info(f"Generated {len(rankings)} ranking categories")
    return rankings


# =============================================================================
# ADVANCED SEARCH QUERIES
# =============================================================================

def search_movies(
    df: pd.DataFrame,
    genres: Optional[List[str]] = None,
    cast: Optional[str] = None,
    director: Optional[str] = None,
    min_rating: Optional[float] = None,
    sort_by: str = 'vote_average',
    ascending: bool = False
) -> pd.DataFrame:
    """
    Advanced movie search with multiple filter criteria.
    
    Args:
        df: Input DataFrame
        genres: List of genres to filter (movie must contain ALL specified genres)
        cast: Actor name to search for in cast
        director: Director name to filter by
        min_rating: Minimum vote_average
        sort_by: Column to sort results by
        ascending: Sort order
    
    Returns:
        Filtered and sorted DataFrame
    
    Example:
        >>> # Find best-rated Sci-Fi Action movies with Bruce Willis
        >>> results = search_movies(df, genres=['Science Fiction', 'Action'],
        ...                         cast='Bruce Willis', sort_by='vote_average')
        
        >>> # Find Uma Thurman movies directed by Tarantino
        >>> results = search_movies(df, cast='Uma Thurman', 
        ...                         director='Quentin Tarantino', sort_by='runtime')
    """
    logger.info("Executing advanced movie search...")
    result = df.copy()
    
    # Filter by genres
    if genres:
        for genre in genres:
            result = result[result['genres'].str.contains(genre, case=False, na=False)]
        logger.debug(f"Filtered by genres: {genres} ({len(result)} remaining)")
    
    # Filter by cast
    if cast:
        result = result[result['cast'].str.contains(cast, case=False, na=False)]
        logger.debug(f"Filtered by cast: {cast} ({len(result)} remaining)")
    
    # Filter by director
    if director:
        result = result[result['director'].str.contains(director, case=False, na=False)]
        logger.debug(f"Filtered by director: {director} ({len(result)} remaining)")
    
    # Filter by minimum rating
    if min_rating is not None:
        result = result[result['vote_average'] >= min_rating]
        logger.debug(f"Filtered by min rating: {min_rating} ({len(result)} remaining)")
    
    # Sort results
    if sort_by in result.columns:
        result = result.sort_values(sort_by, ascending=ascending)
    
    logger.info(f"Search returned {len(result)} results")
    return result.reset_index(drop=True)


def search_bruce_willis_scifi_action(df: pd.DataFrame) -> pd.DataFrame:
    """
    Search 1: Find best-rated Science Fiction Action movies starring Bruce Willis.
    
    Sorted by rating (highest to lowest).
    
    Args:
        df: Input DataFrame
    
    Returns:
        Filtered DataFrame sorted by rating
    
    Example:
        >>> results = search_bruce_willis_scifi_action(df_clean)
    """
    logger.info("Executing Search 1: Bruce Willis Sci-Fi Action movies...")
    return search_movies(
        df, 
        genres=['Science Fiction', 'Action'],
        cast='Bruce Willis',
        sort_by='vote_average',
        ascending=False
    )


def search_thurman_tarantino(df: pd.DataFrame) -> pd.DataFrame:
    """
    Search 2: Find movies starring Uma Thurman, directed by Quentin Tarantino.
    
    Sorted by runtime (shortest to longest).
    
    Args:
        df: Input DataFrame
    
    Returns:
        Filtered DataFrame sorted by runtime
    
    Example:
        >>> results = search_thurman_tarantino(df_clean)
    """
    logger.info("Executing Search 2: Uma Thurman + Quentin Tarantino movies...")
    return search_movies(
        df,
        cast='Uma Thurman',
        director='Quentin Tarantino',
        sort_by='runtime',
        ascending=True
    )


# =============================================================================
# FRANCHISE VS STANDALONE ANALYSIS
# =============================================================================

def franchise_vs_standalone(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare performance metrics between franchise and standalone movies.
    
    Metrics compared:
    - Mean Revenue
    - Median ROI
    - Mean Budget
    - Mean Popularity
    - Mean Rating
    
    Args:
        df: Input DataFrame with belongs_to_collection column
    
    Returns:
        DataFrame with comparison statistics
    
    Example:
        >>> comparison = franchise_vs_standalone(df_clean)
        >>> print(comparison)
    """
    logger.info("Comparing franchise vs standalone movies...")
    
    df = calculate_profit(df)
    df = calculate_roi(df)
    
    # Create franchise flag
    df['is_franchise'] = df['belongs_to_collection'].notna()
    
    # Group and aggregate
    comparison = df.groupby('is_franchise').agg({
        'revenue_musd': 'mean',
        'roi_percent': 'median',
        'budget_musd': 'mean',
        'popularity': 'mean',
        'vote_average': 'mean',
        'id': 'count'
    }).rename(columns={'id': 'movie_count'})
    
    # Rename index for clarity
    comparison.index = comparison.index.map({True: 'Franchise', False: 'Standalone'})
    comparison.index.name = 'Type'
    
    # Round values for readability
    comparison = comparison.round(2)
    
    logger.info("Franchise vs Standalone comparison complete")
    return comparison


# =============================================================================
# TOP FRANCHISES ANALYSIS
# =============================================================================

def top_franchises(df: pd.DataFrame, top_n: int = DEFAULT_TOP_N) -> pd.DataFrame:
    """
    Find the most successful movie franchises.
    
    Metrics:
    - Total number of movies
    - Total & Mean Budget
    - Total & Mean Revenue
    - Mean Rating
    
    Args:
        df: Input DataFrame
        top_n: Number of top franchises to return
    
    Returns:
        DataFrame with franchise statistics
    
    Example:
        >>> franchises = top_franchises(df_clean, top_n=10)
        >>> print(franchises)
    """
    logger.info(f"Analyzing top {top_n} franchises...")
    
    # Filter to franchise movies only
    franchise_df = df[df['belongs_to_collection'].notna()].copy()
    
    if franchise_df.empty:
        logger.warning("No franchise movies found")
        return pd.DataFrame()
    
    # Group by franchise
    franchise_stats = franchise_df.groupby('belongs_to_collection').agg({
        'id': 'count',
        'budget_musd': ['sum', 'mean'],
        'revenue_musd': ['sum', 'mean'],
        'vote_average': 'mean'
    })
    
    # Flatten column names
    franchise_stats.columns = [
        'movie_count', 
        'total_budget_musd', 'mean_budget_musd',
        'total_revenue_musd', 'mean_revenue_musd',
        'mean_rating'
    ]
    
    # Sort by total revenue
    franchise_stats = franchise_stats.sort_values('total_revenue_musd', ascending=False)
    
    # Round and reset index
    franchise_stats = franchise_stats.round(2).head(top_n).reset_index()
    franchise_stats = franchise_stats.rename(columns={'belongs_to_collection': 'franchise'})
    
    logger.info(f"Found {len(franchise_stats)} franchises")
    return franchise_stats


# =============================================================================
# TOP DIRECTORS ANALYSIS
# =============================================================================

def top_directors(df: pd.DataFrame, top_n: int = DEFAULT_TOP_N) -> pd.DataFrame:
    """
    Find the most successful directors.
    
    Metrics:
    - Total number of movies directed
    - Total Revenue
    - Mean Rating
    
    Args:
        df: Input DataFrame
        top_n: Number of top directors to return
    
    Returns:
        DataFrame with director statistics
    
    Example:
        >>> directors = top_directors(df_clean, top_n=10)
        >>> print(directors)
    """
    logger.info(f"Analyzing top {top_n} directors...")
    
    # Filter to movies with director info
    director_df = df[df['director'].notna()].copy()
    
    if director_df.empty:
        logger.warning("No director information found")
        return pd.DataFrame()
    
    # Explode directors (handle multiple directors per movie)
    director_df['director_list'] = director_df['director'].str.split('|')
    director_df = director_df.explode('director_list')
    director_df['director_list'] = director_df['director_list'].str.strip()
    
    # Group by director
    director_stats = director_df.groupby('director_list').agg({
        'id': 'count',
        'revenue_musd': 'sum',
        'vote_average': 'mean'
    })
    
    # Rename columns
    director_stats.columns = ['movie_count', 'total_revenue_musd', 'mean_rating']
    
    # Sort by total revenue
    director_stats = director_stats.sort_values('total_revenue_musd', ascending=False)
    
    # Round and reset index
    director_stats = director_stats.round(2).head(top_n).reset_index()
    director_stats = director_stats.rename(columns={'director_list': 'director'})
    
    logger.info(f"Found {len(director_stats)} directors")
    return director_stats


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def get_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate overall summary statistics for the dataset.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Dictionary with summary statistics
    
    Example:
        >>> stats = get_summary_statistics(df_clean)
        >>> print(f"Total movies: {stats['total_movies']}")
    """
    logger.info("Calculating summary statistics...")
    
    stats = {
        'total_movies': len(df),
        'date_range': f"{df['release_date'].min()} to {df['release_date'].max()}",
        'total_revenue_musd': df['revenue_musd'].sum(),
        'total_budget_musd': df['budget_musd'].sum(),
        'avg_revenue_musd': df['revenue_musd'].mean(),
        'avg_budget_musd': df['budget_musd'].mean(),
        'avg_rating': df['vote_average'].mean(),
        'avg_popularity': df['popularity'].mean(),
        'unique_genres': df['genres'].dropna().str.split('|').explode().nunique(),
        'franchise_movies': df['belongs_to_collection'].notna().sum(),
        'standalone_movies': df['belongs_to_collection'].isna().sum()
    }
    
    logger.info("Summary statistics calculated")
    return stats


if __name__ == "__main__":
    # Test analysis module
    print("Testing analysis module...")
    
    # Create sample data
    sample_data = {
        'id': [1, 2, 3, 4],
        'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D'],
        'release_date': pd.to_datetime(['2023-01-15', '2023-06-20', '2023-08-10', '2023-12-01']),
        'budget_musd': [100, 50, 200, 25],
        'revenue_musd': [500, 100, 800, 50],
        'vote_average': [8.5, 7.2, 9.1, 6.5],
        'vote_count': [1000, 500, 2000, 50],
        'popularity': [150.5, 75.3, 200.2, 40.1],
        'genres': ['Action|Adventure', 'Comedy|Drama', 'Action|Science Fiction', 'Drama'],
        'belongs_to_collection': ['Collection A', None, 'Collection A', None],
        'cast': ['Actor A|Actor B', 'Actor C', 'Actor D', 'Actor A'],
        'director': ['Director A', 'Director B', 'Director A', 'Director C']
    }
    
    df_test = pd.DataFrame(sample_data)
    
    # Test rankings
    print("\n=== Testing Rankings ===")
    rankings = get_best_worst_movies(df_test, top_n=3)
    print(f"Generated {len(rankings)} ranking categories")
    print(f"Highest revenue: {rankings['highest_revenue']['title'].tolist()}")
    
    # Test franchise comparison
    print("\n=== Testing Franchise vs Standalone ===")
    comparison = franchise_vs_standalone(df_test)
    print(comparison)
    
    # Test top directors
    print("\n=== Testing Top Directors ===")
    directors = top_directors(df_test, top_n=3)
    print(directors)
