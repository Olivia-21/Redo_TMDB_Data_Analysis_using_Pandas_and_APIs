"""
Extract module for TMDB Movie Data Analysis Pipeline.

This module handles all API interactions with TMDB, including:
- Fetching movie details
- Fetching cast and crew information
- Retry logic with exponential backoff
- Rate limiting to respect API limits

Example:
    >>> from src.extract import extract
    >>> df_raw = extract()
    >>> print(df_raw.shape)
"""

import time
import requests
import pandas as pd
from typing import Dict, Optional, List, Any
from functools import wraps

# Import configuration
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    API_KEY, BASE_URL, MOVIE_ENDPOINT, CREDITS_ENDPOINT,
    MOVIE_IDS, MAX_RETRIES, RETRY_BACKOFF, INITIAL_WAIT,
    REQUEST_TIMEOUT, RATE_LIMIT_DELAY
)
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger("extract")


def retry_with_backoff(max_retries: int = MAX_RETRIES, 
                       initial_wait: float = INITIAL_WAIT,
                       backoff_multiplier: float = RETRY_BACKOFF):
    """
    Decorator that implements retry logic with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_wait: Initial wait time between retries (seconds)
        backoff_multiplier: Multiplier for exponential backoff
    
    Returns:
        Decorated function with retry capability
    
    Example:
        >>> @retry_with_backoff(max_retries=3)
        ... def fetch_data():
        ...     # API call here
        ...     pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            wait_time = initial_wait
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.RequestException as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed: {str(e)}. "
                            f"Retrying in {wait_time:.1f}s..."
                        )
                        time.sleep(wait_time)
                        wait_time *= backoff_multiplier
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts failed for {func.__name__}"
                        )
            
            raise last_exception
        return wrapper
    return decorator


@retry_with_backoff()
def fetch_movie(movie_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch movie details from TMDB API.
    
    Args:
        movie_id: The TMDB movie ID to fetch
    
    Returns:
        Dictionary containing movie data, or None if not found
    
    Raises:
        requests.exceptions.RequestException: If all retry attempts fail
    
    Example:
        >>> movie_data = fetch_movie(299534)  # Avengers: Endgame
        >>> print(movie_data['title'])
        'Avengers: Endgame'
    """
    url = f"{BASE_URL}{MOVIE_ENDPOINT.format(movie_id=movie_id)}"
    params = {"api_key": API_KEY}
    
    logger.debug(f"Fetching movie ID: {movie_id}")
    
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    
    # Handle 404 (movie not found) without raising exception
    if response.status_code == 404:
        logger.warning(f"Movie ID {movie_id} not found (404)")
        return None
    
    # Raise exception for other error codes (will trigger retry)
    response.raise_for_status()
    
    logger.debug(f"Successfully fetched movie ID: {movie_id}")
    return response.json()


@retry_with_backoff()
def fetch_credits(movie_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch cast and crew information for a movie from TMDB API.
    
    Args:
        movie_id: The TMDB movie ID to fetch credits for
    
    Returns:
        Dictionary containing cast and crew data, or None if not found
    
    Raises:
        requests.exceptions.RequestException: If all retry attempts fail
    
    Example:
        >>> credits = fetch_credits(299534)
        >>> print(len(credits['cast']))  # Number of cast members
    """
    url = f"{BASE_URL}{CREDITS_ENDPOINT.format(movie_id=movie_id)}"
    params = {"api_key": API_KEY}
    
    logger.debug(f"Fetching credits for movie ID: {movie_id}")
    
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    
    # Handle 404 (movie not found) without raising exception
    if response.status_code == 404:
        logger.warning(f"Credits for movie ID {movie_id} not found (404)")
        return None
    
    # Raise exception for other error codes (will trigger retry)
    response.raise_for_status()
    
    logger.debug(f"Successfully fetched credits for movie ID: {movie_id}")
    return response.json()


def process_credits(credits_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Process credits data to extract key information.
    
    Args:
        credits_data: Raw credits data from API
    
    Returns:
        Dictionary with processed cast and crew info:
        - cast: Top 5 cast members (pipe-separated)
        - cast_size: Total number of cast members
        - director: Director name(s)
        - crew_size: Total number of crew members
    
    Example:
        >>> credits = fetch_credits(299534)
        >>> processed = process_credits(credits)
        >>> print(processed['director'])
        'Anthony Russo|Joe Russo'
    """
    if not credits_data:
        return {
            "cast": None,
            "cast_size": 0,
            "director": None,
            "crew_size": 0
        }
    
    # Extract top 5 cast members
    cast_list = credits_data.get("cast", [])
    top_cast = [member["name"] for member in cast_list[:5]]
    cast_str = "|".join(top_cast) if top_cast else None
    
    # Extract director(s)
    crew_list = credits_data.get("crew", [])
    directors = [
        member["name"] 
        for member in crew_list 
        if member.get("job") == "Director"
    ]
    director_str = "|".join(directors) if directors else None
    
    return {
        "cast": cast_str,
        "cast_size": len(cast_list),
        "director": director_str,
        "crew_size": len(crew_list)
    }


def extract(movie_ids: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Main extraction function - fetches all movie data from TMDB API.
    
    This function:
    1. Iterates through all movie IDs
    2. Fetches movie details and credits for each
    3. Combines data into a pandas DataFrame
    4. Implements rate limiting between requests
    
    Args:
        movie_ids: List of movie IDs to fetch (default: uses MOVIE_IDS from config)
    
    Returns:
        pandas.DataFrame: Raw movie data with cast/crew information
    
    Example:
        >>> df_raw = extract()
        >>> print(f"Extracted {len(df_raw)} movies")
        Extracted 18 movies
    """
    if movie_ids is None:
        movie_ids = MOVIE_IDS
    
    logger.info(f"Starting extraction for {len(movie_ids)} movies...")
    
    movies_data = []
    successful = 0
    failed = 0
    
    for i, movie_id in enumerate(movie_ids, 1):
        logger.info(f"Processing movie {i}/{len(movie_ids)} (ID: {movie_id})")
        
        try:
            # Fetch movie details
            movie_data = fetch_movie(movie_id)
            
            if movie_data is None:
                failed += 1
                continue
            
            # Fetch and process credits
            credits_data = fetch_credits(movie_id)
            credits_info = process_credits(credits_data)
            
            # Combine movie data with credits
            movie_data.update(credits_info)
            movies_data.append(movie_data)
            
            successful += 1
            logger.info(f" Successfully extracted: {movie_data.get('title', 'Unknown')}")
            
        except requests.exceptions.RequestException as e:
            failed += 1
            logger.error(f" Failed to extract movie ID {movie_id}: {str(e)}")
        
        # Rate limiting - wait between requests
        if i < len(movie_ids):
            time.sleep(RATE_LIMIT_DELAY)
    
    # Create DataFrame
    df = pd.DataFrame(movies_data)
    
    logger.info(f"Extraction complete! Success: {successful}, Failed: {failed}")
    logger.info(f"DataFrame shape: {df.shape}")
    
    return df


if __name__ == "__main__":
    # Test extraction with a small subset
    print("Testing extraction module...")
    test_df = extract([299534, 19995])  # Test with 2 movies
    print(f"\nTest DataFrame shape: {test_df.shape}")
    print(f"Columns: {list(test_df.columns)}")
