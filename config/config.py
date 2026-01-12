"""
Configuration settings for TMDB Movie Data Analysis Pipeline.

This module contains all configuration constants including API settings,
file paths, and pipeline parameters.
"""

# =============================================================================
# API CONFIGURATION
# =============================================================================

# TMDB API Settings
API_KEY = "d703444a0cdc339616bf4ed9c0609a3b"
BASE_URL = "https://api.themoviedb.org/3"
MOVIE_ENDPOINT = "/movie/{movie_id}"
CREDITS_ENDPOINT = "/movie/{movie_id}/credits"

# Target Movie IDs (from project requirements)
MOVIE_IDS = [
    0, 299534, 19995, 140607, 299536, 597, 135397, 420818, 
    24428, 168259, 99861, 284054, 12445, 181808, 330457, 
    351286, 109445, 321612, 260513
]

# =============================================================================
# REQUEST SETTINGS
# =============================================================================

# Retry configuration
MAX_RETRIES = 3                  # Number of retry attempts
RETRY_BACKOFF = 2                # Exponential backoff multiplier
INITIAL_WAIT = 1                 # Initial wait time in seconds

# Timeout settings
REQUEST_TIMEOUT = 10             # Timeout in seconds for API requests
RATE_LIMIT_DELAY = 0.25          # Delay between requests (seconds)

# =============================================================================
# FILE PATHS
# =============================================================================

# Data directories
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
VISUALIZATIONS_DIR = "visualizations"
LOGS_DIR = "logs"

# File names
RAW_DATA_FILE = "movies_raw.csv"
PROCESSED_DATA_FILE = "movies_clean.csv"
LOG_FILE = "pipeline.log"

# =============================================================================
# DATA PROCESSING SETTINGS
# =============================================================================

# Columns to drop during transformation
COLUMNS_TO_DROP = ['adult', 'imdb_id', 'original_title', 'video', 'homepage']

# JSON-like columns that need parsing
JSON_COLUMNS = [
    'belongs_to_collection', 
    'genres', 
    'production_countries', 
    'production_companies', 
    'spoken_languages'
]

# Final column order for cleaned dataset
FINAL_COLUMN_ORDER = [
    'id', 'title', 'tagline', 'release_date', 'genres', 
    'belongs_to_collection', 'original_language', 'budget_musd', 
    'revenue_musd', 'production_companies', 'production_countries', 
    'vote_count', 'vote_average', 'popularity', 'runtime', 
    'overview', 'spoken_languages', 'poster_path', 'cast', 
    'cast_size', 'director', 'crew_size'
]

# =============================================================================
# ANALYSIS SETTINGS
# =============================================================================

# Minimum budget for ROI calculations (in millions USD)
MIN_BUDGET_FOR_ROI = 10

# Minimum vote count for rating calculations
MIN_VOTES_FOR_RATING = 10

# Default number of top results to return
DEFAULT_TOP_N = 10
