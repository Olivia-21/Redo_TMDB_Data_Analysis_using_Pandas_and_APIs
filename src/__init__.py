"""
Source package for TMDB Movie Data Analysis Pipeline.

This package contains all ETL modules:
- extract: API data fetching
- transform: Data cleaning and preprocessing
- load: Data persistence
- analysis: KPI calculations
- visualize: Chart generation
"""

from .extract import extract
from .transform import transform
from .load import save_raw, save_processed, load_processed
from .analysis import (
    get_best_worst_movies,
    search_movies,
    franchise_vs_standalone,
    top_franchises,
    top_directors
)
from .visualize import create_all_visualizations
