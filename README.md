#  TMDB Movie Data Analysis Pipeline

### 1. Introduction 

This project analyzed movie data collected from the TMDB API. The goal was to clean the dataset, transform key fields, and extract insights about movie performance over the years.



##  Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Stages](#pipeline-stages)
- [Analysis Features](#analysis-features)
- [Visualizations](#visualizations)
- [Configuration](#configuration)
- [Logging](#logging)

---

## Overview

This project builds a complete movie data analysis pipeline that:

1. **Extracts** movie data from the TMDB API
2. **Transforms** and cleans the raw data
3. **Loads** processed data to CSV files
4. **Analyzes** movies using various KPIs
5. **Visualizes** insights with Matplotlib charts

### Key Features

-  **Retry Logic**: Automatic retries with exponential backoff for API failures
-  **Structured Logging**: Console and file logging for debugging
-  **Modular Design**: Separate modules for each pipeline stage
-  **Comprehensive Analysis**: Rankings, search queries, franchise comparisons
-  **Beautiful Visualizations**: 5 chart types for data insights

---

## Project Structure

```
MOVIE_DATA_ANALYSIS/
├── 📁 config/
│   ├── __init__.py
│   └── config.py           # API keys, settings, constants
│
├── 📁 src/
│   ├── __init__.py
│   ├── extract.py          # TMDB API data fetching
│   ├── transform.py        # Data cleaning & preprocessing
│   ├── load.py             # Data persistence (CSV)
│   ├── analysis.py         # KPI calculations & queries
│   ├── visualize.py        # Chart generation
│   └── 📁 utils/
│       ├── __init__.py
│       └── logger.py       # Structured logging setup
│
├── 📁 data/
│   ├── 📁 raw/             # Raw API responses
│   └── 📁 processed/       # Cleaned data
│
├── 📁 visualizations/      # Generated charts (PNG)
├── 📁 logs/                # Log files
│
├── 📓 orchestrator.ipynb   # Main Jupyter notebook
├── 📄 README.md            # This file
├── 📄 requirements.txt     # Python dependencies
└── 📄 .gitignore
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR NOTEBOOK                        │
│                    (orchestrator.ipynb)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   EXTRACT     │    │   TRANSFORM   │    │     LOAD      │
│  (extract.py) │───▶│(transform.py) │───▶│   (load.py)   │
│               │    │               │    │               │
│ • TMDB API    │    │ • Parse JSON  │    │ • Save CSV    │
│ • Retry logic │    │ • Clean data  │    │ • Load CSV    │
│ • Rate limit  │    │ • Type convert│    │               │
└───────────────┘    └───────────────┘    └───────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
┌───────────────┐                          ┌───────────────┐
│   ANALYSIS    │                          │  VISUALIZE    │
│ (analysis.py) │                          │(visualize.py) │
│               │                          │               │
│ • Rankings    │                          │ • Matplotlib  │
│ • Search      │                          │ • 5 Charts    │
│ • Comparisons │                          │ • PNG export  │
└───────────────┘                          └───────────────┘
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository** (or download the project):
   ```bash
   cd MOVIE_DATA_ANALYSIS
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Option 1: Jupyter Notebook (Recommended)

Open and run the orchestrator notebook:

```bash
jupyter notebook orchestrator.ipynb
```

Run cells sequentially to execute the full pipeline.

### Option 2: Python Script

You can also run the pipeline stages individually:

```python
from src.extract import extract
from src.transform import transform
from src.load import save_raw, save_processed
from src.analysis import get_best_worst_movies
from src.visualize import create_all_visualizations

# Step 1: Extract
df_raw = extract()
save_raw(df_raw)

# Step 2: Transform
df_clean = transform(df_raw)
save_processed(df_clean)

# Step 3: Analyze
rankings = get_best_worst_movies(df_clean)

# Step 4: Visualize
create_all_visualizations(df_clean)
```

---

## Pipeline Stages

### 1. Extract (`src/extract.py`)

Fetches movie data from the TMDB API.

| Function | Description |
|----------|-------------|
| `extract()` | Main function - returns DataFrame of all movies |
| `fetch_movie(id)` | Fetch single movie with retry logic |
| `fetch_credits(id)` | Fetch cast/crew information |

**Features:**
- 3 retries with exponential backoff
- 10-second request timeout
- 0.25s rate limiting between requests
- Comprehensive error logging

### 2. Transform (`src/transform.py`)

Cleans and preprocesses the raw data.

| Function | Description |
|----------|-------------|
| `transform(df)` | Main function - orchestrates all cleaning |
| `parse_json_columns(df)` | Extract data from JSON-like columns |
| `convert_datatypes(df)` | Convert to appropriate types |
| `handle_missing_values(df)` | Replace zeros and placeholders |

**Data Transformations:**
- Drops irrelevant columns (adult, imdb_id, etc.)
- Parses genres, companies, countries into pipe-separated strings
- Converts budget/revenue to millions USD
- Converts release_date to datetime

### 3. Load (`src/load.py`)

Handles data persistence to CSV files.

| Function | Description |
|----------|-------------|
| `save_raw(df)` | Save to `data/raw/movies_raw.csv` |
| `save_processed(df)` | Save to `data/processed/movies_clean.csv` |
| `load_processed()` | Load previously saved clean data |

### 4. Analysis (`src/analysis.py`)

Calculates KPIs and implements search queries.

| Function | Description |
|----------|-------------|
| `get_best_worst_movies(df)` | All ranking categories |
| `search_movies(df, ...)` | Advanced filtering |
| `franchise_vs_standalone(df)` | Compare performance |
| `top_franchises(df)` | Rank by revenue/budget |
| `top_directors(df)` | Rank by movies/revenue |

### 5. Visualize (`src/visualize.py`)

Generates charts using Matplotlib.

| Function | Description |
|----------|-------------|
| `create_all_visualizations(df)` | Generate all 5 charts |
| `plot_revenue_vs_budget(df)` | Scatter plot |
| `plot_roi_by_genre(df)` | Bar chart |
| `plot_popularity_vs_rating(df)` | Scatter with trend |
| `plot_yearly_trends(df)` | Line chart |
| `plot_franchise_comparison(df)` | Grouped bar chart |

---

## Analysis Features

### Movie Rankings

The pipeline calculates rankings for:

1. **Highest/Lowest Revenue**
2. **Highest/Lowest Budget**
3. **Highest/Lowest Profit** (Revenue - Budget)
4. **Highest/Lowest ROI** (Budget ≥ $10M filter)
5. **Most Voted Movies**
6. **Highest/Lowest Rated** (≥10 votes filter)
7. **Most Popular Movies**

### Advanced Search Queries

```python
# Example: Find Sci-Fi Action movies with Bruce Willis
results = search_movies(
    df, 
    genres=['Science Fiction', 'Action'],
    cast='Bruce Willis',
    sort_by='vote_average'
)
```

### Franchise Analysis

Compare franchise movies vs standalone:
- Mean Revenue
- Median ROI
- Mean Budget
- Mean Popularity
- Mean Rating

---

## Visualizations

The pipeline generates 5 charts:

| Chart | Description |
|-------|-------------|
| `revenue_vs_budget.png` | Scatter plot with break-even line |
| `roi_by_genre.png` | Horizontal bar chart by genre |
| `popularity_vs_rating.png` | Scatter with trend line |
| `yearly_trends.png` | Dual-axis line chart |
| `franchise_comparison.png` | Grouped bar chart |

---

## Configuration

All settings are in `config/config.py`:

```python
# API Settings
API_KEY = "your_api_key"
BASE_URL = "https://api.themoviedb.org/3"

# Retry Settings
MAX_RETRIES = 3
REQUEST_TIMEOUT = 10

# Analysis Settings
MIN_BUDGET_FOR_ROI = 10  # Million USD
MIN_VOTES_FOR_RATING = 10
```

---

## Logging

Logs are written to both console and `logs/pipeline.log`.

**Log Format:**
```
2024-01-15 10:30:45 | INFO     | extract | Starting extraction for 19 movies...
2024-01-15 10:30:46 | INFO     | extract | ✓ Successfully extracted: Avengers: Endgame
```

**Log Levels:**
- `DEBUG`: Detailed information for debugging
- `INFO`: General progress messages
- `WARNING`: Non-critical issues
- `ERROR`: Errors that prevented an operation

---


