"""
Visualization module for TMDB Movie Data Analysis Pipeline.

This module generates charts and plots using Matplotlib:
- Revenue vs Budget trends
- ROI distribution by genre
- Popularity vs Rating scatter plot
- Yearly box office trends
- Franchise vs Standalone comparison

Example:
    >>> from src.visualize import create_all_visualizations
    >>> create_all_visualizations(df_clean)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Optional, List, Dict

# Import configuration
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import VISUALIZATIONS_DIR
from src.utils.logger import get_logger
from src.analysis import calculate_roi, franchise_vs_standalone

# Initialize logger
logger = get_logger("visualize")

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12


def ensure_viz_directory() -> str:
    """
    Ensure visualization directory exists.
    
    Returns:
        Path to visualization directory
    """
    if not os.path.exists(VISUALIZATIONS_DIR):
        os.makedirs(VISUALIZATIONS_DIR)
        logger.debug(f"Created directory: {VISUALIZATIONS_DIR}")
    return VISUALIZATIONS_DIR


def save_figure(fig: plt.Figure, filename: str) -> str:
    """
    Save a figure to the visualizations directory.
    
    Args:
        fig: Matplotlib figure object
        filename: Name of the file (without extension)
    
    Returns:
        Path to saved file
    """
    ensure_viz_directory()
    filepath = os.path.join(VISUALIZATIONS_DIR, f"{filename}.png")
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    logger.info(f"✓ Saved: {filepath}")
    return filepath


def plot_revenue_vs_budget(df: pd.DataFrame) -> str:
    """
    Create scatter plot of Revenue vs Budget trends.
    
    Shows relationship between movie budgets and revenues,
    with a break-even line for reference.
    
    Args:
        df: DataFrame with budget_musd and revenue_musd columns
    
    Returns:
        Path to saved figure
    
    Example:
        >>> filepath = plot_revenue_vs_budget(df_clean)
    """
    logger.info("Creating Revenue vs Budget plot...")
    
    # Filter data
    plot_df = df[['title', 'budget_musd', 'revenue_musd']].dropna()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot
    scatter = ax.scatter(
        plot_df['budget_musd'], 
        plot_df['revenue_musd'],
        c=plot_df['revenue_musd'] - plot_df['budget_musd'],
        cmap='RdYlGn',
        s=100,
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )
    
    # Add break-even line
    max_val = max(plot_df['budget_musd'].max(), plot_df['revenue_musd'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Break-even line')
    
    # Labels and title
    ax.set_xlabel('Budget (Million USD)', fontsize=12)
    ax.set_ylabel('Revenue (Million USD)', fontsize=12)
    ax.set_title('Revenue vs Budget: Movie Financial Performance', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Profit (Million USD)')
    
    # Format axis with currency-like formatting
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x:,.0f}M'))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x:,.0f}M'))
    
    # Add annotations for some notable movies
    for idx, row in plot_df.nlargest(3, 'revenue_musd').iterrows():
        ax.annotate(
            row['title'][:20],
            (row['budget_musd'], row['revenue_musd']),
            textcoords="offset points",
            xytext=(5, 5),
            ha='left',
            fontsize=8,
            alpha=0.8
        )
    
    ax.legend(loc='upper left')
    plt.tight_layout()
    
    return save_figure(fig, 'revenue_vs_budget')


def plot_roi_by_genre(df: pd.DataFrame) -> str:
    """
    Create bar chart of ROI distribution by genre.
    
    Shows average Return on Investment for each movie genre.
    
    Args:
        df: DataFrame with genres and financial columns
    
    Returns:
        Path to saved figure
    
    Example:
        >>> filepath = plot_roi_by_genre(df_clean)
    """
    logger.info("Creating ROI by Genre plot...")
    
    # Calculate ROI
    df = calculate_roi(df)
    
    # Explode genres into separate rows
    genre_df = df[['genres', 'roi_percent']].dropna()
    genre_df = genre_df.copy()
    genre_df['genre_list'] = genre_df['genres'].str.split('|')
    genre_df = genre_df.explode('genre_list')
    genre_df['genre_list'] = genre_df['genre_list'].str.strip()
    
    # Calculate mean ROI by genre
    roi_by_genre = genre_df.groupby('genre_list')['roi_percent'].agg(['mean', 'count'])
    roi_by_genre = roi_by_genre[roi_by_genre['count'] >= 1]  # Filter genres with minimum movies
    roi_by_genre = roi_by_genre.sort_values('mean', ascending=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color bars based on positive/negative ROI
    colors = ['#e74c3c' if x < 0 else '#27ae60' for x in roi_by_genre['mean']]
    
    # Horizontal bar chart
    bars = ax.barh(
        roi_by_genre.index, 
        roi_by_genre['mean'],
        color=colors,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5
    )
    
    # Add value labels
    for bar, val in zip(bars, roi_by_genre['mean']):
        x_pos = bar.get_width() + 5 if val >= 0 else bar.get_width() - 5
        ha = 'left' if val >= 0 else 'right'
        ax.annotate(
            f'{val:.0f}%',
            (x_pos, bar.get_y() + bar.get_height()/2),
            va='center',
            ha=ha,
            fontsize=9
        )
    
    # Labels and title
    ax.set_xlabel('Average ROI (%)', fontsize=12)
    ax.set_ylabel('Genre', fontsize=12)
    ax.set_title('Return on Investment (ROI) by Genre', fontsize=14, fontweight='bold')
    
    # Add zero line
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-')
    
    plt.tight_layout()
    
    return save_figure(fig, 'roi_by_genre')


def plot_popularity_vs_rating(df: pd.DataFrame) -> str:
    """
    Create scatter plot of Popularity vs Rating.
    
    Shows relationship between movie popularity and user ratings.
    
    Args:
        df: DataFrame with popularity and vote_average columns
    
    Returns:
        Path to saved figure
    
    Example:
        >>> filepath = plot_popularity_vs_rating(df_clean)
    """
    logger.info("Creating Popularity vs Rating plot...")
    
    # Filter data
    plot_df = df[['title', 'popularity', 'vote_average', 'vote_count']].dropna()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot with size based on vote count
    sizes = (plot_df['vote_count'] / plot_df['vote_count'].max() * 300) + 50
    
    scatter = ax.scatter(
        plot_df['vote_average'], 
        plot_df['popularity'],
        s=sizes,
        c=plot_df['vote_average'],
        cmap='viridis',
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )
    
    # Add trend line
    z = np.polyfit(plot_df['vote_average'], plot_df['popularity'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(plot_df['vote_average'].min(), plot_df['vote_average'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.5, label='Trend line')
    
    # Labels and title
    ax.set_xlabel('Rating (Vote Average)', fontsize=12)
    ax.set_ylabel('Popularity Score', fontsize=12)
    ax.set_title('Popularity vs Rating: Are Higher-Rated Movies More Popular?', 
                 fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Rating')
    
    # Add annotations for notable movies
    for idx, row in plot_df.nlargest(3, 'popularity').iterrows():
        ax.annotate(
            row['title'][:20],
            (row['vote_average'], row['popularity']),
            textcoords="offset points",
            xytext=(5, 5),
            ha='left',
            fontsize=8,
            alpha=0.8
        )
    
    ax.legend(loc='upper left')
    
    # Add note about bubble size
    ax.text(0.02, 0.02, 'Bubble size = Vote count',
            transform=ax.transAxes, fontsize=9, alpha=0.7)
    
    plt.tight_layout()
    
    return save_figure(fig, 'popularity_vs_rating')


def plot_yearly_trends(df: pd.DataFrame) -> str:
    """
    Create line chart of yearly box office trends.
    
    Shows revenue and budget trends over the years.
    
    Args:
        df: DataFrame with release_date, budget_musd, and revenue_musd columns
    
    Returns:
        Path to saved figure
    
    Example:
        >>> filepath = plot_yearly_trends(df_clean)
    """
    logger.info("Creating Yearly Trends plot...")
    
    # Prepare data
    trend_df = df[['release_date', 'budget_musd', 'revenue_musd']].dropna()
    trend_df = trend_df.copy()
    trend_df['year'] = pd.to_datetime(trend_df['release_date']).dt.year
    
    # Aggregate by year
    yearly_stats = trend_df.groupby('year').agg({
        'budget_musd': ['mean', 'sum'],
        'revenue_musd': ['mean', 'sum']
    })
    yearly_stats.columns = ['avg_budget', 'total_budget', 'avg_revenue', 'total_revenue']
    yearly_stats = yearly_stats.reset_index()
    
    # Create figure with dual y-axis
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot revenue (primary axis)
    color1 = '#27ae60'
    ax1.fill_between(yearly_stats['year'], yearly_stats['total_revenue'], 
                     alpha=0.3, color=color1)
    line1 = ax1.plot(yearly_stats['year'], yearly_stats['total_revenue'], 
                     color=color1, linewidth=2, marker='o', label='Total Revenue')
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Total Revenue (Million USD)', color=color1, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Create secondary y-axis for budget
    ax2 = ax1.twinx()
    color2 = '#3498db'
    ax2.fill_between(yearly_stats['year'], yearly_stats['total_budget'], 
                     alpha=0.2, color=color2)
    line2 = ax2.plot(yearly_stats['year'], yearly_stats['total_budget'], 
                     color=color2, linewidth=2, marker='s', linestyle='--', label='Total Budget')
    ax2.set_ylabel('Total Budget (Million USD)', color=color2, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Title
    ax1.set_title('Yearly Box Office Trends: Revenue vs Budget', fontsize=14, fontweight='bold')
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    # Format x-axis
    ax1.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
    
    plt.tight_layout()
    
    return save_figure(fig, 'yearly_trends')


def plot_franchise_comparison(df: pd.DataFrame) -> str:
    """
    Create bar chart comparing Franchise vs Standalone movie performance.
    
    Compares key metrics between franchise and standalone movies.
    
    Args:
        df: DataFrame with belongs_to_collection column
    
    Returns:
        Path to saved figure
    
    Example:
        >>> filepath = plot_franchise_comparison(df_clean)
    """
    logger.info("Creating Franchise vs Standalone comparison plot...")
    
    # Get comparison data
    comparison = franchise_vs_standalone(df)
    
    # Prepare data for plotting
    metrics = ['revenue_musd', 'budget_musd', 'popularity', 'vote_average']
    labels = ['Mean Revenue\n(M USD)', 'Mean Budget\n(M USD)', 'Mean Popularity', 'Mean Rating']
    
    franchise_vals = [comparison.loc['Franchise', m] for m in metrics]
    standalone_vals = [comparison.loc['Standalone', m] for m in metrics]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(labels))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, franchise_vals, width, label='Franchise', 
                   color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, standalone_vals, width, label='Standalone', 
                   color='#e74c3c', alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=9)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    # Labels and title
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Franchise vs Standalone Movies: Performance Comparison', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Add grid for readability
    ax.yaxis.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return save_figure(fig, 'franchise_comparison')


def create_all_visualizations(df: pd.DataFrame) -> List[str]:
    """
    Generate all visualizations and save them to the visualizations directory.
    
    Creates:
    1. Revenue vs Budget scatter plot
    2. ROI by Genre bar chart
    3. Popularity vs Rating scatter plot
    4. Yearly Trends line chart
    5. Franchise vs Standalone comparison
    
    Args:
        df: Cleaned DataFrame
    
    Returns:
        List of paths to saved figures
    
    Example:
        >>> paths = create_all_visualizations(df_clean)
        >>> print(f"Created {len(paths)} visualizations")
    """
    logger.info("Creating all visualizations...")
    
    saved_paths = []
    
    try:
        # 1. Revenue vs Budget
        saved_paths.append(plot_revenue_vs_budget(df))
    except Exception as e:
        logger.error(f"Failed to create revenue vs budget plot: {e}")
    
    try:
        # 2. ROI by Genre
        saved_paths.append(plot_roi_by_genre(df))
    except Exception as e:
        logger.error(f"Failed to create ROI by genre plot: {e}")
    
    try:
        # 3. Popularity vs Rating
        saved_paths.append(plot_popularity_vs_rating(df))
    except Exception as e:
        logger.error(f"Failed to create popularity vs rating plot: {e}")
    
    try:
        # 4. Yearly Trends
        saved_paths.append(plot_yearly_trends(df))
    except Exception as e:
        logger.error(f"Failed to create yearly trends plot: {e}")
    
    try:
        # 5. Franchise Comparison
        saved_paths.append(plot_franchise_comparison(df))
    except Exception as e:
        logger.error(f"Failed to create franchise comparison plot: {e}")
    
    logger.info(f"Created {len(saved_paths)} visualizations")
    
    # Close all figures to free memory
    plt.close('all')
    
    return saved_paths


if __name__ == "__main__":
    # Test visualization module with sample data
    print("Testing visualization module...")
    
    # Create sample data
    np.random.seed(42)
    n = 20
    sample_data = {
        'id': range(1, n+1),
        'title': [f'Movie {i}' for i in range(1, n+1)],
        'release_date': pd.date_range('2015-01-01', periods=n, freq='3M'),
        'budget_musd': np.random.uniform(10, 200, n),
        'revenue_musd': np.random.uniform(50, 800, n),
        'vote_average': np.random.uniform(5, 9, n),
        'vote_count': np.random.randint(100, 5000, n),
        'popularity': np.random.uniform(20, 200, n),
        'genres': np.random.choice(['Action|Adventure', 'Comedy|Drama', 'Action|Science Fiction', 
                                   'Drama|Romance', 'Horror|Thriller'], n),
        'belongs_to_collection': np.random.choice([None, 'Collection A', 'Collection B'], n)
    }
    
    df_test = pd.DataFrame(sample_data)
    
    # Create all visualizations
    paths = create_all_visualizations(df_test)
    print(f"\nCreated {len(paths)} visualizations:")
    for path in paths:
        print(f"  - {path}")
