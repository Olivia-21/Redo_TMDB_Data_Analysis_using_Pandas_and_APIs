"""
Structured logging utility for TMDB Movie Data Analysis Pipeline.

This module provides a configured logger that outputs to both console
and a log file with timestamps and log levels.
"""

import logging
import os
from datetime import datetime


def setup_logger(name: str = "tmdb_pipeline", log_level: int = logging.INFO) -> logging.Logger:
    """
    Set up and configure a logger with console and file handlers.
    
    Args:
        name: Name of the logger (default: "tmdb_pipeline")
        log_level: Logging level (default: logging.INFO)
    
    Returns:
        logging.Logger: Configured logger instance
    
    Example:
        >>> logger = setup_logger()
        >>> logger.info("Starting data extraction...")
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger
    
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Log file path with date
    log_file = os.path.join(logs_dir, "pipeline.log")
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    simple_formatter = logging.Formatter(
        fmt="%(levelname)-8s | %(message)s"
    )
    
    # Console handler (simpler format for readability)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter)
    
    # File handler (detailed format for debugging)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "tmdb_pipeline") -> logging.Logger:
    """
    Get an existing logger or create a new one.
    
    Args:
        name: Name of the logger to retrieve
    
    Returns:
        logging.Logger: Logger instance
    
    Example:
        >>> logger = get_logger()
        >>> logger.info("Processing complete!")
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set it up
    if not logger.handlers:
        return setup_logger(name)
    
    return logger


# Create a default logger instance for easy import
logger = setup_logger()
