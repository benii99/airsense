"""
Visualization utilities for AirSense Copenhagen.
Provides functions for exploratory data analysis visualizations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import logging

logger = logging.getLogger(__name__)

def plot_time_series_with_acf_pacf(df, value_col='traffic_count', time_col='datetime', 
                                  title=None, figsize=(15, 12), lags=48, output_path=None):
    """
    Plot time series data with ACF and PACF plots.
    
    Args:
        df: DataFrame with time series data
        value_col: Column name for the values to plot
        time_col: Column name for the time values
        title: Title for the plot (if None, uses value_col)
        figsize: Figure size as (width, height)
        lags: Number of lags for ACF/PACF
        output_path: Path to save the figure
        
    Returns:
        fig: The matplotlib figure object
    """
    if title is None:
        title = f"{value_col} Time Series"
        
    # Check if we have enough data
    if len(df) < 30:
        logger.warning(f"Insufficient data for time series analysis: {len(df)} rows")
        return None
        
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    
    # Plot 1: Time Series
    ax1 = plt.subplot(3, 1, 1)
    plt.plot(df[time_col], df[value_col], linewidth=1)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Prepare data for ACF and PACF
    # Make sure values are sorted by time
    sorted_df = df.sort_values(time_col)
    
    # Check for missing values
    if sorted_df[value_col].isna().any():
        logger.warning(f"Time series contains {sorted_df[value_col].isna().sum()} missing values; interpolating for ACF/PACF")
        values = sorted_df[value_col].interpolate()
    else:
        values = sorted_df[value_col]
    
    # Plot 2: ACF
    ax2 = plt.subplot(3, 1, 2)
    try:
        plot_acf(values, lags=min(lags, len(values)//2), ax=ax2)
        ax2.set_title(f"Autocorrelation Function (ACF)", fontsize=12)
    except Exception as e:
        logger.error(f"Error creating ACF plot: {e}")
        ax2.text(0.5, 0.5, f"Error creating ACF: {str(e)}", 
                 horizontalalignment='center', verticalalignment='center')
    
    # Plot 3: PACF
    ax3 = plt.subplot(3, 1, 3)
    try:
        plot_pacf(values, lags=min(lags, len(values)//2), ax=ax3)
        ax3.set_title(f"Partial Autocorrelation Function (PACF)", fontsize=12)
    except Exception as e:
        logger.error(f"Error creating PACF plot: {e}")
        ax3.text(0.5, 0.5, f"Error creating PACF: {str(e)}", 
                 horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    
    # Save figure if output_path provided
    if output_path:
        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved time series plot to {output_path}")
    
    return fig

def plot_average_by_hour(df, value_col='traffic_count', figsize=(12, 6), output_path=None):
    """
    Plot average value by hour of day.
    
    Args:
        df: DataFrame with datetime and value data
        value_col: Column to average
        figsize: Figure size
        output_path: Path to save the figure
        
    Returns:
        fig: The matplotlib figure object
    """
    # Extract hour from datetime
    if 'hour_of_day' not in df.columns:
        if 'datetime' in df.columns:
            df = df.copy()  # Avoid modifying original DataFrame
            df['hour_of_day'] = df['datetime'].dt.hour
        else:
            logger.error("DataFrame must contain 'datetime' column")
            return None
    
    # Calculate average by hour
    hourly_avg = df.groupby('hour_of_day')[value_col].mean().reset_index()
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Line plot
    ax.plot(hourly_avg['hour_of_day'], hourly_avg[value_col], marker='o', linewidth=2, color='royalblue')
    
    # Format plot
    ax.set_title(f"Average {value_col} by Hour of Day", fontsize=14)
    ax.set_xlabel("Hour of Day", fontsize=12)
    ax.set_ylabel(value_col, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(24))
    ax.set_xlim(-0.5, 23.5)
    
    # Add values above points
    for i, v in enumerate(hourly_avg[value_col]):
        ax.text(hourly_avg['hour_of_day'].iloc[i], v + (v * 0.03), 
                f"{v:.1f}", ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure if output_path provided
    if output_path:
        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved hourly average plot to {output_path}")
    
    return fig

def plot_hourly_boxplots(df, value_col='traffic_count', figsize=(15, 8), output_path=None):
    """
    Create box plots of values for each hour of day.
    
    Args:
        df: DataFrame with datetime and value data
        value_col: Column to analyze
        figsize: Figure size
        output_path: Path to save the figure
        
    Returns:
        fig: The matplotlib figure object
    """
    # Extract hour from datetime
    if 'hour_of_day' not in df.columns:
        if 'datetime' in df.columns:
            df = df.copy()  # Avoid modifying original DataFrame
            df['hour_of_day'] = df['datetime'].dt.hour
        else:
            logger.error("DataFrame must contain 'datetime' column")
            return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create box plots
    sns.boxplot(x='hour_of_day', y=value_col, data=df, ax=ax)
    
    # Format plot
    ax.set_title(f"Distribution of {value_col} by Hour of Day", fontsize=14)
    ax.set_xlabel("Hour of Day", fontsize=12)
    ax.set_ylabel(value_col, fontsize=12)
    ax.set_xticks(range(24))
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure if output_path provided
    if output_path:
        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved hourly boxplot to {output_path}")
    
    return fig
