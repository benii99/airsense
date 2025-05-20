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
from datetime import datetime

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
    
    # Make y-axis 15% taller to accommodate labels
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min, y_max + y_range * 0.15)
    
    # Add values above points with improved positioning
    for i, v in enumerate(hourly_avg[value_col]):
        # Position labels with adequate vertical spacing and limit decimal places
        ax.text(hourly_avg['hour_of_day'].iloc[i], v + (y_range * 0.025), 
                f"{v:.1f}", ha='center', fontsize=9, 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
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
        fig: The matplotlib figure object or None if unable to create
    """
    # Extract hour from datetime
    if 'hour_of_day' not in df.columns:
        if 'datetime' in df.columns:
            df = df.copy()  # Avoid modifying original DataFrame
            df['hour_of_day'] = df['datetime'].dt.hour
        else:
            logger.error("DataFrame must contain 'datetime' column")
            return None
    
    # Check if we have enough valid data
    valid_data = df.dropna(subset=[value_col])
    if len(valid_data) < 24:  # At least need some data for each hour
        logger.warning(f"Insufficient valid data for {value_col} boxplots (only {len(valid_data)} valid rows)")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    try:
        # Create box plots - handling potential errors
        sns.boxplot(x='hour_of_day', y=value_col, data=valid_data, ax=ax)
        
        # Format plot
        ax.set_title(f"Distribution of {value_col} by Hour of Day", fontsize=14)
        ax.set_xlabel("Hour of Day", fontsize=12)
        ax.set_ylabel(value_col, fontsize=12)
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
    except Exception as e:
        logger.error(f"Error creating boxplot for {value_col}: {str(e)}")
        plt.close(fig)
        return None


def create_joint_pollutant_visualizations(df, pollutant_cols, time_col='time', output_dir=None):
    """
    Create joint visualizations for multiple pollutants.
    
    Args:
        df: DataFrame with pollutant data
        pollutant_cols: List of column names for pollutants
        time_col: Column name for time data
        output_dir: Directory to save output figures
    """
    if not pollutant_cols or len(pollutant_cols) == 0:
        return
    
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d")
    
    # Ensure output directory exists
    if output_dir is None:
        output_dir = "figures/air_quality_eda"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Create time series plot for all pollutants
    plt.figure(figsize=(15, 8))
    
    for pollutant in pollutant_cols:
        # Scale value to fit on same plot (normalize to 0-1)
        series = df[pollutant]
        if series.max() == series.min():
            # Handle constant case
            normalized = series / series.max() if series.max() != 0 else series
        else:
            normalized = (series - series.min()) / (series.max() - series.min())
        
        plt.plot(df[time_col], normalized, label=pollutant, alpha=0.7)
    
    plt.title("Normalized Pollutant Levels Over Time")
    plt.xlabel("Time")
    plt.ylabel("Normalized Level (0-1)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{timestamp}_joint_pollutant_timeseries.png", dpi=300)
    plt.close()
    
    # 2. Create hourly average comparison
    df_with_hour = df.copy()
    df_with_hour['hour_of_day'] = df_with_hour[time_col].dt.hour
    
    hourly_avgs = {}
    for pollutant in pollutant_cols:
        hourly_avgs[pollutant] = df_with_hour.groupby('hour_of_day')[pollutant].mean()
    
    plt.figure(figsize=(15, 8))
    
    for pollutant, values in hourly_avgs.items():
        plt.plot(values.index, values.values, 'o-', label=pollutant, alpha=0.7)
    
    plt.title("Average Pollutant Levels by Hour of Day")
    plt.xlabel("Hour of Day")
    plt.ylabel("Concentration")
    plt.legend()
    plt.xticks(range(24))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{timestamp}_joint_pollutant_hourly_averages.png", dpi=300)
    plt.close()
    
    # 3. Create histograms for all pollutants
    fig, axes = plt.subplots(len(pollutant_cols), 1, figsize=(12, 3*len(pollutant_cols)))
    
    for i, pollutant in enumerate(pollutant_cols):
        ax = axes[i] if len(pollutant_cols) > 1 else axes
        
        ax.hist(df[pollutant].dropna(), bins=50, alpha=0.7)
        ax.set_title(f"Distribution of {pollutant}")
        ax.set_xlabel("Concentration")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{timestamp}_joint_pollutant_histograms.png", dpi=300)
    plt.close()
    
    # 4. Create correlation heatmap between pollutants
    plt.figure(figsize=(10, 8))
    corr_matrix = df[pollutant_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title("Correlation Between Pollutants")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{timestamp}_pollutant_correlation_heatmap.png", dpi=300)
    plt.close()

def create_joint_weather_visualizations(df, weather_cols, time_col='time', output_dir=None):
    """
    Create joint visualizations for multiple weather variables.
    
    Args:
        df: DataFrame with weather data
        weather_cols: List of column names for weather variables
        time_col: Column name for time data
        output_dir: Directory to save output figures
    """
    if not weather_cols or len(weather_cols) == 0:
        return
    
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d")
    
    # Ensure output directory exists
    if output_dir is None:
        output_dir = "figures/weather_eda"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Create time series plot for all weather variables (normalized)
    plt.figure(figsize=(15, 8))
    
    for variable in weather_cols:
        # Scale value to fit on same plot (normalize to 0-1)
        series = df[variable]
        if series.max() == series.min():
            # Handle constant case
            normalized = series / series.max() if series.max() != 0 else series
        else:
            normalized = (series - series.min()) / (series.max() - series.min())
        
        plt.plot(df[time_col], normalized, label=variable, alpha=0.7)
    
    plt.title("Normalized Weather Variables Over Time")
    plt.xlabel("Time")
    plt.ylabel("Normalized Value (0-1)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{timestamp}_joint_weather_timeseries.png", dpi=300)
    plt.close()
    
    # 2. Create hourly average comparison
    df_with_hour = df.copy()
    df_with_hour['hour_of_day'] = df_with_hour[time_col].dt.hour
    
    hourly_avgs = {}
    for variable in weather_cols:
        hourly_avgs[variable] = df_with_hour.groupby('hour_of_day')[variable].mean()
    
    # Plot in two subplots to avoid overcrowding
    plt.figure(figsize=(18, 10))
    
    # First half of variables
    ax1 = plt.subplot(2, 1, 1)
    for i, (variable, values) in enumerate(hourly_avgs.items()):
        if i < len(hourly_avgs) // 2:
            ax1.plot(values.index, values.values, 'o-', label=variable, alpha=0.7)
    
    ax1.set_title("Average Weather Variables by Hour of Day (Part 1)")
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.set_xticks(range(24))
    ax1.grid(True, alpha=0.3)
    
    # Second half of variables
    ax2 = plt.subplot(2, 1, 2)
    for i, (variable, values) in enumerate(hourly_avgs.items()):
        if i >= len(hourly_avgs) // 2:
            ax2.plot(values.index, values.values, 'o-', label=variable, alpha=0.7)
    
    ax2.set_title("Average Weather Variables by Hour of Day (Part 2)")
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Value")
    ax2.legend()
    ax2.set_xticks(range(24))
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{timestamp}_joint_weather_hourly_averages.png", dpi=300)
    plt.close()
    
    # 3. Create histograms for all weather variables
    fig, axes = plt.subplots(len(weather_cols), 1, figsize=(12, 3*len(weather_cols)))
    
    for i, variable in enumerate(weather_cols):
        ax = axes[i] if len(weather_cols) > 1 else axes
        
        ax.hist(df[variable].dropna(), bins=50, alpha=0.7)
        ax.set_title(f"Distribution of {variable}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{timestamp}_joint_weather_histograms.png", dpi=300)
    plt.close()
    
    # 4. Create correlation heatmap between weather variables
    plt.figure(figsize=(12, 10))
    corr_matrix = df[weather_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title("Correlation Between Weather Variables")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{timestamp}_weather_correlation_heatmap.png", dpi=300)
    plt.close()
