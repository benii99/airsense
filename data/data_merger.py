"""
Data merger module for AirSense Copenhagen.
Handles combining traffic, air quality, and weather data into a unified dataset.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)

def merge_datasets(traffic_df=None, air_quality_df=None, weather_df=None, time_col='time'):
    """
    Merge traffic, air quality, and weather datasets based on time column.
    
    Parameters:
    traffic_df (DataFrame or tuple, optional): DataFrame with traffic data or tuple containing it
    air_quality_df (DataFrame or tuple, optional): DataFrame with air quality data or tuple containing it
    weather_df (DataFrame or tuple, optional): DataFrame with weather data or tuple containing it
    time_col (str): Name of the time column for merging
    
    Returns:
    DataFrame: Merged dataset with all metrics
    """
    # Extract DataFrames from tuples if needed
    if isinstance(traffic_df, tuple):
        logger.info("Traffic data provided as a tuple, extracting DataFrame")
        # Assuming the first element is the DataFrame
        traffic_df = traffic_df[0] if traffic_df else None
    
    if isinstance(air_quality_df, tuple):
        logger.info("Air quality data provided as a tuple, extracting DataFrame")
        air_quality_df = air_quality_df[0] if air_quality_df else None
    
    if isinstance(weather_df, tuple):
        logger.info("Weather data provided as a tuple, extracting DataFrame")
        weather_df = weather_df[0] if weather_df else None
    
    if traffic_df is None and air_quality_df is None and weather_df is None:
        logger.error("No datasets provided for merging")
        return None
    
    logger.info("Merging datasets")
    
    # Track which datasets are available
    available_datasets = []
    if traffic_df is not None and len(traffic_df) > 0:
        available_datasets.append("traffic")
    if air_quality_df is not None and len(air_quality_df) > 0:
        available_datasets.append("air quality")
    if weather_df is not None and len(weather_df) > 0:
        available_datasets.append("weather")
        
    logger.info(f"Datasets available for merging: {', '.join(available_datasets)}")
    
    # Ensure all dataframes have the same time column name
    if traffic_df is not None and 'datetime' in traffic_df.columns and time_col == 'time':
        traffic_df = traffic_df.copy()
        traffic_df = traffic_df.rename(columns={'datetime': time_col})
        logger.info("Renamed traffic_df 'datetime' column to 'time'")
    
    # Start with the first available dataset
    merged_df = None
    
    # Add traffic data if available
    if traffic_df is not None and len(traffic_df) > 0:
        merged_df = traffic_df.copy()
        logger.info(f"Starting with traffic data: {len(merged_df)} rows")
    
    # Add air quality data if available
    if air_quality_df is not None and len(air_quality_df) > 0:
        if merged_df is not None:
            # Merge with existing data
            merged_df = pd.merge(merged_df, air_quality_df, on=time_col, how='outer')
            logger.info(f"Merged with air quality data: {len(merged_df)} rows")
        else:
            # Start with air quality data
            merged_df = air_quality_df.copy()
            logger.info(f"Starting with air quality data: {len(merged_df)} rows")
    
    # Add weather data if available
    if weather_df is not None and len(weather_df) > 0:
        if merged_df is not None:
            # Merge with existing data
            merged_df = pd.merge(merged_df, weather_df, on=time_col, how='outer')
            logger.info(f"Merged with weather data: {len(merged_df)} rows")
        else:
            # Start with weather data
            merged_df = weather_df.copy()
            logger.info(f"Starting with weather data: {len(merged_df)} rows")
    
    # Sort by time
    if merged_df is not None:
        merged_df = merged_df.sort_values(time_col)
        
        # Log merge statistics
        total_rows = len(merged_df)
        complete_rows = merged_df.dropna().shape[0]
        logger.info(f"Final merged dataset: {total_rows} rows ({complete_rows} complete rows)")
        
        # Check timestamp range
        if total_rows > 0:
            start_time = merged_df[time_col].min()
            end_time = merged_df[time_col].max()
            logger.info(f"Timestamp range: {start_time} to {end_time}")
    else:
        logger.error("Failed to create merged dataset")
    
    return merged_df
