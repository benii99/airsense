"""
Traffic data module for AirSense Copenhagen.
Handles loading traffic data from CSV files.
"""

import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

def get_traffic_data(file_path, location_name=None):
    """
    Load traffic data from CSV file, optionally filtering by location.
    
    Args:
        file_path (str): Path to the traffic data CSV file
        location_name (str, optional): Location to filter by
        
    Returns:
        DataFrame: Traffic data, or None if loading failed
    """
    # Check if file exists
    if not os.path.exists(file_path):
        logger.error(f"Traffic data file not found: {file_path}")
        return None
    
    try:
        # Load CSV file with semicolon delimiter
        logger.info(f"Loading traffic data from {file_path}")
        df = pd.read_csv(file_path, delimiter=';')
        
        # Convert datetime to proper format if needed
        if 'datetime' in df.columns and df['datetime'].dtype == 'object':
            logger.info("Converting datetime column to datetime type")
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', format='%d.%m.%y %H:%M')
            
        # Convert traffic_count to numeric
        if 'traffic_count' in df.columns:
            df['traffic_count'] = pd.to_numeric(df['traffic_count'], errors='coerce')
        
        # Filter by location if specified
        if location_name and 'location' in df.columns:
            logger.info(f"Filtering traffic data for location: {location_name}")
            filtered_df = df[df['location'] == location_name]
            
            if len(filtered_df) == 0:
                logger.warning(f"No traffic data found for location '{location_name}'")
                return filtered_df
                
            logger.info(f"Filtered to {len(filtered_df)} records for location '{location_name}'")
            return filtered_df
        
        # Log summary statistics
        logger.info(f"Loaded traffic data with {len(df)} records")
        if 'datetime' in df.columns:
            logger.info(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        if 'location' in df.columns:
            logger.info(f"Number of unique locations: {df['location'].nunique()}")
            logger.info(f"Locations: {', '.join(df['location'].unique())}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading traffic data: {e}")
        return None
