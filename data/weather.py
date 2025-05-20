"""
Weather data module for AirSense Copenhagen.
Handles fetching weather data from Open-Meteo API.
"""

import requests
import pandas as pd
import logging
from datetime import datetime, timedelta
from config import WEATHER_API_URL, WEATHER_VARIABLES, DEFAULT_HISTORY_DAYS

logger = logging.getLogger(__name__)

def get_weather_data(latitude, longitude, start_date=None, end_date=None, days_back=None):
    """
    Fetch weather data from Open-Meteo API.
    
    Parameters:
    latitude (float): Location latitude
    longitude (float): Location longitude
    start_date (str, optional): Start date in YYYY-MM-DD format
    end_date (str, optional): End date in YYYY-MM-DD format
    days_back (int, optional): Number of days back from today (if start_date not provided)
    
    Returns:
    DataFrame: Hourly weather data or None if fetch failed
    """
    try:
        # Handle date parameters
        if not start_date:
            if days_back is None:
                days_back = DEFAULT_HISTORY_DAYS
                
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        elif not end_date:
            # If only start_date is provided, use current date as end_date
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        logger.info(f"Fetching weather data from {start_date} to {end_date} for coordinates ({latitude}, {longitude})")
        
        # Create empty dataframe to store all results
        all_weather_data = pd.DataFrame()
        
        # Process in 3-month chunks (typical API limitation)
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        current_start = start_dt
        
        while current_start < end_dt:
            # Calculate chunk end (3 months later or end_dt, whichever comes first)
            current_end = min(current_start + timedelta(days=90), end_dt)
            
            chunk_start = current_start.strftime("%Y-%m-%d")
            chunk_end = current_end.strftime("%Y-%m-%d")
            logger.info(f"  Fetching weather chunk: {chunk_start} to {chunk_end}")
            
            # Construct API URL for this chunk with Copenhagen timezone
            url = f"{WEATHER_API_URL}?latitude={latitude}&longitude={longitude}&start_date={chunk_start}&end_date={chunk_end}&hourly={','.join(WEATHER_VARIABLES)}&timezone=Europe/Copenhagen"
            
            # Make the request
            response = requests.get(url)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            data = response.json()
            hourly_data = data.get("hourly", {})
            
            if not hourly_data or "time" not in hourly_data:
                logger.warning(f"  API response missing hourly data for chunk {chunk_start} to {chunk_end}")
            else:
                chunk_df = pd.DataFrame(hourly_data)
                
                if not chunk_df.empty and 'time' in chunk_df.columns:
                    chunk_df["time"] = pd.to_datetime(chunk_df["time"])
                    all_weather_data = pd.concat([all_weather_data, chunk_df])
                    logger.info(f"  Retrieved {len(chunk_df)} records for chunk {chunk_start} to {chunk_end}")
                    
            # Move to next chunk
            current_start = current_end + timedelta(days=1)
        
        # Process the combined data
        if all_weather_data.empty:
            logger.error("No weather data retrieved for the specified period")
            return None
        
        # Remove any duplicates that might have been created at chunk boundaries
        all_weather_data = all_weather_data.drop_duplicates(subset=['time'])
        
        # Sort by time to ensure chronological order
        all_weather_data = all_weather_data.sort_values('time')
        
        # Success statistics
        logger.info(f"Successfully fetched {len(all_weather_data)} hours of weather data")
        if len(all_weather_data) > 0:
            logger.info(f"Date range: {all_weather_data['time'].min()} to {all_weather_data['time'].max()}")
            
        return all_weather_data
        
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error: {e}")
        return None
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error: {e}")
        return None
    except requests.exceptions.Timeout as e:
        logger.error(f"Request timed out: {e}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching weather data: {e}")
        return None
