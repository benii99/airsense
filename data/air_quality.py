"""
Air quality data module for AirSense Copenhagen.
Handles fetching air quality data from Open-Meteo API.
"""

import requests
import pandas as pd
import logging
from datetime import datetime, timedelta
from config import AIR_QUALITY_API_URL, POLLUTANTS, DEFAULT_HISTORY_DAYS
from data.aqi_calculator import calculate_aqi

logger = logging.getLogger(__name__)

def get_air_quality_data(latitude, longitude, start_date=None, end_date=None, days_back=None, pollutants=None):
    """
    Fetch air quality data from Open-Meteo API.
    
    Parameters:
    latitude (float): Location latitude
    longitude (float): Location longitude
    start_date (str, optional): Start date in YYYY-MM-DD format
    end_date (str, optional): End date in YYYY-MM-DD format
    days_back (int, optional): Number of days back from today (if start_date not provided)
    pollutants (str, optional): Comma-separated pollutant names (defaults to config.POLLUTANTS)
    
    Returns:
    DataFrame: Hourly air quality data or None if fetch failed
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
            
        # Use default pollutants from config if not specified
        if pollutants is None:
            pollutants = POLLUTANTS
        
        # Build API URL with parameters using Copenhagen timezone
        url = f"{AIR_QUALITY_API_URL}?latitude={latitude}&longitude={longitude}&hourly={pollutants}&start_date={start_date}&end_date={end_date}&timezone=Europe/Copenhagen"
        
        logger.info(f"Fetching air quality data from {start_date} to {end_date} for coordinates ({latitude}, {longitude})")
        
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        data = response.json()
        hourly_data = data.get("hourly", {})
        
        if not hourly_data or "time" not in hourly_data:
            logger.error(f"API response missing hourly data")
            return None
            
        df = pd.DataFrame(hourly_data)
        
        # Convert time column to datetime
        df["time"] = pd.to_datetime(df["time"])
        
        # Success statistics
        logger.info(f"Successfully fetched {len(df)} hours of air quality data")
        if len(df) > 0:
            logger.info(f"Date range: {df['time'].min()} to {df['time'].max()}")
        
        # Calculate AQI for the fetched data
        if len(df) > 0:
            df = calculate_aqi(df)
            logger.info(f"Added AQI calculations to air quality data")
            
        return df
        
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
        logger.error(f"Unexpected error: {e}")
        return None
