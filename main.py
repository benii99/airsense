"""
AirSense Copenhagen - Data Science Project
"""

import os
import sys
import logging
from datetime import datetime
import config
from data import traffic, air_quality, weather
from analysis import exploratory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler('airsense.log')])
logger = logging.getLogger(__name__)

def main():
    print("\n" + "="*80)
    print(" AirSense Copenhagen - Data Science Project")
    print("="*80)
    
    # Create directories inline
    os.makedirs(config.DEBUG_DIR, exist_ok=True)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    logger.info(f"Created output directories: {config.DEBUG_DIR}, {config.FIGURES_DIR}")
    
    # Load and analyze traffic data 
    exploratory.analyze_traffic_data(
        config.TRAFFIC_DATA_FILE,
        output_dir=os.path.join(config.FIGURES_DIR, "traffic_eda"),
        debug_dir=config.DEBUG_DIR
    )
    
    # Get first location from config
    first_location = next(iter(config.LOCATIONS))
    coordinates = config.LOCATIONS[first_location]
    
    # Define date range for traffic year
    start_date = f"{config.TRAFFIC_YEAR}-01-01"
    end_date = f"{config.TRAFFIC_YEAR}-12-31"
    
    # Fetch air quality data
    print(f"\nFetching air quality data for {first_location} ({config.TRAFFIC_YEAR})...")
    air_quality_data = air_quality.get_air_quality_data(
        latitude=coordinates[0],
        longitude=coordinates[1],
        start_date=start_date,
        end_date=end_date
    )
    
    if air_quality_data is not None:
        # Perform exploratory analysis on air quality data
        exploratory.analyze_air_quality_data(
            air_quality_data,
            output_dir=os.path.join(config.FIGURES_DIR, "air_quality_eda"),
            debug_dir=config.DEBUG_DIR
        )
    else:
        print(f"Failed to fetch air quality data for {first_location}")
    
    # Fetch weather data
    print(f"\nFetching weather data for {first_location} ({config.TRAFFIC_YEAR})...")
    weather_data = weather.get_weather_data(
        latitude=coordinates[0],
        longitude=coordinates[1],
        start_date=start_date,
        end_date=end_date
    )
    
    if weather_data is not None:
        # Save debug data
        filename = f"weather_data_{first_location.replace(' ', '_')}"
        saved_file = exploratory.save_data_to_csv(weather_data, filename, config.DEBUG_DIR)
        if saved_file:
            print(f"Saved weather data to {saved_file}")
            print(f"Records: {len(weather_data)}")
            print(f"Date range: {weather_data['time'].min()} to {weather_data['time'].max()}")
        
        # Analyze weather data if analysis function exists
        if hasattr(exploratory, 'analyze_weather_data'):
            exploratory.analyze_weather_data(
                weather_data,
                output_dir=os.path.join(config.FIGURES_DIR, "weather_eda"),
                debug_dir=config.DEBUG_DIR
            )
        else:
            print("Weather data analysis function not yet implemented")
    else:
        print(f"Failed to fetch weather data for {first_location}")
    
    print("\nExecution complete")

if __name__ == "__main__":
    main()
