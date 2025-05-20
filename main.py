"""
AirSense Copenhagen - Data Science Project
"""

import os
import sys
import logging
from datetime import datetime
import config
from data import traffic, air_quality, weather, data_merger
from analysis import exploratory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler('airsense.log')])
logger = logging.getLogger(__name__)

def fetch_and_analyze_data(location_name, coordinates, year):
    """
    Fetch and analyze data for a specific location and year.
    
    Args:
        location_name: Name of the location
        coordinates: (latitude, longitude) tuple
        year: Year to analyze
    
    Returns:
        dict: Dictionary of datasets
    """
    datasets = {}
    
    # Define date range
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    
    # Load and analyze traffic data
    print(f"\nLoading traffic data for {location_name} ({year})...")
    traffic_data = exploratory.analyze_traffic_data(
        config.TRAFFIC_DATA_FILE,
        output_dir=os.path.join(config.FIGURES_DIR, "traffic_eda"),
        debug_dir=config.DEBUG_DIR
    )
    datasets['traffic'] = traffic_data
    
    # Fetch air quality data
    print(f"\nFetching air quality data for {location_name} ({year})...")
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
        datasets['air_quality'] = air_quality_data
    else:
        print(f"Failed to fetch air quality data for {location_name}")
        datasets['air_quality'] = None
    
    # Fetch weather data
    print(f"\nFetching weather data for {location_name} ({year})...")
    weather_data = weather.get_weather_data(
        latitude=coordinates[0],
        longitude=coordinates[1],
        start_date=start_date,
        end_date=end_date
    )
    
    if weather_data is not None:
        # Save debug data
        filename = f"weather_data_{location_name.replace(' ', '_')}"
        saved_file = exploratory.save_data_to_csv(weather_data, filename, config.DEBUG_DIR)
        if saved_file:
            print(f"Saved weather data to {saved_file}")
            print(f"Records: {len(weather_data)}")
            print(f"Date range: {weather_data['time'].min()} to {weather_data['time'].max()}")
        
        # Analyze weather data
        exploratory.analyze_weather_data(
            weather_data,
            output_dir=os.path.join(config.FIGURES_DIR, "weather_eda"),
            debug_dir=config.DEBUG_DIR
        )
        datasets['weather'] = weather_data
    else:
        print(f"Failed to fetch weather data for {location_name}")
        datasets['weather'] = None
    
    # Merge all datasets
    print("\nMerging traffic, air quality, and weather data...")
    merged_data = data_merger.merge_datasets(
        traffic_df=datasets.get('traffic'),
        air_quality_df=datasets.get('air_quality'),
        weather_df=datasets.get('weather')
    )
    
    if merged_data is not None and len(merged_data) > 0:
        # Save merged data
        filename = f"merged_data_{location_name.replace(' ', '_')}_{year}"
        saved_file = exploratory.save_data_to_csv(merged_data, filename, config.DEBUG_DIR)
        if saved_file:
            print(f"Saved merged data to {saved_file}")
            print(f"Records: {len(merged_data)}")
            print(f"Date range: {merged_data['time'].min()} to {merged_data['time'].max()}")
            
            # Get column counts by data type
            column_counts = merged_data.dtypes.value_counts()
            print("\nMerged data summary:")
            print(f"Total columns: {len(merged_data.columns)}")
            print(f"Column data types: {column_counts.to_dict()}")
            
            # Check for complete rows
            complete_rows = merged_data.dropna().shape[0]
            print(f"Complete rows: {complete_rows} ({complete_rows/len(merged_data)*100:.1f}%)")
        
        datasets['merged'] = merged_data
    else:
        print("Failed to create merged dataset")
        datasets['merged'] = None
    
    return datasets

def main():
    print("\n" + "="*80)
    print(" AirSense Copenhagen - Data Science Project")
    print("="*80)
    
    # Create directories inline
    os.makedirs(config.DEBUG_DIR, exist_ok=True)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    logger.info(f"Created output directories: {config.DEBUG_DIR}, {config.FIGURES_DIR}")
    
    # Get first location from config
    first_location = next(iter(config.LOCATIONS))
    coordinates = config.LOCATIONS[first_location]
    
    # Fetch and analyze all data
    datasets = fetch_and_analyze_data(first_location, coordinates, config.TRAFFIC_YEAR)
    
    print("\nExecution complete")

if __name__ == "__main__":
    main()
