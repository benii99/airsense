"""
Data workflow module for AirSense Copenhagen.
Handles high-level data processing flows across multiple data sources.
"""

import os
import logging
import pandas as pd
from datetime import datetime
import config
from data import traffic, air_quality, weather, data_merger
from analysis import exploratory, transformation, correlation

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
        
        # Perform time series transformation analysis
        print("\nPerforming time series transformation analysis...")
        
        # Define relevant metrics for analysis
        metrics = []
        
        # Add pollutant metrics
        pollutant_metrics = ['pm10', 'pm2_5', 'nitrogen_dioxide', 'carbon_monoxide', 
                            'sulphur_dioxide', 'ozone', 'AQI']
        metrics.extend([m for m in pollutant_metrics if m in merged_data.columns])
        
        # Add weather metrics
        weather_metrics = ['temperature_2m', 'relative_humidity_2m', 'precipitation', 
                          'windspeed_10m', 'pressure_msl', 'winddirection_10m']
        metrics.extend([m for m in weather_metrics if m in merged_data.columns])
        
        # Add traffic metrics
        if 'traffic_count' in merged_data.columns:
            metrics.append('traffic_count')
            
        # Create transformation analysis directory
        transform_dir = os.path.join(config.FIGURES_DIR, "transformations")
        os.makedirs(transform_dir, exist_ok=True)
        
        # Analyze metrics
        print(f"Analyzing distributions for {len(metrics)} metrics...")
        analysis_results = transformation.analyze_metrics(
            merged_data,
            metrics=metrics,
            output_dir=transform_dir
        )
        
        # Create transformed dataset
        print("Creating transformed dataset...")
        transformed_data = transformation.create_transformed_dataset(
            merged_data, 
            analysis_results=analysis_results,
            output_dir=config.DEBUG_DIR
        )
        
        # Save transformed data
        filename = f"transformed_data_{location_name.replace(' ', '_')}_{year}"
        saved_file = exploratory.save_data_to_csv(transformed_data, filename, config.DEBUG_DIR)
        if saved_file:
            print(f"Saved transformed data to {saved_file}")
            print(f"Records: {len(transformed_data)}")
            print(f"Added {len(transformed_data.columns) - len(merged_data.columns)} transformed columns")
        
        # Add transformed data to datasets
        datasets['transformed'] = transformed_data
        
        # Save analysis results
        analysis_file = os.path.join(config.DEBUG_DIR, "transformation_analysis.csv")
        analysis_results.to_csv(analysis_file, index=False)
        print(f"Saved transformation analysis to {analysis_file}")
        
        # Perform correlation analysis
        print("\nPerforming correlation analysis on original and transformed data...")
        
        # Create correlation analysis directory
        correlation_dir = os.path.join(config.FIGURES_DIR, "correlations")
        os.makedirs(correlation_dir, exist_ok=True)
        
        # Run correlation analysis
        corr_results = correlation.analyze_correlations(
            transformed_data,
            metrics=metrics,
            output_dir=correlation_dir
        )
        
        if corr_results:
            # Add correlation results to datasets
            datasets['correlation'] = corr_results
            print(f"Correlation analysis complete. Results saved to {correlation_dir}")
            
            # Save correlation metrics summary if available
            if 'comparison' in corr_results and corr_results['comparison'] is not None:
                comp_df = corr_results['comparison']
                comp_file = os.path.join(config.DEBUG_DIR, "correlation_improvements.csv")
                comp_df.to_csv(comp_file, index=False)
                print(f"Saved correlation improvement metrics to {comp_file}")
        else:
            print("Correlation analysis failed or produced no results")
            
    else:
        print("Failed to create merged dataset")
        datasets['merged'] = None
    
    return datasets
