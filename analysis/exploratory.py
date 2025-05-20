"""
Exploratory Data Analysis module for AirSense Copenhagen.
"""

import os
import pandas as pd
import logging
from datetime import datetime

# Import visualization utilities
from utils.visualization import (
    plot_time_series_with_acf_pacf,
    plot_average_by_hour,
    plot_hourly_boxplots,
    create_joint_pollutant_visualizations,
    create_joint_weather_visualizations
)

from data import traffic

logger = logging.getLogger(__name__)

def perform_timeseries_eda(df, value_col='traffic_count', time_col='datetime', output_dir=None):
    """
    Perform exploratory data analysis on time series data.
    
    Args:
        df: DataFrame with time series data
        value_col: Name of the column containing values to analyze
        time_col: Name of the time column
        output_dir: Directory to save output figures
        
    Returns:
        dict: Summary of EDA findings
    """
    if df is None or len(df) == 0:
        logger.error("No data provided for EDA")
        return None
    
    # Check for required columns
    required_cols = [time_col, value_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns for EDA: {missing_cols}")
        return None
    
    # Setup output directory
    if output_dir is None:
        output_dir = "figures/eda"
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Starting EDA analysis with {len(df)} records")
    
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d")
    
    # 1. Time Series Analysis
    logger.info("Creating time series plots with ACF/PACF")
    
    # Ensure data is sorted by time
    df_sorted = df.sort_values(time_col)
    
    # Create time series plot
    plot_time_series_with_acf_pacf(
        df_sorted,
        value_col=value_col,
        time_col=time_col,
        title=f'{value_col.capitalize()} Time Series with ACF/PACF',
        output_path=f"{output_dir}/{timestamp}_{value_col}_time_series_acf_pacf.png"
    )
    
    # 2. Average by Hour
    logger.info(f"Creating average {value_col} by hour plot")
    
    # Extract hour from datetime if not present
    if 'hour_of_day' not in df.columns:
        df_with_hour = df.copy()
        df_with_hour['hour_of_day'] = df_with_hour[time_col].dt.hour
    else:
        df_with_hour = df
    
    plot_average_by_hour(
        df_with_hour,
        value_col=value_col,
        output_path=f"{output_dir}/{timestamp}_{value_col}_hourly_average.png"
    )
    
    # 3. Hourly Box Plots
    logger.info(f"Creating hourly {value_col} box plots")
    plot_hourly_boxplots(
        df_with_hour,
        value_col=value_col,
        output_path=f"{output_dir}/{timestamp}_{value_col}_hourly_boxplots.png"
    )
    
    # 4. Calculate summary statistics
    logger.info("Calculating summary statistics")
    
    # Overall statistics
    overall_stats = df[value_col].describe()
    
    # Hourly statistics
    hourly_stats = df_with_hour.groupby('hour_of_day')[value_col].agg([
        'mean', 'std', 'min', 'max', 'count'
    ])
    
    # Peak hour
    peak_hour = hourly_stats['mean'].idxmax()
    peak_value = hourly_stats.loc[peak_hour, 'mean']
    
    # Low hour
    low_hour = hourly_stats['mean'].idxmin()
    low_value = hourly_stats.loc[low_hour, 'mean']
    
    # Day of week patterns
    day_stats = {}
    busiest_day = quietest_day = "Unknown"
    busiest_day_value = quietest_day_value = None
    
    if time_col in df.columns:
        df_with_day = df.copy()
        df_with_day['day_of_week'] = df_with_day[time_col].dt.dayofweek
        
        # Day names dictionary
        day_names = {
            0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
            4: 'Friday', 5: 'Saturday', 6: 'Sunday'
        }
        
        # Day of week statistics
        day_stats = df_with_day.groupby('day_of_week')[value_col].mean().reset_index()
        day_stats['day_name'] = day_stats['day_of_week'].map(day_names)
        
        # Find busiest and quietest days
        if len(day_stats) > 0:
            busiest_day_idx = day_stats[value_col].idxmax()
            busiest_day = day_stats.loc[busiest_day_idx, 'day_name']
            busiest_day_value = day_stats.loc[busiest_day_idx, value_col]
            
            quietest_day_idx = day_stats[value_col].idxmin()
            quietest_day = day_stats.loc[quietest_day_idx, 'day_name']
            quietest_day_value = day_stats.loc[quietest_day_idx, value_col]
    
    # Compile the EDA summary
    summary = {
        "data_points": len(df),
        "date_range": f"{df[time_col].min()} to {df[time_col].max()}",
        "statistics": {
            "mean": overall_stats['mean'],
            "std": overall_stats['std'],
            "min": overall_stats['min'],
            "max": overall_stats['max']
        },
        "hourly_patterns": {
            "peak_hour": int(peak_hour),
            "peak_value": peak_value,
            "low_hour": int(low_hour),
            "low_value": low_value
        },
        "daily_patterns": {
            "busiest_day": busiest_day,
            "busiest_day_value": busiest_day_value,
            "quietest_day": quietest_day,
            "quietest_day_value": quietest_day_value
        },
        "output_files": {
            "time_series": f"{output_dir}/{timestamp}_{value_col}_time_series_acf_pacf.png",
            "hourly_average": f"{output_dir}/{timestamp}_{value_col}_hourly_average.png",
            "hourly_boxplots": f"{output_dir}/{timestamp}_{value_col}_hourly_boxplots.png"
        }
    }
    
    # Log summary
    logger.info("Completed EDA analysis")
    logger.info(f"Peak {value_col} observed at hour {peak_hour} (average: {peak_value:.2f})")
    logger.info(f"Minimum {value_col} observed at hour {low_hour} (average: {low_value:.2f})")
    
    return summary

def save_data_to_csv(df, filename, directory):
    """Save DataFrame to CSV with timestamp in filename."""
    if df is None or len(df) == 0:
        logger.error("No data to save")
        return None
        
    os.makedirs(directory, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_filename = f"{filename}_{timestamp}.csv"
    filepath = os.path.join(directory, full_filename)
    
    df.to_csv(filepath, index=False)
    logger.info(f"Saved {len(df)} rows to {filepath}")
    
    return filepath

def analyze_traffic_data(traffic_file, output_dir=None, debug_dir=None):
    """
    Load traffic data and perform exploratory analysis.
    
    Args:
        traffic_file: Path to traffic data file
        output_dir: Directory to save figures
        debug_dir: Directory to save debug files
    
    Returns:
        tuple: (traffic_data, eda_results) or (None, None) if loading fails
    """
    print("\nLoading traffic data...")
    
    # Load traffic data
    traffic_data = traffic.get_traffic_data(traffic_file)
    
    if traffic_data is None or len(traffic_data) == 0:
        print("\nNo traffic data found")
        return None, None
    
    # Save to CSV
    if debug_dir:
        filename = "traffic_data"
        saved_file = save_data_to_csv(traffic_data, filename, debug_dir)
        
        if saved_file:
            print(f"\nSuccessfully loaded traffic data and saved to {saved_file}")
            print(f"Records: {len(traffic_data)}")
            print(f"Date range: {traffic_data['datetime'].min()} to {traffic_data['datetime'].max()}")
            
            # Display sample of the data
            print("\nSample of traffic data:")
            print(traffic_data.head().to_string())
    
    # Perform exploratory data analysis
    print("\nPerforming exploratory data analysis...")
    eda_results = perform_timeseries_eda(
        traffic_data, 
        value_col='traffic_count',
        time_col='datetime',
        output_dir=output_dir
    )
    
    if eda_results:
        print(f"\nEDA Summary:")
        print(f"- Data points analyzed: {eda_results['data_points']}")
        print(f"- Peak traffic hour: {eda_results['hourly_patterns']['peak_hour']:02d}:00 (avg: {eda_results['hourly_patterns']['peak_value']:.1f} vehicles)")
        print(f"- Lowest traffic hour: {eda_results['hourly_patterns']['low_hour']:02d}:00 (avg: {eda_results['hourly_patterns']['low_value']:.1f} vehicles)")
        print(f"- Busiest day: {eda_results['daily_patterns']['busiest_day']}")
        print(f"- Visualizations saved to {output_dir}/")
    else:
        print("\nEDA analysis failed")
    
    return traffic_data, eda_results

def analyze_air_quality_data(df, output_dir=None, debug_dir=None):
    """
    Perform exploratory analysis on air quality data with multiple pollutants.
    
    Args:
        df: DataFrame with air quality data
        output_dir: Directory to save figures
        debug_dir: Directory to save debug files
    
    Returns:
        dict: Summary of analyses for each pollutant
    """
    print("\nPerforming exploratory analysis on air quality data...")
    
    if df is None or len(df) == 0:
        print("No air quality data to analyze")
        return None
    
    # Setup output directory
    if output_dir is None:
        output_dir = "figures/air_quality_eda"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw data to CSV if debug_dir provided
    if debug_dir:
        filename = "air_quality_data"
        saved_file = save_data_to_csv(df, filename, debug_dir)
        if saved_file:
            print(f"Saved raw air quality data to {saved_file}")
            print(f"Records: {len(df)}")
            print(f"Date range: {df['time'].min()} to {df['time'].max()}")
            
            # Display sample of the data
            print("\nSample of air quality data:")
            print(df.head().to_string())
    
    # Identify pollutant columns and check which ones have valid data
    time_col = 'time'
    potential_pollutant_cols = [
        'pm10', 'pm2_5', 'nitrogen_dioxide', 'carbon_monoxide', 'sulphur_dioxide', 'ozone', 'AQI'
    ]
    
    # Filter to only include columns with sufficient valid data
    pollutant_cols = []
    for col in potential_pollutant_cols:
        if col in df.columns:
            valid_count = df[col].notna().sum()
            if valid_count > 100:  # Require at least 100 valid data points
                pollutant_cols.append(col)
                print(f"  {col}: {valid_count} valid values ({valid_count/len(df)*100:.1f}%)")
            else:
                print(f"  {col}: Insufficient valid data ({valid_count} values) - skipping")
    
    if not pollutant_cols:
        print("No pollutant columns with sufficient valid data found")
        return None
    
    print(f"Found {len(pollutant_cols)} pollutants with sufficient data: {', '.join(pollutant_cols)}")
    
    # Create joint visualizations
    print("\nCreating joint pollutant visualizations...")
    create_joint_pollutant_visualizations(
        df, 
        pollutant_cols, 
        time_col,
        output_dir
    )
    
    # Perform individual pollutant analysis
    pollutant_summaries = {}
    
    for pollutant in pollutant_cols:
        # Perform timeseries EDA for each pollutant
        print(f"\nAnalyzing {pollutant}...")
        
        # Create pollutant subdirectory
        pollutant_dir = os.path.join(output_dir, pollutant.replace("_", "-"))
        
        summary = perform_timeseries_eda(
            df,
            value_col=pollutant,
            time_col=time_col,
            output_dir=pollutant_dir
        )
        
        if summary:
            pollutant_summaries[pollutant] = summary
            
            # Print key stats
            print(f"  - Average: {summary['statistics']['mean']:.2f}")
            print(f"  - Peak hour: {summary['hourly_patterns']['peak_hour']:02d}:00 ({summary['hourly_patterns']['peak_value']:.2f})")
            print(f"  - Daily pattern: Highest on {summary['daily_patterns']['busiest_day']}")
    
    print(f"\nAir quality analysis complete. Visualizations saved to {output_dir}")
    
    # Return summaries
    return {
        'pollutant_summaries': pollutant_summaries,
        'output_dir': output_dir
    }

def analyze_weather_data(df, output_dir=None, debug_dir=None):
    """
    Perform exploratory analysis on weather data.
    
    Args:
        df: DataFrame with weather data
        output_dir: Directory to save figures
        debug_dir: Directory to save debug files
    
    Returns:
        dict: Summary of analyses for each weather variable
    """
    print("\nPerforming exploratory analysis on weather data...")
    
    if df is None or len(df) == 0:
        print("No weather data to analyze")
        return None
    
    # Setup output directory
    if output_dir is None:
        output_dir = "figures/weather_eda"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw data to CSV if debug_dir provided
    if debug_dir:
        filename = "weather_data"
        saved_file = save_data_to_csv(df, filename, debug_dir)
        if saved_file:
            print(f"Saved raw weather data to {saved_file}")
            print(f"Records: {len(df)}")
            print(f"Date range: {df['time'].min()} to {df['time'].max()}")
            
            # Display sample of the data
            print("\nSample of weather data:")
            print(df.head().to_string())
    
    # Identify weather variable columns
    time_col = 'time'
    potential_weather_cols = [
        'temperature_2m', 'relative_humidity_2m', 'precipitation',
        'windspeed_10m', 'pressure_msl', 'winddirection_10m'
    ]
    
    weather_cols = [col for col in potential_weather_cols if col in df.columns]
    
    if not weather_cols:
        print("No weather variable columns found in data")
        return None
    
    print(f"Found {len(weather_cols)} weather variables: {', '.join(weather_cols)}")
    
    # Create joint visualizations
    print("\nCreating joint weather visualizations...")
    create_joint_weather_visualizations(
        df, 
        weather_cols, 
        time_col,
        output_dir
    )
    
    # Perform individual weather variable analysis
    weather_summaries = {}
    
    for variable in weather_cols:
        # Perform timeseries EDA for each weather variable
        print(f"\nAnalyzing {variable}...")
        
        # Create weather variable subdirectory
        variable_dir = os.path.join(output_dir, variable.replace("_", "-"))
        
        summary = perform_timeseries_eda(
            df,
            value_col=variable,
            time_col=time_col,
            output_dir=variable_dir
        )
        
        if summary:
            weather_summaries[variable] = summary
            
            # Print key stats
            print(f"  - Average: {summary['statistics']['mean']:.2f}")
            print(f"  - Peak hour: {summary['hourly_patterns']['peak_hour']:02d}:00 ({summary['hourly_patterns']['peak_value']:.2f})")
            print(f"  - Daily pattern: Highest on {summary['daily_patterns']['busiest_day']}")
    
    print(f"\nWeather analysis complete. Visualizations saved to {output_dir}")
    
    # Return summaries
    return {
        'weather_summaries': weather_summaries,
        'output_dir': output_dir
    }
