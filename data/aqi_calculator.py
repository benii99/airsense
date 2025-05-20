"""
Air Quality Index (AQI) calculator module for AirSense Copenhagen.
Based on EPA standards for calculating AQI from pollutant concentrations.
"""

import numpy as np
import pandas as pd
import logging
from config import NO2_CONVERSION_FACTOR, SO2_CONVERSION_FACTOR, CO_CONVERSION_FACTOR, O3_CONVERSION_FACTOR
from config import MIN_HOURS_24H, MIN_HOURS_8H
from utils.aqi_breakpoints import (PM25_BREAKPOINTS, PM10_BREAKPOINTS, NO2_BREAKPOINTS, 
                           CO_BREAKPOINTS, SO2_BREAKPOINTS, O3_BREAKPOINTS, O3_1H_BREAKPOINTS)

logger = logging.getLogger(__name__)

def calc_aqi(Cp, BPl, BPh, Il, Ih):
    """
    Calculate AQI using EPA's linear interpolation formula.
    
    Parameters:
    Cp: Pollutant concentration
    BPl, BPh: Concentration breakpoints
    Il, Ih: Index breakpoints
    
    Returns:
    AQI value
    """
    a = (Ih - Il)
    b = (BPh - BPl)
    c = (Cp - BPl)
    return round((a / b) * c + Il)

def find_breakpoint(value, breakpoints):
    """Find the appropriate breakpoint interval for a given concentration value."""
    for (BPl, BPh, Il, Ih) in breakpoints:
        if BPl <= value <= BPh:
            return (BPl, BPh, Il, Ih)
    
    # If value exceeds the highest breakpoint, use the highest breakpoint
    if value > breakpoints[-1][1]:
        return breakpoints[-1]
    
    return None

def pollutant_aqi(pollutant, value, is_1h_ozone=False):
    """Calculate AQI for a specific pollutant given its concentration value."""
    # Handle missing, null or negative values
    if value is None or np.isnan(value) or value < 0:
        return None
        
    if pollutant == 'pm2_5':
        bp = find_breakpoint(value, PM25_BREAKPOINTS)
        if bp:
            return calc_aqi(value, *bp)
    elif pollutant == 'pm10':
        bp = find_breakpoint(value, PM10_BREAKPOINTS)
        if bp:
            return calc_aqi(value, *bp)
    elif pollutant == 'nitrogen_dioxide':
        # Convert μg/m³ to ppb
        ppb = value / NO2_CONVERSION_FACTOR
        bp = find_breakpoint(ppb, NO2_BREAKPOINTS)
        if bp:
            return calc_aqi(ppb, *bp)
    elif pollutant == 'carbon_monoxide':
        # Convert μg/m³ to ppm
        ppm = value / CO_CONVERSION_FACTOR
        bp = find_breakpoint(ppm, CO_BREAKPOINTS)
        if bp:
            return calc_aqi(ppm, *bp)
    elif pollutant == 'sulphur_dioxide':
        # Convert μg/m³ to ppb
        ppb = value / SO2_CONVERSION_FACTOR
        bp = find_breakpoint(ppb, SO2_BREAKPOINTS)
        if bp:
            return calc_aqi(ppb, *bp)
    elif pollutant == 'ozone':
        # Convert μg/m³ to ppm
        ppm = value / O3_CONVERSION_FACTOR
        if is_1h_ozone:
            bp = find_breakpoint(ppm, O3_1H_BREAKPOINTS)
        else:
            bp = find_breakpoint(ppm, O3_BREAKPOINTS)
        if bp:
            return calc_aqi(ppm, *bp)
    return None

def get_aqi_category(aqi):
    """Convert numeric AQI value to EPA category."""
    if aqi is None:
        return None
    elif aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def calculate_rolling_averages(df):
    """
    Calculate rolling averages as per EPA standards.
    
    Parameters:
    df: DataFrame with hourly pollutant data
    
    Returns:
    DataFrame with added columns for averaged values
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Sort by time ascending for proper rolling calculations
    result_df = result_df.sort_values(by="time", ascending=True)
    
    # List of potential pollutants
    pollutants = ['pm2_5', 'pm10', 'nitrogen_dioxide', 'carbon_monoxide', 'sulphur_dioxide', 'ozone']
    available_pollutants = [p for p in pollutants if p in result_df.columns]
    
    if not available_pollutants:
        logger.warning("No pollutant columns found for calculating rolling averages")
        return result_df
    
    logger.info(f"Calculating rolling averages for {len(available_pollutants)} pollutants")
    
    # 24-hour averages for PM2.5 and PM10
    if 'pm2_5' in available_pollutants:
        result_df['pm2_5_24h'] = result_df['pm2_5'].rolling(window=24, min_periods=MIN_HOURS_24H).mean()
        logger.info(f"Calculated 24-hour rolling average for PM2.5 (min periods: {MIN_HOURS_24H})")
        
    if 'pm10' in available_pollutants:
        result_df['pm10_24h'] = result_df['pm10'].rolling(window=24, min_periods=MIN_HOURS_24H).mean()
        logger.info(f"Calculated 24-hour rolling average for PM10 (min periods: {MIN_HOURS_24H})")
    
    # 8-hour averages for O3 and CO
    if 'ozone' in available_pollutants:
        result_df['ozone_8h'] = result_df['ozone'].rolling(window=8, min_periods=MIN_HOURS_8H).mean()
        result_df['ozone_1h'] = result_df['ozone']  # Keep 1-hour values for high concentrations
        logger.info(f"Calculated 8-hour rolling average for ozone (min periods: {MIN_HOURS_8H})")
        
    if 'carbon_monoxide' in available_pollutants:
        result_df['carbon_monoxide_8h'] = result_df['carbon_monoxide'].rolling(window=8, min_periods=MIN_HOURS_8H).mean()
        logger.info(f"Calculated 8-hour rolling average for carbon monoxide (min periods: {MIN_HOURS_8H})")
    
    # 1-hour values for NO2 and SO2 (just rename for consistency)
    if 'nitrogen_dioxide' in available_pollutants:
        result_df['nitrogen_dioxide_1h'] = result_df['nitrogen_dioxide']
        
    if 'sulphur_dioxide' in available_pollutants:
        result_df['sulphur_dioxide_1h'] = result_df['sulphur_dioxide']
    
    return result_df

def calculate_aqi(df):
    """
    Calculate AQI for each row in a dataframe containing pollutant data.
    Uses appropriate rolling averages as per EPA standards.
    
    Parameters:
    df: DataFrame with columns for each pollutant
    
    Returns:
    DataFrame with additional AQI column
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Calculate rolling averages first
    result_df = calculate_rolling_averages(result_df)
    
    # Prepare for AQI calculation
    aqi_values = []
    dominant_pollutants = []
    
    # For each row in the dataframe
    for idx, row in result_df.iterrows():
        pollutant_aqis = {}
        
        # Use the appropriate averaged values for AQI calculation
        if 'pm2_5_24h' in result_df.columns and pd.notna(row['pm2_5_24h']):
            aqi = pollutant_aqi('pm2_5', row['pm2_5_24h'])
            if aqi is not None:
                pollutant_aqis['PM2.5'] = aqi
                
        if 'pm10_24h' in result_df.columns and pd.notna(row['pm10_24h']):
            aqi = pollutant_aqi('pm10', row['pm10_24h'])
            if aqi is not None:
                pollutant_aqis['PM10'] = aqi
                
        if 'nitrogen_dioxide_1h' in result_df.columns and pd.notna(row['nitrogen_dioxide_1h']):
            aqi = pollutant_aqi('nitrogen_dioxide', row['nitrogen_dioxide_1h'])
            if aqi is not None:
                pollutant_aqis['NO2'] = aqi
                
        if 'carbon_monoxide_8h' in result_df.columns and pd.notna(row['carbon_monoxide_8h']):
            aqi = pollutant_aqi('carbon_monoxide', row['carbon_monoxide_8h'])
            if aqi is not None:
                pollutant_aqis['CO'] = aqi
                
        if 'sulphur_dioxide_1h' in result_df.columns and pd.notna(row['sulphur_dioxide_1h']):
            aqi = pollutant_aqi('sulphur_dioxide', row['sulphur_dioxide_1h'])
            if aqi is not None:
                pollutant_aqis['SO2'] = aqi
        
        # Calculate both 1-hour and 8-hour ozone AQI and use the higher value
        o3_aqi_8h = None
        o3_aqi_1h = None
        
        if 'ozone_8h' in result_df.columns and pd.notna(row['ozone_8h']):
            o3_aqi_8h = pollutant_aqi('ozone', row['ozone_8h'])
            
        if 'ozone_1h' in result_df.columns and pd.notna(row['ozone_1h']):
            o3_aqi_1h = pollutant_aqi('ozone', row['ozone_1h'], is_1h_ozone=True)
            
        # Use the higher of the two ozone AQI values
        if o3_aqi_8h is not None and o3_aqi_1h is not None:
            pollutant_aqis['O3'] = max(o3_aqi_8h, o3_aqi_1h)
        elif o3_aqi_8h is not None:
            pollutant_aqis['O3'] = o3_aqi_8h
        elif o3_aqi_1h is not None:
            pollutant_aqis['O3'] = o3_aqi_1h
        
        if pollutant_aqis:
            # AQI is the maximum of the individual pollutant AQIs
            max_aqi = max(pollutant_aqis.values())
            # Dominant pollutant is the one with the highest AQI
            dominant_pollutant = [p for p, v in pollutant_aqis.items() if v == max_aqi][0]
            aqi_values.append(max_aqi)
            dominant_pollutants.append(dominant_pollutant)
        else:
            aqi_values.append(None)
            dominant_pollutants.append(None)
    
    # Add results to the dataframe
    result_df['AQI'] = aqi_values
    result_df['Dominant_Pollutant'] = dominant_pollutants
    result_df['AQI_Category'] = result_df['AQI'].apply(get_aqi_category)
    
    valid_aqi_count = result_df['AQI'].notna().sum()
    logger.info(f"Successfully calculated AQI for {valid_aqi_count}/{len(result_df)} records")
    
    return result_df
