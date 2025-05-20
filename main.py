"""
AirSense Copenhagen - Data Science Project
"""

import os
import sys
import logging
from datetime import datetime
import config
from workflows.data_workflow import fetch_and_analyze_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                   handlers=[logging.FileHandler('airsense.log')])
logger = logging.getLogger(__name__)

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
