# main.py
import os
import sys
import logging
from datetime import datetime
import config
from data import traffic
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
    
    print("\nExecution complete")

if __name__ == "__main__":
    main()
