# config.py

"""
Configuration for the AirSense Copenhagen project.
All configuration variables should be defined here to avoid hardcoding.
"""

# Locations with coordinates and descriptions
LOCATIONS = {
    "Torvegade": (55.6716, 12.5929),  # High Traffic Urban Center
    "Hvidovre Residential Area": (55.6406, 12.4846),  # Medium Density Mixed Use
    "Amager Strandpark": (55.6580, 12.6478)  # Coastal Recreational Area
}

# Data API configurations
AIR_QUALITY_API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_API_URL = "https://archive-api.open-meteo.com/v1/archive"

# Pollutants to retrieve from API
POLLUTANTS = "pm10,pm2_5,nitrogen_dioxide,carbon_monoxide,sulphur_dioxide,ozone"

# Weather variables to retrieve
WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "windspeed_10m",
    "pressure_msl",
    "winddirection_10m"
]

# Traffic data settings
TRAFFIC_DATA_FILE = "traffic_data_2014.csv"
TRAFFIC_YEAR = 2014

# Historical data retrieval settings
DEFAULT_HISTORY_DAYS = 365

# EPA data completeness requirements
MIN_HOURS_24H = 18  # Minimum hours for 24-hour average (75%)
MIN_HOURS_8H = 6    # Minimum hours for 8-hour average (75%)

# Conversion factors from μg/m³ to ppb or ppm
NO2_CONVERSION_FACTOR = 1.88  # 1 ppb = 1.88 μg/m³
SO2_CONVERSION_FACTOR = 2.62  # 1 ppb = 2.62 μg/m³
CO_CONVERSION_FACTOR = 1145   # 1 ppm = 1145 μg/m³
O3_CONVERSION_FACTOR = 1960   # 1 ppm = 2000 μg/m³

# Output directories
FIGURES_DIR = "figures"
DEBUG_DIR = "debug_data"
