"""
Time series transformation module for AirSense Copenhagen.
Handles data evaluation and transformation for time series analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose, STL
import os
import logging
import warnings

# Suppress specific statsmodels warnings
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

logger = logging.getLogger(__name__)

def evaluate_distribution(series, name, output_dir=None):
    """
    Evaluate the distribution of a time series and suggest transformations.
    
    Args:
        series: Pandas Series to evaluate
        name: Name of the variable
        output_dir: Directory to save plots
        
    Returns:
        dict: Dictionary with distribution metrics and suggested transformations
    """
    # Skip if all values are NaN
    if series.isna().all():
        logger.warning(f"Series '{name}' contains only NaN values. Skipping evaluation.")
        return {
            "name": name,
            "valid_values": 0,
            "skewness": None,
            "kurtosis": None,
            "zero_percentage": None,
            "distribution_shape": "invalid",
            "suggested_transform": None
        }
    
    # Basic statistics
    valid_series = series.dropna()
    n_valid = len(valid_series)
    if n_valid == 0:
        return {
            "name": name,
            "valid_values": 0,
            "skewness": None,
            "kurtosis": None,
            "zero_percentage": None,
            "distribution_shape": "invalid",
            "suggested_transform": None
        }
    
    # Calculate distribution metrics
    skewness = stats.skew(valid_series)
    kurtosis = stats.kurtosis(valid_series)
    zero_percentage = (valid_series == 0).mean() * 100
    
    # Determine distribution shape
    if abs(skewness) < 0.5:
        distribution = "approximately_symmetric"
    elif skewness > 0.5:
        distribution = "right_skewed"
    else:
        distribution = "left_skewed"
    
    # Check for bimodality
    if kurtosis < -1.0:
        distribution = "possible_bimodal"
    
    # Check for zero inflation
    if zero_percentage > 50:
        distribution = "zero_inflated"
    
    # Suggest transformations
    if distribution == "right_skewed" and zero_percentage < 10:
        suggested_transform = "log"
    elif distribution == "right_skewed" and zero_percentage >= 10:
        suggested_transform = "boxcox"
    elif distribution == "left_skewed":
        suggested_transform = "square"
    elif distribution == "zero_inflated":
        suggested_transform = "two_part_model"
    elif distribution == "possible_bimodal":
        suggested_transform = "seasonal_decomposition"
    else:
        suggested_transform = "none"
    
    # Special case for directional data
    if name == "winddirection_10m":
        suggested_transform = "sine_cosine"
    
    # Create output dictionary
    result = {
        "name": name,
        "valid_values": n_valid,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "zero_percentage": zero_percentage,
        "distribution_shape": distribution,
        "suggested_transform": suggested_transform
    }
    
    # Create visualization if output_dir specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        
        # Original distribution
        plt.subplot(1, 2, 1)
        sns.histplot(valid_series, kde=True)
        plt.title(f"Original Distribution of {name}\nSkew: {skewness:.3f}, Kurt: {kurtosis:.3f}")
        
        # Transformed distribution (if applicable)
        plt.subplot(1, 2, 2)
        if suggested_transform == "log":
            # Add small constant to handle zeros
            transformed = np.log1p(valid_series)
            sns.histplot(transformed, kde=True)
            plt.title(f"Log Transformed\nSkew: {stats.skew(transformed):.3f}")
        elif suggested_transform == "boxcox":
            if (valid_series > 0).all():
                transformed, lambda_val = stats.boxcox(valid_series)
                sns.histplot(transformed, kde=True)
                plt.title(f"Box-Cox Transformed (λ={lambda_val:.3f})\nSkew: {stats.skew(transformed):.3f}")
            else:
                transformed = np.log1p(valid_series)
                sns.histplot(transformed, kde=True)
                plt.title(f"Log Transformed (Box-Cox not applicable)\nSkew: {stats.skew(transformed):.3f}")
        elif suggested_transform == "square":
            transformed = np.square(valid_series)
            sns.histplot(transformed, kde=True)
            plt.title(f"Squared\nSkew: {stats.skew(transformed):.3f}")
        elif suggested_transform == "sine_cosine":
            # Convert to radians and take sine
            radians = valid_series * np.pi / 180
            sine_vals = np.sin(radians)
            sns.histplot(sine_vals, kde=True)
            plt.title(f"Sine Transform\nSkew: {stats.skew(sine_vals):.3f}")
        else:
            plt.text(0.5, 0.5, f"No transformation suggested", 
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.title("No Transformation")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name}_distribution.png"))
        plt.close()
    
    return result

def transform_series(series, method="log", offset=1e-8):
    """
    Apply transformation to a series.
    
    Args:
        series: Pandas Series to transform
        method: Transformation method ('log', 'boxcox', 'square', 'sine_cosine', 'two_part_model')
        offset: Small value to add before log transform to handle zeros
    
    Returns:
        Transformed series, or tuple of series for two-part models
    """
    if series.isna().all():
        return series
        
    if method == "log":
        return np.log1p(series + offset)
        
    elif method == "boxcox":
        # Handle non-positive values if present
        if (series > 0).all():
            transformed, _ = stats.boxcox(series)
            return pd.Series(transformed, index=series.index)
        else:
            # Fallback to log transform if non-positive values
            return np.log1p(series + offset)
            
    elif method == "square":
        return np.square(series)
        
    elif method == "sine_cosine":
        # This returns two series
        radians = series * np.pi / 180
        return pd.Series(np.sin(radians), index=series.index), pd.Series(np.cos(radians), index=series.index)
        
    elif method == "two_part_model":
        # Return binary indicator and continuous part
        is_nonzero = (series > 0).astype(float)
        continuous_part = series.copy()
        continuous_part[continuous_part == 0] = np.nan  # Keep only non-zero values
        return is_nonzero, continuous_part
        
    else:
        # No transformation
        return series

def difference_series(series, lag=1, order=1, seasonal_lag=None):
    """
    Apply differencing transformation to a series.
    
    Args:
        series: Pandas Series to transform
        lag: Lag for first differencing
        order: Number of times to difference
        seasonal_lag: Seasonal lag (e.g., 24 for hourly data with daily seasonality)
        
    Returns:
        Differenced series
    """
    # Make a copy to avoid modifying the original
    result = series.copy()
    
    # First apply seasonal differencing if specified
    if seasonal_lag is not None and seasonal_lag > 0:
        result = result.diff(seasonal_lag)
        # Drop NaN values from seasonal differencing
        result = result.dropna()
    
    # Then apply regular differencing of specified order
    for _ in range(order):
        result = result.diff(lag)
        # Drop NaN values from differencing
        result = result.dropna()
    
    return result

def test_stationarity(series, name=None, alpha=0.05):
    """
    Test for stationarity using Augmented Dickey-Fuller and KPSS tests.
    
    Args:
        series: Time series to test
        name: Name of the series (for reporting)
        alpha: Significance level
    
    Returns:
        dict: Dictionary with stationarity test results
    """
    # Drop NAs for the tests
    series = series.dropna()
    
    # Skip if too few observations
    if len(series) < 20:
        return {
            "name": name or "series",
            "is_stationary": False,
            "adf_pvalue": None,
            "kpss_pvalue": None,
            "suggested_differencing": None
        }
    
    try:
        # Augmented Dickey-Fuller test
        adf_result = adfuller(series)
        adf_pvalue = adf_result[1]
        
        # KPSS test
        kpss_result = kpss(series, regression='c', nlags="auto")
        kpss_pvalue = kpss_result[1]
        
        # Determine stationarity
        # ADF: H0 is non-stationary, so small p-value = stationary
        # KPSS: H0 is stationary, so small p-value = non-stationary
        adf_stationary = adf_pvalue < alpha
        kpss_stationary = kpss_pvalue > alpha
        
        # Combined decision
        is_stationary = adf_stationary and kpss_stationary
        
        # Suggest differencing if non-stationary
        if not is_stationary:
            if kpss_stationary and not adf_stationary:
                suggested_differencing = "first_order"
            else:
                suggested_differencing = "seasonal_and_first_order"
        else:
            suggested_differencing = "none"
            
        return {
            "name": name or "series",
            "is_stationary": is_stationary,
            "adf_pvalue": adf_pvalue,
            "kpss_pvalue": kpss_pvalue,
            "suggested_differencing": suggested_differencing
        }
    
    except Exception as e:
        logger.error(f"Error in stationarity test for {name}: {e}")
        return {
            "name": name or "series",
            "is_stationary": None,
            "adf_pvalue": None,
            "kpss_pvalue": None,
            "suggested_differencing": None,
            "error": str(e)
        }

def decompose_time_series(series, period=24, method='STL', seasonal=7):
    """
    Decompose time series into trend, seasonal, and residual components.
    
    Args:
        series: Time series to decompose
        period: Period for decomposition (e.g., 24 for hourly data with daily seasonality)
        method: Decomposition method ('STL' or 'classical')
        seasonal: Seasonal smoothing parameter for STL
        
    Returns:
        Dictionary containing decomposed components
    """
    try:
        # Ensure series is a pandas Series with datetime index
        if not isinstance(series, pd.Series):
            series = pd.Series(series)
        
        # Check for sufficient data
        if len(series) < period * 2:
            logger.warning(f"Series too short for decomposition with period {period}")
            return None
        
        # Remove NaNs
        series = series.dropna()
        
        if method == 'STL':
            # STL decomposition
            stl = STL(series, period=period, seasonal=seasonal)
            result = stl.fit()
            decomposition = {
                'trend': result.trend,
                'seasonal': result.seasonal,
                'residual': result.resid,
                'observed': series,
                'seasonally_adjusted': result.trend + result.resid
            }
        else:
            # Classical decomposition
            result = seasonal_decompose(series, model='additive', period=period)
            decomposition = {
                'trend': result.trend,
                'seasonal': result.seasonal,
                'residual': result.resid,
                'observed': result.observed,
                'seasonally_adjusted': result.trend + result.resid
            }
        
        return decomposition
        
    except Exception as e:
        logger.error(f"Error in time series decomposition: {e}")
        return None

def plot_time_series_decomposition(decomposition, title, output_path):
    """
    Plot time series decomposition components.
    
    Args:
        decomposition: Dictionary with decomposition components
        title: Title for the plot
        output_path: Path to save the plot
    """
    if decomposition is None:
        logger.warning("Cannot plot decomposition: No decomposition provided")
        return
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create plot
        fig, axes = plt.subplots(5, 1, figsize=(12, 15))
        
        # Original series
        axes[0].plot(decomposition['observed'])
        axes[0].set_title('Original Series')
        axes[0].tick_params(labelbottom=False)
        
        # Trend component
        axes[1].plot(decomposition['trend'])
        axes[1].set_title('Trend')
        axes[1].tick_params(labelbottom=False)
        
        # Seasonal component
        axes[2].plot(decomposition['seasonal'])
        axes[2].set_title('Seasonality')
        axes[2].tick_params(labelbottom=False)
        
        # Residual component
        axes[3].plot(decomposition['residual'])
        axes[3].set_title('Residuals')
        axes[3].tick_params(labelbottom=False)
        
        # Seasonally adjusted series
        axes[4].plot(decomposition['seasonally_adjusted'])
        axes[4].set_title('Seasonally Adjusted Series')
        
        # Format and save
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved time series decomposition plot to {output_path}")
        
    except Exception as e:
        logger.error(f"Error plotting time series decomposition: {e}")

def plot_differencing_comparison(series, series_name, lag=1, seasonal_lag=None, output_path=None):
    """
    Plot comparison of original series vs differenced series.
    
    Args:
        series: Original time series
        series_name: Name of the series
        lag: Lag for first differencing
        seasonal_lag: Lag for seasonal differencing
        output_path: Path to save the plot
    """
    try:
        # Create directory if it doesn't exist
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create differenced series
        first_diff = difference_series(series, lag=lag)
        
        seasonal_diff = None
        seasonal_diff_first = None
        
        if seasonal_lag is not None:
            seasonal_diff = difference_series(series, lag=seasonal_lag)
            seasonal_diff_first = difference_series(series, lag=lag, seasonal_lag=seasonal_lag)
        
        # Create plot
        if seasonal_lag is not None:
            fig, axes = plt.subplots(4, 1, figsize=(12, 16))
        else:
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Original series
        axes[0].plot(series)
        axes[0].set_title(f'Original Series: {series_name}')
        
        # First differenced
        axes[1].plot(first_diff)
        axes[1].set_title(f'First Differenced (lag={lag})')
        
        if seasonal_lag is not None:
            # Seasonal differenced
            axes[2].plot(seasonal_diff)
            axes[2].set_title(f'Seasonal Differenced (lag={seasonal_lag})')
            
            # Seasonal + First differenced
            axes[3].plot(seasonal_diff_first)
            axes[3].set_title(f'Seasonal + First Differenced')
        
        plt.tight_layout()
        
        # Save or display
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved differencing comparison plot to {output_path}")
        else:
            plt.show()
            
    except Exception as e:
        logger.error(f"Error plotting differencing comparison: {e}")

def analyze_metrics(df, metrics=None, output_dir="figures/transformations"):
    """
    Analyze multiple metrics in a dataframe and suggest transformations.
    
    Args:
        df: DataFrame with time series data
        metrics: List of column names to analyze (default: analyze all numeric columns)
        output_dir: Directory to save analysis results
    
    Returns:
        DataFrame: Summary of analysis results
    """
    if metrics is None:
        # Use all numeric columns except time/date columns
        exclude_patterns = ['time', 'date', 'year', 'month', 'day', 'hour', 'minute', 'second']
        metrics = [col for col in df.select_dtypes(include=np.number).columns 
                  if not any(pattern in col.lower() for pattern in exclude_patterns)]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    decomp_dir = os.path.join(output_dir, "decompositions")
    diff_dir = os.path.join(output_dir, "differencing")
    dist_dir = os.path.join(output_dir, "distributions")
    os.makedirs(decomp_dir, exist_ok=True)
    os.makedirs(diff_dir, exist_ok=True)
    os.makedirs(dist_dir, exist_ok=True)
    
    # Time-based index - needed for decomposition and differencing
    if 'time' in df.columns:
        df_indexed = df.set_index('time')
    else:
        df_indexed = df.copy()
    
    # Analyze each metric
    results = []
    for metric in metrics:
        logger.info(f"Analyzing {metric}")
        
        # Skip if column doesn't exist
        if metric not in df.columns:
            logger.warning(f"Column {metric} not found in DataFrame")
            continue
        
        # Evaluate distribution
        dist_result = evaluate_distribution(df[metric], metric, os.path.join(dist_dir))
        
        # Test stationarity
        stat_result = test_stationarity(df[metric], metric)
        
        # Add seasonal period detection (daily=24 for hourly data)
        seasonal_period = 24 if 'time' in df.columns else None
        
        # Perform time series decomposition
        if metric in df_indexed.columns and len(df_indexed[metric].dropna()) >= seasonal_period*2:
            try:
                # Decompose the time series
                decomposition = decompose_time_series(
                    df_indexed[metric], 
                    period=seasonal_period,
                    method='STL'
                )
                
                # Plot decomposition
                if decomposition:
                    plot_time_series_decomposition(
                        decomposition,
                        f"Time Series Decomposition of {metric}",
                        os.path.join(decomp_dir, f"{metric}_decomposition.png")
                    )
                    
                    # Update results with seasonal information
                    dist_result['has_seasonality'] = True
                    dist_result['seasonality_strength'] = np.std(decomposition['seasonal']) / np.std(decomposition['observed'])
                else:
                    dist_result['has_seasonality'] = False
                    dist_result['seasonality_strength'] = None
            except Exception as e:
                logger.error(f"Error in decomposition for {metric}: {e}")
                dist_result['has_seasonality'] = None
                dist_result['seasonality_strength'] = None
        else:
            dist_result['has_seasonality'] = False
            dist_result['seasonality_strength'] = None
            
        # Generate differencing plots
        if metric in df_indexed.columns and len(df_indexed[metric].dropna()) > 50:
            try:
                plot_differencing_comparison(
                    df_indexed[metric],
                    metric,
                    lag=1,
                    seasonal_lag=seasonal_period,
                    output_path=os.path.join(diff_dir, f"{metric}_differencing.png")
                )
            except Exception as e:
                logger.error(f"Error creating differencing plot for {metric}: {e}")
        
        # Combine results
        combined = {**dist_result, **stat_result}
        results.append(combined)
        
        logger.info(f"Analysis for {metric}: Distribution={dist_result['distribution_shape']}, "
                   f"Transform={dist_result['suggested_transform']}, "
                   f"Stationary={stat_result['is_stationary']}")
    
    # Convert to DataFrame for easier viewing
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(os.path.join(output_dir, "metrics_analysis.csv"), index=False)
    
    return results_df

def create_transformed_dataset(df, analysis_results=None, metrics=None, output_dir=None):
    """
    Create a transformed dataset based on analysis results.
    
    Args:
        df: Original DataFrame
        analysis_results: DataFrame with analysis results (from analyze_metrics)
        metrics: List of metrics to transform (if analysis_results not provided)
        output_dir: Directory to save results
        
    Returns:
        DataFrame with transformed variables
    """
    # Create a copy of the dataframe to avoid modifying the original
    transformed_df = df.copy()
    
    # If analysis_results not provided, analyze the metrics
    if analysis_results is None and metrics is not None:
        analysis_results = analyze_metrics(df, metrics, output_dir)
    elif analysis_results is None:
        logger.warning("No analysis results or metrics provided. Using all numeric columns.")
        analysis_results = analyze_metrics(df, None, output_dir)
    
    # Check if we have time column
    has_time_index = 'time' in df.columns
    
    # Create indexed version for time series operations
    if has_time_index:
        df_indexed = df.set_index('time')
    else:
        df_indexed = df.copy()
    
    # Apply transformations based on analysis results
    for _, row in analysis_results.iterrows():
        name = row['name']
        transform = row['suggested_transform']
        is_stationary = row['is_stationary']
        suggested_differencing = row['suggested_differencing']
        has_seasonality = row.get('has_seasonality', False)
        
        # Skip if column doesn't exist
        if name not in df.columns:
            continue
        
        try:
            # 1. Apply distribution transformations
            if transform not in ('none', None):
                logger.info(f"Applying {transform} transformation to {name}")
                
                if transform == 'sine_cosine':
                    # Special handling for directional data - creates two new columns
                    sin_series, cos_series = transform_series(df[name], method=transform)
                    transformed_df[f"{name}_sin"] = sin_series
                    transformed_df[f"{name}_cos"] = cos_series
                    
                elif transform == 'two_part_model':
                    # Special handling for zero-inflated data - creates two new columns
                    binary, continuous = transform_series(df[name], method=transform)
                    transformed_df[f"{name}_binary"] = binary
                    
                    # Apply log transform to the continuous part
                    if continuous.notna().any():
                        transformed_df[f"{name}_value"] = np.log1p(continuous)
                        
                else:
                    # Standard transformation
                    transformed_df[f"{name}_transformed"] = transform_series(df[name], method=transform)
            
            # 2. Apply seasonal adjustment if needed
            if has_seasonality and has_time_index and name in df_indexed.columns:
                try:
                    decomposition = decompose_time_series(
                        df_indexed[name], 
                        period=24,  # Hourly data with daily seasonality
                        method='STL'
                    )
                    
                    if decomposition and 'seasonally_adjusted' in decomposition:
                        # Add seasonally adjusted series
                        transformed_df[f"{name}_seas_adj"] = decomposition['seasonally_adjusted'].values
                        logger.info(f"Added seasonally adjusted series for {name}")
                except Exception as e:
                    logger.error(f"Error in seasonal adjustment for {name}: {e}")
            
            # 3. Apply differencing if needed for non-stationarity
            if (not is_stationary or is_stationary is None) and suggested_differencing != 'none':
                if has_time_index and name in df_indexed.columns:
                    try:
                        if suggested_differencing == 'first_order':
                            # Apply first-order differencing
                            differenced = difference_series(df_indexed[name], lag=1)
                            transformed_df[f"{name}_diff1"] = differenced.values
                            logger.info(f"Applied first-order differencing to {name}")
                            
                        elif suggested_differencing == 'seasonal_and_first_order':
                            # Apply seasonal differencing followed by first-order
                            seasonal_lag = 24  # For hourly data with daily seasonality
                            differenced = difference_series(
                                df_indexed[name], 
                                lag=1, 
                                seasonal_lag=seasonal_lag
                            )
                            transformed_df[f"{name}_diff_seas"] = differenced.values
                            logger.info(f"Applied seasonal and first-order differencing to {name}")
                    except Exception as e:
                        logger.error(f"Error applying differencing to {name}: {e}")
                
        except Exception as e:
            logger.error(f"Error transforming {name}: {e}")
    
    # Save the transformed dataset if output_dir provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        transformed_df.to_csv(os.path.join(output_dir, "transformed_dataset.csv"), index=False)
    
    return transformed_df
