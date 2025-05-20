"""
Correlation analysis module for AirSense Copenhagen.
Analyzes relationships between original and transformed variables.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def compute_correlation_matrix(df, variables, method='pearson'):
    """
    Compute correlation matrix for selected variables.
    
    Args:
        df: DataFrame with variables
        variables: List of variable names to include
        method: Correlation method ('pearson', 'spearman', or 'kendall')
        
    Returns:
        DataFrame: Correlation matrix
    """
    # Filter variables that exist in the dataframe
    valid_vars = [var for var in variables if var in df.columns]
    if len(valid_vars) < 2:
        logger.warning(f"Not enough valid variables for correlation analysis. Found only: {valid_vars}")
        return None
    
    # Drop rows with NaN values
    data = df[valid_vars].dropna()
    if len(data) == 0:
        logger.warning("No complete cases found for correlation analysis after removing NaN values")
        return None
    
    # Compute correlation
    corr_matrix = data.corr(method=method)
    return corr_matrix

def plot_correlation_matrix(corr_matrix, title="Correlation Matrix", figsize=(12, 10), output_path=None, 
                           vmin=-1, vmax=1, cmap='coolwarm', annotate=True, mask_upper=False):
    """
    Plot a correlation matrix as a heatmap.
    
    Args:
        corr_matrix: DataFrame with correlation matrix
        title: Plot title
        figsize: Figure size
        output_path: Path to save the plot (if None, the plot is displayed)
        vmin, vmax: Minimum and maximum values for the colorbar
        cmap: Colormap to use
        annotate: Whether to annotate the heatmap with correlation values
        mask_upper: Whether to mask the upper triangle of the matrix
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    if corr_matrix is None or corr_matrix.empty:
        logger.warning("Cannot plot empty correlation matrix")
        return None
    
    # Create the figure
    plt.figure(figsize=figsize)
    
    # Create mask for upper triangle if requested
    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(corr_matrix), k=1)
    
    # Plot heatmap
    ax = sns.heatmap(
        corr_matrix, 
        annot=annotate, 
        fmt=".2f" if annotate else "", 
        cmap=cmap, 
        vmin=vmin, 
        vmax=vmax,
        mask=mask,
        square=True,
        linewidths=.5
    )
    
    # Set title and adjust layout
    plt.title(title, fontsize=14, pad=20)
    plt.tight_layout()
    
    # Save or show
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Correlation matrix plot saved to {output_path}")
    
    return plt.gcf()

def plot_variable_pair(df, x_col, y_col, transformed_x=None, transformed_y=None, 
                      output_path=None, figsize=(12, 6)):
    """
    Create scatter plots for a pair of variables, optionally comparing original and transformed.
    
    Args:
        df: DataFrame with variables
        x_col: Name of the x variable
        y_col: Name of the y variable
        transformed_x: Name of the transformed x variable
        transformed_y: Name of the transformed y variable
        output_path: Path to save the plot
        figsize: Figure size
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Check if variables exist in DataFrame
    required_cols = [x_col, y_col]
    if transformed_x:
        required_cols.append(transformed_x)
    if transformed_y:
        required_cols.append(transformed_y)
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"Columns not found in DataFrame: {missing_cols}")
        return None
    
    # Drop rows with NaN in the required columns
    data = df[required_cols].dropna()
    if len(data) == 0:
        logger.warning("No complete cases found for scatter plot after removing NaN values")
        return None
    
    # Determine number of plots
    has_transformed = transformed_x is not None or transformed_y is not None
    n_plots = 2 if has_transformed else 1
    
    # Create figure
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    
    # Plot 1: Original variables
    sns.scatterplot(x=x_col, y=y_col, data=data, alpha=0.5, ax=axes[0])
    axes[0].set_title(f"{y_col} vs {x_col} (Original)", fontsize=12)
    
    # Calculate and display Pearson correlation
    corr_orig, p_orig = stats.pearsonr(data[x_col], data[y_col])
    axes[0].annotate(f"r = {corr_orig:.3f}\np = {p_orig:.3g}", 
                    xy=(0.05, 0.95), xycoords='axes fraction', 
                    ha='left', va='top', fontsize=12, 
                    bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    
    # Plot regression line
    sns.regplot(x=x_col, y=y_col, data=data, scatter=False, ax=axes[0], 
               line_kws={"color": "red", "lw": 2})
    
    # Plot 2: Transformed variables (if applicable)
    if has_transformed:
        x_to_plot = transformed_x if transformed_x else x_col
        y_to_plot = transformed_y if transformed_y else y_col
        
        sns.scatterplot(x=x_to_plot, y=y_to_plot, data=data, alpha=0.5, ax=axes[1])
        axes[1].set_title(f"{y_col} vs {x_col} (Transformed)", fontsize=12)
        
        # Calculate and display Pearson correlation for transformed variables
        corr_trans, p_trans = stats.pearsonr(data[x_to_plot], data[y_to_plot])
        axes[1].annotate(f"r = {corr_trans:.3f}\np = {p_trans:.3g}", 
                        xy=(0.05, 0.95), xycoords='axes fraction', 
                        ha='left', va='top', fontsize=12, 
                        bbox=dict(boxstyle='round', fc='white', alpha=0.8))
        
        # Plot regression line
        sns.regplot(x=x_to_plot, y=y_to_plot, data=data, scatter=False, ax=axes[1], 
                   line_kws={"color": "red", "lw": 2})
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Variable pair plot saved to {output_path}")
    
    return fig

def compare_correlations(df, original_vars, transformed_vars, method='pearson', 
                        output_dir=None):
    """
    Compare correlations between original and transformed variables.
    
    Args:
        df: DataFrame with variables
        original_vars: List of original variable names
        transformed_vars: List of transformed variable names (matching order of original_vars)
        method: Correlation method ('pearson', 'spearman', or 'kendall')
        output_dir: Directory to save outputs
        
    Returns:
        DataFrame: Comparison of correlations
    """
    # Check if variables exist in DataFrame
    all_vars = original_vars + transformed_vars
    missing_vars = [var for var in all_vars if var not in df.columns]
    if missing_vars:
        logger.warning(f"Variables not found in DataFrame: {missing_vars}")
        return None
    
    # Check that lengths match
    if len(original_vars) != len(transformed_vars):
        logger.error("Lists of original and transformed variables must have the same length")
        return None
    
    # Create mappings between original and transformed variables
    var_mapping = {orig: trans for orig, trans in zip(original_vars, transformed_vars)}
    
    # Compute correlation matrices
    orig_corr = compute_correlation_matrix(df, original_vars, method)
    trans_corr = compute_correlation_matrix(df, transformed_vars, method)
    
    if orig_corr is None or trans_corr is None:
        logger.warning("Could not compute correlation matrices")
        return None
    
    # Create comparison dataframe
    comparisons = []
    for i, var1 in enumerate(original_vars):
        for j, var2 in enumerate(original_vars):
            if i >= j:  # Only use lower triangle to avoid duplicates
                continue
                
            # Get transformed variable names
            trans1 = var_mapping[var1]
            trans2 = var_mapping[var2]
            
            # Get correlation values
            orig_val = orig_corr.loc[var1, var2]
            trans_val = trans_corr.loc[trans1, trans2]
            
            # Calculate improvement
            abs_improvement = abs(trans_val) - abs(orig_val)
            
            comparisons.append({
                'Variable1': var1,
                'Variable2': var2,
                'Original_Correlation': orig_val,
                'Transformed_Correlation': trans_val,
                'Absolute_Improvement': abs_improvement
            })
    
    comparison_df = pd.DataFrame(comparisons)
    
    # Sort by absolute improvement in descending order
    comparison_df = comparison_df.sort_values('Absolute_Improvement', ascending=False)
    
    # Save comparison to CSV
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        comparison_df.to_csv(os.path.join(output_dir, 'correlation_comparison.csv'), index=False)
        logger.info(f"Correlation comparison saved to {output_dir}/correlation_comparison.csv")
    
    return comparison_df

def analyze_correlations(df, metrics=None, transformed_suffix='_transformed', 
                        output_dir="figures/correlations"):
    """
    Perform a comprehensive correlation analysis on the dataset.
    
    Args:
        df: DataFrame with variables to analyze
        metrics: List of original metrics to analyze (if None, will try to infer from transformed variables)
        transformed_suffix: Suffix used for transformed variables
        output_dir: Directory to save outputs
        
    Returns:
        dict: Dictionary with analysis results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # If metrics not provided, try to infer from transformed variables
    if metrics is None:
        metrics = []
        transformed_cols = [col for col in df.columns if col.endswith(transformed_suffix)]
        for col in transformed_cols:
            original_col = col.replace(transformed_suffix, '')
            if original_col in df.columns:
                metrics.append(original_col)
        
        if not metrics:
            logger.warning("Could not infer metrics from transformed variables")
            return None
    
    # Find transformed variables
    transformed_metrics = []
    for metric in metrics:
        transformed_col = f"{metric}{transformed_suffix}"
        if transformed_col in df.columns:
            transformed_metrics.append(transformed_col)
        else:
            # No transformed version found, use original
            transformed_metrics.append(metric)
    
    # 1. Create correlation matrices for original and transformed variables
    print("Generating correlation matrices...")
    
    # Original variables
    orig_corr = compute_correlation_matrix(df, metrics, method='pearson')
    if orig_corr is not None:
        plot_correlation_matrix(
            orig_corr,
            title="Correlation Matrix - Original Variables",
            output_path=os.path.join(output_dir, "original_correlation_matrix.png"),
            mask_upper=True
        )
        # Save to CSV
        orig_corr.to_csv(os.path.join(output_dir, "original_correlation_matrix.csv"))
        print(f"- Saved original correlation matrix (shape: {orig_corr.shape})")
    
    # Transformed variables
    trans_corr = compute_correlation_matrix(df, transformed_metrics, method='pearson')
    if trans_corr is not None:
        plot_correlation_matrix(
            trans_corr,
            title="Correlation Matrix - Transformed Variables",
            output_path=os.path.join(output_dir, "transformed_correlation_matrix.png"),
            mask_upper=True
        )
        # Save to CSV
        trans_corr.to_csv(os.path.join(output_dir, "transformed_correlation_matrix.csv"))
        print(f"- Saved transformed correlation matrix (shape: {trans_corr.shape})")
    
    # 2. Compare original vs transformed correlations
    print("Comparing original and transformed correlations...")
    comparison = compare_correlations(
        df, metrics, transformed_metrics, method='pearson',
        output_dir=output_dir
    )
    
    if comparison is not None:
        # Display top improvements
        top_improvements = comparison.nlargest(5, 'Absolute_Improvement')
        print("\nTop 5 correlation improvements from transformations:")
        for _, row in top_improvements.iterrows():
            print(f"- {row['Variable1']} vs {row['Variable2']}: {row['Original_Correlation']:.3f} → {row['Transformed_Correlation']:.3f} (Δ = {row['Absolute_Improvement']:.3f})")
    
    # 3. Generate scatter plots for variables with significant improvements
    if comparison is not None and len(comparison) > 0:
        print("\nGenerating scatter plots for improved variable pairs...")
        pairs_dir = os.path.join(output_dir, "variable_pairs")
        os.makedirs(pairs_dir, exist_ok=True)
        
        # Get top 10 improved pairs
        top_pairs = comparison.nlargest(10, 'Absolute_Improvement')
        
        for _, row in top_pairs.iterrows():
            var1 = row['Variable1']
            var2 = row['Variable2']
            trans1 = var1 + transformed_suffix if var1 + transformed_suffix in df.columns else var1
            trans2 = var2 + transformed_suffix if var2 + transformed_suffix in df.columns else var2
            
            # Create plot
            plot_variable_pair(
                df, 
                x_col=var1, 
                y_col=var2,
                transformed_x=trans1 if trans1 != var1 else None,
                transformed_y=trans2 if trans2 != var2 else None,
                output_path=os.path.join(pairs_dir, f"{var1}_vs_{var2}.png")
            )
            print(f"- Created scatter plot for {var1} vs {var2}")
    
    # 4. Generate correlation submatrix for specific variable groups
    try:
        # Air quality vs weather
        pollutants = [m for m in metrics if m in ['pm10', 'pm2_5', 'nitrogen_dioxide', 'ozone', 'AQI']]
        weather_vars = [m for m in metrics if m in ['temperature_2m', 'relative_humidity_2m', 
                                                  'precipitation', 'windspeed_10m', 
                                                  'pressure_msl', 'winddirection_10m']]
        
        if pollutants and weather_vars:
            print("\nAnalyzing pollutants vs weather correlations...")
            
            # Get corresponding transformed variables
            trans_pollutants = [m + transformed_suffix if m + transformed_suffix in df.columns else m 
                              for m in pollutants]
            trans_weather = [m + transformed_suffix if m + transformed_suffix in df.columns else m 
                           for m in weather_vars]
            
            # Calculate cross-correlations
            cross_corr_orig = df[pollutants + weather_vars].corr().loc[pollutants, weather_vars]
            cross_corr_trans = df[trans_pollutants + trans_weather].corr().loc[trans_pollutants, trans_weather]
            
            # Plot cross-correlations
            plt.figure(figsize=(14, 8))
            
            # Original variables
            plt.subplot(1, 2, 1)
            sns.heatmap(cross_corr_orig, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
            plt.title("Pollutants vs Weather (Original)", fontsize=12)
            
            # Transformed variables
            plt.subplot(1, 2, 2)
            sns.heatmap(cross_corr_trans, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
            plt.title("Pollutants vs Weather (Transformed)", fontsize=12)
            
            plt.tight_layout()
            cross_corr_path = os.path.join(output_dir, "pollutants_vs_weather_correlation.png")
            plt.savefig(cross_corr_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save to CSV
            cross_corr_orig.to_csv(os.path.join(output_dir, "pollutants_vs_weather_correlation_original.csv"))
            cross_corr_trans.to_csv(os.path.join(output_dir, "pollutants_vs_weather_correlation_transformed.csv"))
            
            print(f"- Saved pollutants vs weather correlation analysis")
            
        # Traffic vs air quality
        if 'traffic_count' in metrics and pollutants:
            print("\nAnalyzing traffic vs pollutants correlations...")
            
            traffic_var = 'traffic_count'
            traffic_trans = traffic_var + transformed_suffix if traffic_var + transformed_suffix in df.columns else traffic_var
            
            # For each pollutant, create a separate plot
            for i, pollutant in enumerate(pollutants):
                pollutant_trans = pollutant + transformed_suffix if pollutant + transformed_suffix in df.columns else pollutant
                
                plot_variable_pair(
                    df, 
                    x_col=traffic_var, 
                    y_col=pollutant,
                    transformed_x=traffic_trans if traffic_trans != traffic_var else None,
                    transformed_y=pollutant_trans if pollutant_trans != pollutant else None,
                    output_path=os.path.join(output_dir, f"traffic_vs_{pollutant}.png")
                )
                print(f"- Created scatter plot for traffic vs {pollutant}")
    except Exception as e:
        logger.error(f"Error generating group correlations: {e}")
    
    # Return results
    return {
        "original_correlation": orig_corr,
        "transformed_correlation": trans_corr,
        "comparison": comparison,
        "output_dir": output_dir
    }

# Main function to run from command line
if __name__ == "__main__":
    # This allows running this module directly for testing
    import sys
    import pandas as pd
    
    # Check if file path is provided
    if len(sys.argv) < 2:
        print("Usage: python correlation.py <transformed_data_csv>")
        sys.exit(1)
    
    # Load data
    data_path = sys.argv[1]
    try:
        data = pd.read_csv(data_path)
        print(f"Loaded data from {data_path} with shape {data.shape}")
        
        # Run analysis
        results = analyze_correlations(data)
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
