"""
HiGHS solver root gap comparison script.

This script compares the root node gap distributions between two HiGHS solver result datasets
using Wasserstein distance and statistical tests.
This approach is consistent with the Gurobi and SCIP analysis methodology.
"""

import os
import sys
import logging
import hydra
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import wasserstein_distance, ks_2samp, mannwhitneyu
from omegaconf import DictConfig
from datetime import datetime
from typing import Dict, List, Tuple, Any
import json

# Add the project root directory to the system path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@hydra.main(config_path="../configs", config_name="highs_compare_rootgap", version_base=None)
def compare_highs_rootgap(cfg: DictConfig):
    """
    Compare root node gap distributions between two HiGHS solver result datasets.
    
    Args:
        cfg: Hydra configuration object
    """
    # Set up logging
    log_level = getattr(logging, cfg.logging.level)
    logging.basicConfig(
        level=log_level,
        format=cfg.logging.format
    )
    
    logging.info("Starting HiGHS root gap comparison analysis...")
    
    # Get output directory
    try:
        if hasattr(cfg, 'hydra') and hasattr(cfg.hydra, 'run') and hasattr(cfg.hydra.run, 'dir'):
            output_dir = cfg.hydra.run.dir
        else:
            output_dir = cfg.paths.output_dir
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        logging.error(f"Error creating output directory: {str(e)}")
        return
    
    # Load the two datasets
    try:
        logging.info(f"Loading dataset 1: {cfg.paths.csv_file1}")
        df1 = pd.read_csv(cfg.paths.csv_file1)
        
        logging.info(f"Loading dataset 2: {cfg.paths.csv_file2}")
        df2 = pd.read_csv(cfg.paths.csv_file2)
        
        logging.info(f"Dataset 1 shape: {df1.shape}")
        logging.info(f"Dataset 2 shape: {df2.shape}")
        
    except Exception as e:
        logging.error(f"Error loading datasets: {str(e)}")
        return
    
    # Extract root gap data
    try:
        gap1, gap2 = extract_root_gap_data(df1, df2, cfg)
        logging.info(f"Extracted {len(gap1)} and {len(gap2)} valid root gap values")
        
    except Exception as e:
        logging.error(f"Error extracting root gap data: {str(e)}")
        return
    
    # Perform statistical analysis
    try:
        results = perform_statistical_analysis(gap1, gap2, cfg)
        logging.info("Statistical analysis completed successfully")
        
    except Exception as e:
        logging.error(f"Error in statistical analysis: {str(e)}")
        return
    
    # Create comparison plots
    try:
        plot_path = os.path.join(output_dir, "highs_rootgap_comparison.png")
        create_comparison_plots(gap1, gap2, cfg, plot_path, results)
        logging.info(f"Comparison plots saved to {plot_path}")
        
    except Exception as e:
        logging.error(f"Error creating plots: {str(e)}")
    
    # Save results
    try:
        results_path = os.path.join(output_dir, "highs_rootgap_comparison_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Results saved to {results_path}")
        
    except Exception as e:
        logging.error(f"Error saving results: {str(e)}")
    
    # Print summary
    print_summary(results)
    
    logging.info("HiGHS root gap comparison analysis completed!")

def extract_root_gap_data(df1: pd.DataFrame, df2: pd.DataFrame, cfg: DictConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Extract valid root gap data from both datasets."""
    
    # Check if root_gap column exists
    if 'root_gap' not in df1.columns or 'root_gap' not in df2.columns:
        raise ValueError("root_gap column not found in one or both datasets")
    
    # Extract root gap values and filter out invalid ones
    gap1 = df1['root_gap'].dropna()
    gap2 = df2['root_gap'].dropna()
    
    # Filter out zero or negative gaps (invalid)
    gap1 = gap1[gap1 > 0]
    gap2 = gap2[gap2 > 0]
    
    # Filter out extremely large gaps (likely errors)
    gap1 = gap1[gap1 < 1000]  # 1000% gap is unrealistic
    gap2 = gap2[gap2 < 1000]
    
    if len(gap1) == 0 or len(gap2) == 0:
        raise ValueError("No valid root gap data found in one or both datasets")
    
    logging.info(f"Valid root gaps - Dataset 1: {len(gap1)}, Dataset 2: {len(gap2)}")
    
    return gap1.values, gap2.values

def perform_statistical_analysis(gap1: np.ndarray, gap2: np.ndarray, cfg: DictConfig) -> Dict[str, Any]:
    """Perform comprehensive statistical analysis on root gap data."""
    
    results = {
        'dataset1_name': cfg.comparison.name1,
        'dataset2_name': cfg.comparison.name2,
        'dataset1_stats': calculate_descriptive_stats(gap1),
        'dataset2_stats': calculate_descriptive_stats(gap2),
        'wasserstein_distance': float(wasserstein_distance(gap1, gap2)),
        'statistical_tests': {}
    }
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_pvalue = ks_2samp(gap1, gap2)
    results['statistical_tests']['kolmogorov_smirnov'] = {
        'statistic': float(ks_stat),
        'p_value': float(ks_pvalue),
        'significant': ks_pvalue < 0.05
    }
    
    # Mann-Whitney U test
    mw_stat, mw_pvalue = mannwhitneyu(gap1, gap2, alternative='two-sided')
    results['statistical_tests']['mann_whitney_u'] = {
        'statistic': float(mw_stat),
        'p_value': float(mw_pvalue),
        'significant': mw_pvalue < 0.05
    }
    
    # Welch's t-test (for means)
    t_stat, t_pvalue = stats.ttest_ind(gap1, gap2, equal_var=False)
    results['statistical_tests']['welch_t_test'] = {
        'statistic': float(t_stat),
        'p_value': float(t_pvalue),
        'significant': t_pvalue < 0.05
    }
    
    # Levene's test (for variance equality)
    levene_stat, levene_pvalue = stats.levene(gap1, gap2)
    results['statistical_tests']['levene_test'] = {
        'statistic': float(levene_stat),
        'p_value': float(levene_pvalue),
        'significant': levene_pvalue < 0.05
    }
    
    return results

def calculate_descriptive_stats(data: np.ndarray) -> Dict[str, float]:
    """Calculate descriptive statistics for a dataset."""
    
    return {
        'count': int(len(data)),
        'mean': float(np.mean(data)),
        'median': float(np.median(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'q25': float(np.percentile(data, 25)),
        'q75': float(np.percentile(data, 75)),
        'skewness': float(stats.skew(data)),
        'kurtosis': float(stats.kurtosis(data))
    }

def create_comparison_plots(gap1: np.ndarray, gap2: np.ndarray, cfg: DictConfig, 
                          output_path: str, results: Dict[str, Any]):
    """Create comprehensive comparison plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'HiGHS Root Gap Comparison: {cfg.comparison.name1} vs {cfg.comparison.name2}', 
                 fontsize=16)
    
    # Plot 1: Histogram comparison
    ax1 = axes[0, 0]
    ax1.hist(gap1, alpha=0.6, label=cfg.comparison.name1, bins=30, density=True)
    ax1.hist(gap2, alpha=0.6, label=cfg.comparison.name2, bins=30, density=True)
    ax1.set_xlabel('Root Gap (%)')
    ax1.set_ylabel('Density')
    ax1.set_title('Root Gap Distribution Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plot comparison
    ax2 = axes[0, 1]
    data_to_plot = [gap1, gap2]
    labels = [cfg.comparison.name1, cfg.comparison.name2]
    bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax2.set_ylabel('Root Gap (%)')
    ax2.set_title('Root Gap Box Plot Comparison')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Q-Q plot
    ax3 = axes[1, 0]
    # Sort both datasets for Q-Q plot
    gap1_sorted = np.sort(gap1)
    gap2_sorted = np.sort(gap2)
    
    # Interpolate to same length for Q-Q plot
    n_points = min(len(gap1_sorted), len(gap2_sorted))
    gap1_qq = np.interp(np.linspace(0, 1, n_points), 
                        np.linspace(0, 1, len(gap1_sorted)), gap1_sorted)
    gap2_qq = np.interp(np.linspace(0, 1, n_points), 
                        np.linspace(0, 1, len(gap2_sorted)), gap2_sorted)
    
    ax3.scatter(gap1_qq, gap2_qq, alpha=0.6, s=20)
    min_val = min(np.min(gap1_qq), np.min(gap2_qq))
    max_val = max(np.max(gap1_qq), np.max(gap2_qq))
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    ax3.set_xlabel(f'{cfg.comparison.name1} Root Gap (%)')
    ax3.set_ylabel(f'{cfg.comparison.name2} Root Gap (%)')
    ax3.set_title('Q-Q Plot')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Cumulative distribution comparison
    ax4 = axes[1, 1]
    x1_sorted = np.sort(gap1)
    y1 = np.arange(1, len(x1_sorted) + 1) / len(x1_sorted)
    x2_sorted = np.sort(gap2)
    y2 = np.arange(1, len(x2_sorted) + 1) / len(x2_sorted)
    
    ax4.plot(x1_sorted, y1, label=cfg.comparison.name1, linewidth=2)
    ax4.plot(x2_sorted, y2, label=cfg.comparison.name2, linewidth=2)
    ax4.set_xlabel('Root Gap (%)')
    ax4.set_ylabel('Cumulative Probability')
    ax4.set_title('Cumulative Distribution Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add statistical information as text
    wd = results['wasserstein_distance']
    ks_pval = results['statistical_tests']['kolmogorov_smirnov']['p_value']
    
    fig.text(0.02, 0.02, f'Wasserstein Distance: {wd:.4f}\nKS Test p-value: {ks_pval:.4f}', 
             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create additional detailed plots
    create_detailed_plots(gap1, gap2, cfg, os.path.dirname(output_path), results)

def create_detailed_plots(gap1: np.ndarray, gap2: np.ndarray, cfg: DictConfig, 
                         output_dir: str, results: Dict[str, Any]):
    """Create additional detailed plots."""
    
    # Violin plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    data_to_plot = [gap1, gap2]
    labels = [cfg.comparison.name1, cfg.comparison.name2]
    
    parts = ax.violinplot(data_to_plot, positions=[1, 2], showmeans=True, showmedians=True)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels)
    ax.set_ylabel('Root Gap (%)')
    ax.set_title('Root Gap Violin Plot Comparison')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'highs_rootgap_violin.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Log-scale histogram (if data spans multiple orders of magnitude)
    if np.max(gap1) / np.min(gap1) > 100 or np.max(gap2) / np.min(gap2) > 100:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.hist(gap1, alpha=0.6, label=cfg.comparison.name1, bins=30, density=True)
        ax.hist(gap2, alpha=0.6, label=cfg.comparison.name2, bins=30, density=True)
        ax.set_xlabel('Root Gap (%)')
        ax.set_ylabel('Density')
        ax.set_title('Root Gap Distribution (Log Scale)')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'highs_rootgap_log_scale.png'), dpi=300, bbox_inches='tight')
        plt.close()

def print_summary(results: Dict[str, Any]):
    """Print a summary of the root gap comparison results."""
    
    print("\n" + "="*60)
    print("HiGHS ROOT GAP COMPARISON SUMMARY")
    print("="*60)
    
    print(f"Dataset 1: {results['dataset1_name']}")
    print(f"Dataset 2: {results['dataset2_name']}")
    
    # Descriptive statistics
    stats1 = results['dataset1_stats']
    stats2 = results['dataset2_stats']
    
    print(f"\nDescriptive Statistics:")
    print(f"{'Metric':<15} {'Dataset 1':<15} {'Dataset 2':<15}")
    print("-" * 45)
    print(f"{'Count':<15} {stats1['count']:<15} {stats2['count']:<15}")
    print(f"{'Mean':<15} {stats1['mean']:<15.4f} {stats2['mean']:<15.4f}")
    print(f"{'Median':<15} {stats1['median']:<15.4f} {stats2['median']:<15.4f}")
    print(f"{'Std Dev':<15} {stats1['std']:<15.4f} {stats2['std']:<15.4f}")
    print(f"{'Min':<15} {stats1['min']:<15.4f} {stats2['min']:<15.4f}")
    print(f"{'Max':<15} {stats1['max']:<15.4f} {stats2['max']:<15.4f}")
    
    # Distance measures
    print(f"\nDistance Measures:")
    print(f"Wasserstein Distance: {results['wasserstein_distance']:.4f}")
    
    # Statistical tests
    print(f"\nStatistical Tests:")
    tests = results['statistical_tests']
    
    for test_name, test_result in tests.items():
        test_display_name = test_name.replace('_', ' ').title()
        significance = "Significant" if test_result['significant'] else "Not Significant"
        print(f"{test_display_name}:")
        print(f"  Statistic: {test_result['statistic']:.4f}")
        print(f"  P-value: {test_result['p_value']:.4f}")
        print(f"  Result: {significance}")
    
    # Interpretation
    wd = results['wasserstein_distance']
    if wd < 1.0:
        interpretation = "Very similar root gap distributions"
    elif wd < 5.0:
        interpretation = "Moderately similar root gap distributions"
    elif wd < 10.0:
        interpretation = "Somewhat different root gap distributions"
    else:
        interpretation = "Very different root gap distributions"
    
    print(f"\nInterpretation: {interpretation}")
    
    # Practical significance
    mean_diff = abs(stats1['mean'] - stats2['mean'])
    pooled_std = np.sqrt((stats1['std']**2 + stats2['std']**2) / 2)
    effect_size = mean_diff / pooled_std if pooled_std > 0 else 0
    
    print(f"Effect Size (Cohen's d): {effect_size:.4f}")
    
    if effect_size < 0.2:
        effect_interpretation = "Small effect"
    elif effect_size < 0.5:
        effect_interpretation = "Medium effect"
    else:
        effect_interpretation = "Large effect"
    
    print(f"Effect Size Interpretation: {effect_interpretation}")
    print("="*60)

if __name__ == "__main__":
    compare_highs_rootgap()
