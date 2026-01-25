"""
SCIP solver root gap comparison script.

This script compares the root node gap distributions between two SCIP solver result datasets
using Wasserstein distance and provides statistical analysis and visualization.
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
from scipy.stats import wasserstein_distance
from omegaconf import DictConfig
from datetime import datetime
from typing import Dict, List, Tuple, Any
import json

# Add the project root directory to the system path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@hydra.main(config_path="../configs", config_name="scip_compare_rootgap", version_base=None)
def compare_scip_rootgap(cfg: DictConfig):
    """
    Compare root gap distributions between two SCIP solver result datasets.
    
    Args:
        cfg: Hydra configuration object
    """
    # Set up logging
    log_level = getattr(logging, cfg.logging.level)
    logging.basicConfig(
        level=log_level,
        format=cfg.logging.format
    )
    
    logging.info("Starting SCIP root gap comparison analysis...")
    
    # Get output directory
    try:
        if hasattr(cfg, 'hydra') and hasattr(cfg.hydra, 'run') and hasattr(cfg.hydra.run, 'dir'):
            output_dir = cfg.hydra.run.dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"outputs/scip_rootgap_comparison_{timestamp}"
            os.makedirs(output_dir, exist_ok=True)
    except Exception:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/scip_rootgap_comparison_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Output directory: {output_dir}")
    
    # Load datasets
    try:
        df1 = pd.read_csv(cfg.data.dataset1_path)
        df2 = pd.read_csv(cfg.data.dataset2_path)
        logging.info(f"Loaded dataset 1: {len(df1)} instances from {cfg.data.dataset1_path}")
        logging.info(f"Loaded dataset 2: {len(df2)} instances from {cfg.data.dataset2_path}")
    except Exception as e:
        logging.error(f"Error loading datasets: {str(e)}")
        return
    
    # Extract root gap data
    gap_col = cfg.analysis.root_gap_column
    if gap_col not in df1.columns or gap_col not in df2.columns:
        logging.error(f"Root gap column '{gap_col}' not found in one or both datasets")
        return
    
    # Get root gap values and remove NaN
    gaps1 = df1[gap_col].dropna().values
    gaps2 = df2[gap_col].dropna().values
    
    logging.info(f"Dataset 1 valid root gaps: {len(gaps1)}")
    logging.info(f"Dataset 2 valid root gaps: {len(gaps2)}")
    
    if len(gaps1) == 0 or len(gaps2) == 0:
        logging.error("One or both datasets have no valid root gap data")
        return
    
    # Remove outliers if requested
    if cfg.analysis.remove_outliers:
        gaps1_clean = remove_outliers(gaps1, cfg.analysis.outlier_method, cfg.analysis.outlier_threshold)
        gaps2_clean = remove_outliers(gaps2, cfg.analysis.outlier_method, cfg.analysis.outlier_threshold)
        logging.info(f"After outlier removal - Dataset 1: {len(gaps1_clean)}, Dataset 2: {len(gaps2_clean)}")
    else:
        gaps1_clean = gaps1
        gaps2_clean = gaps2
    
    # Perform statistical analysis
    results = perform_statistical_analysis(gaps1_clean, gaps2_clean, cfg)
    
    # Add metadata to results
    results['metadata'] = {
        'dataset1_path': cfg.data.dataset1_path,
        'dataset2_path': cfg.data.dataset2_path,
        'dataset1_name': cfg.data.dataset1_name,
        'dataset2_name': cfg.data.dataset2_name,
        'analysis_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'solver': 'SCIP',
        'original_sizes': {'dataset1': len(gaps1), 'dataset2': len(gaps2)},
        'cleaned_sizes': {'dataset1': len(gaps1_clean), 'dataset2': len(gaps2_clean)},
        'outliers_removed': cfg.analysis.remove_outliers
    }
    
    # Generate visualization
    if cfg.visualization.save_plots:
        plot_path = os.path.join(output_dir, cfg.output.plot_filename)
        create_comparison_plots(gaps1_clean, gaps2_clean, cfg, plot_path, results)
    
    # Save results
    if cfg.output.save_results:
        results_path = os.path.join(output_dir, cfg.output.result_filename)
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        results_serializable = convert_numpy_types(results)
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, ensure_ascii=False, indent=4)
        logging.info(f"Results saved to {results_path}")
    
    # Print summary
    print_summary(results)
    
    logging.info("SCIP root gap comparison analysis completed successfully")

def remove_outliers(data: np.ndarray, method: str, threshold: float) -> np.ndarray:
    """Remove outliers from data using specified method."""
    if method == "iqr":
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return data[(data >= lower_bound) & (data <= upper_bound)]
    elif method == "zscore":
        z_scores = np.abs(stats.zscore(data))
        return data[z_scores < threshold]
    else:
        logging.warning(f"Unknown outlier removal method: {method}. Returning original data.")
        return data

def perform_statistical_analysis(gaps1: np.ndarray, gaps2: np.ndarray, cfg: DictConfig) -> Dict[str, Any]:
    """Perform comprehensive statistical analysis on root gap distributions."""
    results = {}
    
    # Basic descriptive statistics
    results['descriptive_stats'] = {
        'dataset1': {
            'mean': float(np.mean(gaps1)),
            'median': float(np.median(gaps1)),
            'std': float(np.std(gaps1)),
            'min': float(np.min(gaps1)),
            'max': float(np.max(gaps1)),
            'q25': float(np.percentile(gaps1, 25)),
            'q75': float(np.percentile(gaps1, 75))
        },
        'dataset2': {
            'mean': float(np.mean(gaps2)),
            'median': float(np.median(gaps2)),
            'std': float(np.std(gaps2)),
            'min': float(np.min(gaps2)),
            'max': float(np.max(gaps2)),
            'q25': float(np.percentile(gaps2, 25)),
            'q75': float(np.percentile(gaps2, 75))
        }
    }
    
    # Wasserstein distance
    wasserstein_dist = wasserstein_distance(gaps1, gaps2)
    results['wasserstein_distance'] = float(wasserstein_dist)
    
    # Statistical tests
    # Mann-Whitney U test (non-parametric)
    mannwhitney_stat, mannwhitney_p = stats.mannwhitneyu(gaps1, gaps2, alternative='two-sided')
    results['mann_whitney_test'] = {
        'statistic': float(mannwhitney_stat),
        'p_value': float(mannwhitney_p),
        'significant': mannwhitney_p < (1 - cfg.statistics.confidence_level)
    }
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_p = stats.ks_2samp(gaps1, gaps2)
    results['kolmogorov_smirnov_test'] = {
        'statistic': float(ks_stat),
        'p_value': float(ks_p),
        'significant': ks_p < (1 - cfg.statistics.confidence_level)
    }
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(gaps1) - 1) * np.var(gaps1) + (len(gaps2) - 1) * np.var(gaps2)) / 
                        (len(gaps1) + len(gaps2) - 2))
    cohens_d = (np.mean(gaps1) - np.mean(gaps2)) / pooled_std if pooled_std > 0 else 0
    results['effect_size'] = {
        'cohens_d': float(cohens_d),
        'interpretation': interpret_cohens_d(cohens_d)
    }
    
    # Bootstrap confidence intervals for Wasserstein distance
    if cfg.statistics.bootstrap_samples > 0:
        bootstrap_distances = []
        for _ in range(cfg.statistics.bootstrap_samples):
            sample1 = np.random.choice(gaps1, size=len(gaps1), replace=True)
            sample2 = np.random.choice(gaps2, size=len(gaps2), replace=True)
            bootstrap_distances.append(wasserstein_distance(sample1, sample2))
        
        bootstrap_distances = np.array(bootstrap_distances)
        alpha = 1 - cfg.statistics.confidence_level
        ci_lower = np.percentile(bootstrap_distances, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_distances, 100 * (1 - alpha / 2))
        
        results['wasserstein_confidence_interval'] = {
            'lower': float(ci_lower),
            'upper': float(ci_upper),
            'confidence_level': cfg.statistics.confidence_level
        }
    
    return results

def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"

def create_comparison_plots(gaps1: np.ndarray, gaps2: np.ndarray, cfg: DictConfig, 
                          output_path: str, results: Dict[str, Any]):
    """Create comprehensive comparison plots."""
    fig = plt.figure(figsize=cfg.visualization.figure_size)
    
    # Create a 2x2 subplot layout
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Overlapping histograms
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(gaps1, bins=cfg.visualization.bins, alpha=cfg.visualization.alpha, 
             color=cfg.visualization.colors[0], label=cfg.data.dataset1_name, density=True)
    ax1.hist(gaps2, bins=cfg.visualization.bins, alpha=cfg.visualization.alpha, 
             color=cfg.visualization.colors[1], label=cfg.data.dataset2_name, density=True)
    ax1.set_xlabel('Root Gap (%)')
    ax1.set_ylabel('Density')
    ax1.set_title('Root Gap Distribution Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plots
    ax2 = fig.add_subplot(gs[0, 1])
    box_data = [gaps1, gaps2]
    box_labels = [cfg.data.dataset1_name, cfg.data.dataset2_name]
    bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
    bp['boxes'][0].set_facecolor(cfg.visualization.colors[0])
    bp['boxes'][1].set_facecolor(cfg.visualization.colors[1])
    ax2.set_ylabel('Root Gap (%)')
    ax2.set_title('Root Gap Box Plot Comparison')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Q-Q plot
    ax3 = fig.add_subplot(gs[1, 0])
    # Sort both datasets for Q-Q plot
    sorted1 = np.sort(gaps1)
    sorted2 = np.sort(gaps2)
    # Interpolate to same length for comparison
    min_len = min(len(sorted1), len(sorted2))
    q1 = np.interp(np.linspace(0, 1, min_len), np.linspace(0, 1, len(sorted1)), sorted1)
    q2 = np.interp(np.linspace(0, 1, min_len), np.linspace(0, 1, len(sorted2)), sorted2)
    ax3.scatter(q1, q2, alpha=0.6, s=20)
    min_val = min(np.min(q1), np.min(q2))
    max_val = max(np.max(q1), np.max(q2))
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    ax3.set_xlabel(f'{cfg.data.dataset1_name} Quantiles')
    ax3.set_ylabel(f'{cfg.data.dataset2_name} Quantiles')
    ax3.set_title('Q-Q Plot')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics text
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Create summary text
    summary_text = f"""Statistical Summary:
    
Wasserstein Distance: {results['wasserstein_distance']:.4f}

{cfg.data.dataset1_name}:
  Mean: {results['descriptive_stats']['dataset1']['mean']:.2f}%
  Median: {results['descriptive_stats']['dataset1']['median']:.2f}%
  Std: {results['descriptive_stats']['dataset1']['std']:.2f}%

{cfg.data.dataset2_name}:
  Mean: {results['descriptive_stats']['dataset2']['mean']:.2f}%
  Median: {results['descriptive_stats']['dataset2']['median']:.2f}%
  Std: {results['descriptive_stats']['dataset2']['std']:.2f}%

Effect Size (Cohen's d): {results['effect_size']['cohens_d']:.3f}
Interpretation: {results['effect_size']['interpretation']}

Mann-Whitney U Test:
  p-value: {results['mann_whitney_test']['p_value']:.4f}
  Significant: {results['mann_whitney_test']['significant']}
"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle('SCIP Root Gap Distribution Comparison', fontsize=14, fontweight='bold')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Comparison plots saved to {output_path}")

def print_summary(results: Dict[str, Any]):
    """Print a summary of the comparison results."""
    print("\n" + "="*60)
    print("SCIP ROOT GAP COMPARISON SUMMARY")
    print("="*60)
    
    meta = results['metadata']
    print(f"Analysis Time: {meta['analysis_time']}")
    print(f"Solver: {meta['solver']}")
    print(f"Dataset 1: {meta['dataset1_name']} ({meta['cleaned_sizes']['dataset1']} instances)")
    print(f"Dataset 2: {meta['dataset2_name']} ({meta['cleaned_sizes']['dataset2']} instances)")
    
    print(f"\nWasserstein Distance: {results['wasserstein_distance']:.6f}")
    
    if 'wasserstein_confidence_interval' in results:
        ci = results['wasserstein_confidence_interval']
        print(f"95% Confidence Interval: [{ci['lower']:.6f}, {ci['upper']:.6f}]")
    
    print(f"\nEffect Size (Cohen's d): {results['effect_size']['cohens_d']:.4f}")
    print(f"Interpretation: {results['effect_size']['interpretation']}")
    
    print(f"\nMann-Whitney U Test p-value: {results['mann_whitney_test']['p_value']:.6f}")
    print(f"Distributions significantly different: {results['mann_whitney_test']['significant']}")
    
    print(f"\nKolmogorov-Smirnov Test p-value: {results['kolmogorov_smirnov_test']['p_value']:.6f}")
    print(f"Distributions significantly different: {results['kolmogorov_smirnov_test']['significant']}")
    
    print("="*60)

if __name__ == "__main__":
    compare_scip_rootgap()
