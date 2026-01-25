"""
SCIP solver heuristic methods comparison script.

This script compares the heuristic method usage patterns between two SCIP solver result datasets
using PCA dimensionality reduction and Wasserstein distance on principal components.
This approach is consistent with the Gurobi analysis methodology.
"""

import os
import sys
import logging
import hydra
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import wasserstein_distance
from omegaconf import DictConfig
from datetime import datetime
from typing import Dict, List, Tuple, Any
import json

# Add the project root directory to the system path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@hydra.main(config_path="../configs", config_name="scip_compare_heuristic", version_base=None)
def compare_scip_heuristics(cfg: DictConfig):
    """
    Compare heuristic method usage patterns using PCA between two SCIP solver result datasets.
    
    Args:
        cfg: Hydra configuration object
    """
    # Set up logging
    log_level = getattr(logging, cfg.logging.level)
    logging.basicConfig(
        level=log_level,
        format=cfg.logging.format
    )
    
    logging.info("Starting SCIP heuristic methods comparison analysis...")
    
    # Get output directory
    try:
        if hasattr(cfg, 'hydra') and hasattr(cfg.hydra, 'run') and hasattr(cfg.hydra.run, 'dir'):
            output_dir = cfg.hydra.run.dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"outputs/scip_heuristic_comparison_{timestamp}"
            os.makedirs(output_dir, exist_ok=True)
    except Exception:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/scip_heuristic_comparison_{timestamp}"
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
    
    # Extract and prepare heuristic features
    heur_features1, heur_features2, feature_names = prepare_heuristic_features(df1, df2, cfg)
    
    if heur_features1.shape[1] == 0:
        logging.error("No valid heuristic features found")
        return
    
    logging.info(f"Prepared heuristic features: {heur_features1.shape[1]} features")
    
    # Perform PCA analysis
    pca_results = perform_pca_analysis(heur_features1, heur_features2, feature_names, cfg)
    
    # Calculate Wasserstein distances on principal components
    wasserstein_results = calculate_pca_wasserstein_distances(pca_results, cfg)
    
    # Combine results
    results = {
        'pca_analysis': pca_results,
        'wasserstein_distances': wasserstein_results,
        'feature_info': {
            'feature_names': feature_names,
            'n_features': len(feature_names),
            'n_components': cfg.analysis.pca_components
        }
    }
    
    # Add metadata to results
    results['metadata'] = {
        'dataset1_path': cfg.data.dataset1_path,
        'dataset2_path': cfg.data.dataset2_path,
        'dataset1_name': cfg.data.dataset1_name,
        'dataset2_name': cfg.data.dataset2_name,
        'analysis_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'solver': 'SCIP',
        'methodology': 'PCA + Wasserstein Distance',
        'dataset_sizes': {'dataset1': len(df1), 'dataset2': len(df2)},
        'min_usage_threshold': cfg.analysis.min_usage_threshold
    }
    
    # Generate visualization
    if cfg.visualization.save_plots:
        plot_path = os.path.join(output_dir, cfg.output.plot_filename)
        create_pca_comparison_plots(pca_results, cfg, plot_path, results)
    
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
    print_pca_summary(results)
    
    logging.info("SCIP heuristic methods comparison analysis completed successfully")

def prepare_heuristic_features(df1: pd.DataFrame, df2: pd.DataFrame, cfg: DictConfig) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Prepare heuristic features for PCA analysis."""
    
    # Extract heuristic columns
    heur_prefix = cfg.analysis.heuristic_prefix
    heur_cols1 = [col for col in df1.columns if col.startswith(heur_prefix)]
    heur_cols2 = [col for col in df2.columns if col.startswith(heur_prefix)]
    
    # Get common heuristic columns
    common_heur_cols = list(set(heur_cols1) & set(heur_cols2))
    
    if not common_heur_cols:
        logging.error("No common heuristic columns found")
        return np.array([]), np.array([]), []
    
    # Fill NaN values with 0
    for col in common_heur_cols:
        df1[col] = df1[col].fillna(0)
        df2[col] = df2[col].fillna(0)
    
    # Filter by minimum usage threshold
    filtered_cols = []
    for col in common_heur_cols:
        total_usage = df1[col].sum() + df2[col].sum()
        if total_usage >= cfg.analysis.min_usage_threshold:
            filtered_cols.append(col)
    
    if not filtered_cols:
        logging.error("No heuristic columns meet the minimum usage threshold")
        return np.array([]), np.array([]), []
    
    logging.info(f"Using {len(filtered_cols)} heuristic features after filtering")
    
    # Extract feature matrices
    features1 = df1[filtered_cols].values
    features2 = df2[filtered_cols].values
    
    # Remove zero variance features if requested
    if cfg.analysis.remove_zero_variance:
        # Calculate variance across both datasets
        combined_features = np.vstack([features1, features2])
        feature_variances = np.var(combined_features, axis=0)
        non_zero_var_mask = feature_variances > 1e-10
        
        features1 = features1[:, non_zero_var_mask]
        features2 = features2[:, non_zero_var_mask]
        filtered_cols = [col for i, col in enumerate(filtered_cols) if non_zero_var_mask[i]]
        
        logging.info(f"After removing zero variance features: {len(filtered_cols)} features")
    
    return features1, features2, filtered_cols

def perform_pca_analysis(features1: np.ndarray, features2: np.ndarray, 
                        feature_names: List[str], cfg: DictConfig) -> Dict[str, Any]:
    """Perform PCA analysis on cutting plane features."""
    
    # Combine datasets for consistent PCA fitting
    combined_features = np.vstack([features1, features2])
    
    # Normalize features if requested
    if cfg.analysis.normalize_features:
        scaler = StandardScaler()
        combined_features_scaled = scaler.fit_transform(combined_features)
        features1_scaled = combined_features_scaled[:len(features1)]
        features2_scaled = combined_features_scaled[len(features1):]
    else:
        features1_scaled = features1
        features2_scaled = features2
        scaler = None
    
    # Fit PCA
    n_components = min(cfg.analysis.pca_components, features1_scaled.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(combined_features_scaled if cfg.analysis.normalize_features else combined_features)
    
    # Transform datasets
    pca_features1 = pca.transform(features1_scaled)
    pca_features2 = pca.transform(features2_scaled)
    
    # Calculate component statistics
    component_stats = {}
    for i in range(n_components):
        component_stats[f'PC{i+1}'] = {
            'explained_variance_ratio': float(pca.explained_variance_ratio_[i]),
            'explained_variance': float(pca.explained_variance_[i]),
            'dataset1_mean': float(np.mean(pca_features1[:, i])),
            'dataset1_std': float(np.std(pca_features1[:, i])),
            'dataset2_mean': float(np.mean(pca_features2[:, i])),
            'dataset2_std': float(np.std(pca_features2[:, i]))
        }
    
    # Get feature loadings (components)
    loadings = pca.components_
    
    # Find top contributing features for each component
    top_features = {}
    for i in range(n_components):
        # Get absolute loadings for this component
        abs_loadings = np.abs(loadings[i])
        # Get indices of top 5 features
        top_indices = np.argsort(abs_loadings)[-5:][::-1]
        top_features[f'PC{i+1}'] = [
            {
                'feature': feature_names[idx],
                'loading': float(loadings[i][idx]),
                'abs_loading': float(abs_loadings[idx])
            }
            for idx in top_indices
        ]

    return {
        'pca_features1': pca_features1,
        'pca_features2': pca_features2,
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist(),
        'component_stats': component_stats,
        'top_features': top_features,
        'loadings': loadings.tolist(),
        'feature_names': feature_names,
        'scaler_used': cfg.analysis.normalize_features,
        'n_components': n_components
    }

def calculate_pca_wasserstein_distances(pca_results: Dict[str, Any], cfg: DictConfig) -> Dict[str, Any]:
    """Calculate Wasserstein distances on PCA components."""
    
    pca_features1 = pca_results['pca_features1']
    pca_features2 = pca_results['pca_features2']
    n_components = pca_results['n_components']

    wasserstein_distances = {}
    statistical_tests = {}

    for i in range(n_components):
        pc_name = f'PC{i+1}'

        # Extract component values
        pc1_values = pca_features1[:, i]
        pc2_values = pca_features2[:, i]

        # Calculate Wasserstein distance
        wd = wasserstein_distance(pc1_values, pc2_values)
        wasserstein_distances[pc_name] = float(wd)

        # Perform statistical tests
        try:
            # Mann-Whitney U test
            mw_stat, mw_p = stats.mannwhitneyu(pc1_values, pc2_values, alternative='two-sided')

            # Kolmogorov-Smirnov test
            ks_stat, ks_p = stats.ks_2samp(pc1_values, pc2_values)

            statistical_tests[pc_name] = {
                'mann_whitney_statistic': float(mw_stat),
                'mann_whitney_p_value': float(mw_p),
                'mann_whitney_significant': mw_p < (1 - cfg.statistics.confidence_level),
                'ks_statistic': float(ks_stat),
                'ks_p_value': float(ks_p),
                'ks_significant': ks_p < (1 - cfg.statistics.confidence_level)
            }
        except Exception as e:
            logging.warning(f"Could not perform statistical tests for {pc_name}: {str(e)}")
            statistical_tests[pc_name] = {'error': str(e)}

    # Calculate overall metrics
    total_wd = sum(wasserstein_distances.values())
    weighted_wd = sum(
        wasserstein_distances[f'PC{i+1}'] * pca_results['explained_variance_ratio'][i]
        for i in range(n_components)
    )

    return {
        'component_distances': wasserstein_distances,
        'statistical_tests': statistical_tests,
        'total_wasserstein_distance': float(total_wd),
        'weighted_wasserstein_distance': float(weighted_wd),
        'average_wasserstein_distance': float(total_wd / n_components)
    }

def create_pca_comparison_plots(pca_results: Dict[str, Any], cfg: DictConfig, 
                              output_path: str, results: Dict[str, Any]):
    """Create comprehensive PCA comparison plots."""
    
    fig = plt.figure(figsize=cfg.visualization.figure_size)
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
    
    pca_features1 = pca_results['pca_features1']
    pca_features2 = pca_results['pca_features2']
    n_components = pca_results['n_components']
    
    # Plot 1: Explained variance
    ax1 = fig.add_subplot(gs[0, 0])
    components = [f'PC{i+1}' for i in range(n_components)]
    explained_var = pca_results['explained_variance_ratio']
    cumulative_var = pca_results['cumulative_variance_ratio']
    
    ax1.bar(components, explained_var, alpha=0.7, color='skyblue', label='Individual')
    ax1.plot(components, cumulative_var, 'ro-', color='red', label='Cumulative')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('PCA Explained Variance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: PC1 vs PC2 scatter plot
    if n_components >= 2:
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(pca_features1[:, 0], pca_features1[:, 1], 
                   alpha=cfg.visualization.alpha, color=cfg.visualization.colors[0], 
                   label=cfg.data.dataset1_name, s=20)
        ax2.scatter(pca_features2[:, 0], pca_features2[:, 1], 
                   alpha=cfg.visualization.alpha, color=cfg.visualization.colors[1], 
                   label=cfg.data.dataset2_name, s=20)
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.set_title('PC1 vs PC2 Scatter Plot')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Wasserstein distances by component
    ax3 = fig.add_subplot(gs[1, 0])
    wd_values = [results['wasserstein_distances']['component_distances'][f'PC{i+1}'] for i in range(n_components)]
    bars = ax3.bar(components, wd_values, color='purple', alpha=0.7)
    ax3.set_ylabel('Wasserstein Distance')
    ax3.set_title('Wasserstein Distance by Principal Component')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, wd_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(wd_values)*0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Component distributions (PC1)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(pca_features1[:, 0], bins=30, alpha=0.7, color=cfg.visualization.colors[0], 
             label=cfg.data.dataset1_name, density=True)
    ax4.hist(pca_features2[:, 0], bins=30, alpha=0.7, color=cfg.visualization.colors[1], 
             label=cfg.data.dataset2_name, density=True)
    ax4.set_xlabel('PC1 Values')
    ax4.set_ylabel('Density')
    ax4.set_title('PC1 Distribution Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Top feature loadings for PC1
    ax5 = fig.add_subplot(gs[2, 0])
    if 'PC1' in pca_results['top_features']:
        top_features_pc1 = pca_results['top_features']['PC1'][:5]
        feature_names = [f['feature'].replace('cut_', '') for f in top_features_pc1]
        loadings = [f['loading'] for f in top_features_pc1]
        
        colors = ['red' if l < 0 else 'blue' for l in loadings]
        bars = ax5.barh(feature_names, loadings, color=colors, alpha=0.7)
        ax5.set_xlabel('Loading')
        ax5.set_title('Top 5 Feature Loadings for PC1')
        ax5.grid(True, alpha=0.3)
    
    # Plot 6: Summary statistics
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    
    # Create summary text
    wd_results = results['wasserstein_distances']
    summary_text = f"""PCA Analysis Summary:

Total Components: {n_components}
Total Explained Variance: {sum(explained_var):.3f}

Wasserstein Distances:
Total: {wd_results['total_wasserstein_distance']:.4f}
Weighted: {wd_results['weighted_wasserstein_distance']:.4f}
Average: {wd_results['average_wasserstein_distance']:.4f}

Top Component by Distance:
PC1: {wd_results['component_distances']['PC1']:.4f}

Dataset Sizes:
{cfg.data.dataset1_name}: {results['metadata']['dataset_sizes']['dataset1']}
{cfg.data.dataset2_name}: {results['metadata']['dataset_sizes']['dataset2']}"""
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle('SCIP Cutting Planes PCA Comparison', fontsize=16, fontweight='bold')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Comparison plots saved to {output_path}")

def print_pca_summary(results: Dict[str, Any]):
    """Print a summary of the PCA comparison results."""
    print("\n" + "="*70)
    print("SCIP HEURISTIC METHODS PCA COMPARISON SUMMARY")
    print("="*70)
    
    meta = results['metadata']
    print(f"Analysis Time: {meta['analysis_time']}")
    print(f"Solver: {meta['solver']}")
    print(f"Methodology: {meta['methodology']}")
    print(f"Dataset 1: {meta['dataset1_name']} ({meta['dataset_sizes']['dataset1']} instances)")
    print(f"Dataset 2: {meta['dataset2_name']} ({meta['dataset_sizes']['dataset2']} instances)")
    
    feature_info = results['feature_info']
    print(f"\nFeature Information:")
    print(f"  Total Features: {feature_info['n_features']}")
    print(f"  PCA Components: {feature_info['n_components']}")
    
    pca_info = results['pca_analysis']
    print(f"  Total Explained Variance: {sum(pca_info['explained_variance_ratio']):.3f}")
    
    wd_results = results['wasserstein_distances']
    print(f"\nWasserstein Distance Analysis:")
    print(f"  Total Distance: {wd_results['total_wasserstein_distance']:.6f}")
    print(f"  Weighted Distance: {wd_results['weighted_wasserstein_distance']:.6f}")
    print(f"  Average Distance: {wd_results['average_wasserstein_distance']:.6f}")
    
    print(f"\nPrincipal Component Distances:")
    for i in range(feature_info['n_components']):
        pc_name = f'PC{i+1}'
        distance = wd_results['component_distances'][pc_name]
        variance = pca_info['explained_variance_ratio'][i]
        print(f"  {pc_name}: {distance:.6f} (explains {variance:.3f} variance)")
    
    print("="*70)

if __name__ == "__main__":
    compare_scip_heuristics()
