"""
HiGHS solver cutting planes comparison script.

This script compares the cutting plane usage patterns between two HiGHS solver result datasets
using PCA dimensionality reduction and Wasserstein distance on principal components.
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

@hydra.main(config_path="../configs", config_name="highs_compare_cutplane", version_base=None)
def compare_highs_cutplanes(cfg: DictConfig):
    """
    Compare cutting plane usage patterns using PCA between two HiGHS solver result datasets.
    
    Args:
        cfg: Hydra configuration object
    """
    # Set up logging
    log_level = getattr(logging, cfg.logging.level)
    logging.basicConfig(
        level=log_level,
        format=cfg.logging.format
    )
    
    logging.info("Starting HiGHS cutting planes comparison analysis...")
    
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
    
    # Prepare cutting plane features for PCA analysis
    try:
        features1, features2, feature_names = prepare_cutplane_features(df1, df2, cfg)
        logging.info(f"Prepared features with {len(feature_names)} cutting plane types")
        
    except Exception as e:
        logging.error(f"Error preparing features: {str(e)}")
        return
    
    # Perform PCA analysis
    try:
        pca_results = perform_pca_analysis(features1, features2, feature_names, cfg)
        logging.info("PCA analysis completed successfully")
        
    except Exception as e:
        logging.error(f"Error in PCA analysis: {str(e)}")
        return
    
    # Calculate Wasserstein distances on PCA components
    try:
        results = calculate_pca_wasserstein_distances(pca_results, cfg)
        logging.info("Wasserstein distance calculation completed")
        
    except Exception as e:
        logging.error(f"Error calculating Wasserstein distances: {str(e)}")
        return
    
    # Create comparison plots
    try:
        plot_path = os.path.join(output_dir, "highs_cutplane_comparison.png")
        create_pca_comparison_plots(pca_results, cfg, plot_path, results)
        logging.info(f"Comparison plots saved to {plot_path}")
        
    except Exception as e:
        logging.error(f"Error creating plots: {str(e)}")
    
    # Save results
    try:
        results_path = os.path.join(output_dir, "highs_cutplane_comparison_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Results saved to {results_path}")
        
    except Exception as e:
        logging.error(f"Error saving results: {str(e)}")
    
    # Print summary
    print_pca_summary(results)
    
    logging.info("HiGHS cutting planes comparison analysis completed!")

def prepare_cutplane_features(df1: pd.DataFrame, df2: pd.DataFrame, cfg: DictConfig) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Prepare cutting plane features for PCA analysis."""
    
    # Identify cutting plane columns
    cut_columns = [col for col in df1.columns if col.startswith('cut_')]
    
    if not cut_columns:
        raise ValueError("No cutting plane columns found in the datasets")
    
    # Ensure both datasets have the same cutting plane columns
    cut_columns_df2 = [col for col in df2.columns if col.startswith('cut_')]
    common_cut_columns = list(set(cut_columns) & set(cut_columns_df2))
    
    if not common_cut_columns:
        raise ValueError("No common cutting plane columns found between datasets")
    
    logging.info(f"Using {len(common_cut_columns)} common cutting plane features")
    
    # Extract features
    features1 = df1[common_cut_columns].fillna(0).values
    features2 = df2[common_cut_columns].fillna(0).values
    
    # Remove instances where all cuts are zero (if any)
    non_zero_mask1 = np.any(features1 > 0, axis=1)
    non_zero_mask2 = np.any(features2 > 0, axis=1)
    
    if np.sum(non_zero_mask1) == 0 or np.sum(non_zero_mask2) == 0:
        logging.warning("One or both datasets have no instances with cutting plane usage")
        # Keep all instances even if they have zero cuts
        features1_filtered = features1
        features2_filtered = features2
    else:
        features1_filtered = features1[non_zero_mask1]
        features2_filtered = features2[non_zero_mask2]
        logging.info(f"Filtered to {features1_filtered.shape[0]} and {features2_filtered.shape[0]} instances with cutting plane usage")
    
    return features1_filtered, features2_filtered, common_cut_columns

def perform_pca_analysis(features1: np.ndarray, features2: np.ndarray, 
                        feature_names: List[str], cfg: DictConfig) -> Dict[str, Any]:
    """Perform PCA analysis on cutting plane features."""
    
    # Combine features for fitting PCA
    combined_features = np.vstack([features1, features2])
    
    # Standardize features
    scaler = StandardScaler()
    combined_features_scaled = scaler.fit_transform(combined_features)
    
    # Apply PCA
    n_components = min(cfg.pca.n_components, combined_features_scaled.shape[1], combined_features_scaled.shape[0])
    pca = PCA(n_components=n_components)
    combined_pca = pca.fit_transform(combined_features_scaled)
    
    # Split back into two datasets
    n1 = features1.shape[0]
    pca1 = combined_pca[:n1]
    pca2 = combined_pca[n1:]
    
    # Transform individual datasets
    features1_scaled = scaler.transform(features1)
    features2_scaled = scaler.transform(features2)
    
    return {
        'pca': pca,
        'scaler': scaler,
        'pca1': pca1,
        'pca2': pca2,
        'features1_scaled': features1_scaled,
        'features2_scaled': features2_scaled,
        'feature_names': feature_names,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'components': pca.components_
    }

def calculate_pca_wasserstein_distances(pca_results: Dict[str, Any], cfg: DictConfig) -> Dict[str, Any]:
    """Calculate Wasserstein distances on PCA components."""
    
    pca1 = pca_results['pca1']
    pca2 = pca_results['pca2']
    explained_variance = pca_results['explained_variance_ratio']
    
    results = {
        'dataset1_name': cfg.comparison.name1,
        'dataset2_name': cfg.comparison.name2,
        'n_components': pca1.shape[1],
        'explained_variance_ratio': explained_variance.tolist(),
        'total_explained_variance': np.sum(explained_variance),
        'pca_wasserstein_distances': [],
        'component_statistics': []
    }
    
    # Calculate Wasserstein distance for each PCA component
    for i in range(pca1.shape[1]):
        pc1 = pca1[:, i]
        pc2 = pca2[:, i]
        
        # Calculate Wasserstein distance
        wd = wasserstein_distance(pc1, pc2)
        
        # Calculate basic statistics
        stats1 = {
            'mean': float(np.mean(pc1)),
            'std': float(np.std(pc1)),
            'min': float(np.min(pc1)),
            'max': float(np.max(pc1))
        }
        
        stats2 = {
            'mean': float(np.mean(pc2)),
            'std': float(np.std(pc2)),
            'min': float(np.min(pc2)),
            'max': float(np.max(pc2))
        }
        
        results['pca_wasserstein_distances'].append({
            'component': i + 1,
            'wasserstein_distance': float(wd),
            'explained_variance': float(explained_variance[i])
        })
        
        results['component_statistics'].append({
            'component': i + 1,
            'dataset1_stats': stats1,
            'dataset2_stats': stats2
        })
    
    # Calculate weighted average Wasserstein distance
    weighted_wd = np.sum([
        wd['wasserstein_distance'] * wd['explained_variance'] 
        for wd in results['pca_wasserstein_distances']
    ])
    
    results['weighted_average_wasserstein_distance'] = float(weighted_wd)
    
    return results

def create_pca_comparison_plots(pca_results: Dict[str, Any], cfg: DictConfig, 
                              output_path: str, results: Dict[str, Any]):
    """Create comprehensive PCA comparison plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'HiGHS Cutting Plane PCA Comparison: {cfg.comparison.name1} vs {cfg.comparison.name2}', 
                 fontsize=16)
    
    pca1 = pca_results['pca1']
    pca2 = pca_results['pca2']
    explained_variance = pca_results['explained_variance_ratio']
    
    # Plot 1: PCA scatter plot (first two components)
    ax1 = axes[0, 0]
    if pca1.shape[1] >= 2:
        ax1.scatter(pca1[:, 0], pca1[:, 1], alpha=0.6, label=cfg.comparison.name1, s=30)
        ax1.scatter(pca2[:, 0], pca2[:, 1], alpha=0.6, label=cfg.comparison.name2, s=30)
        ax1.set_xlabel(f'PC1 ({explained_variance[0]:.1%} variance)')
        ax1.set_ylabel(f'PC2 ({explained_variance[1]:.1%} variance)')
    else:
        ax1.hist(pca1[:, 0], alpha=0.6, label=cfg.comparison.name1, bins=20)
        ax1.hist(pca2[:, 0], alpha=0.6, label=cfg.comparison.name2, bins=20)
        ax1.set_xlabel(f'PC1 ({explained_variance[0]:.1%} variance)')
        ax1.set_ylabel('Frequency')
    ax1.set_title('PCA Scatter Plot')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Explained variance ratio
    ax2 = axes[0, 1]
    components = range(1, len(explained_variance) + 1)
    ax2.bar(components, explained_variance, alpha=0.7)
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Explained Variance Ratio')
    ax2.set_title('Explained Variance by Component')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Wasserstein distances by component
    ax3 = axes[1, 0]
    wd_values = [wd['wasserstein_distance'] for wd in results['pca_wasserstein_distances']]
    ax3.bar(components, wd_values, alpha=0.7, color='orange')
    ax3.set_xlabel('Principal Component')
    ax3.set_ylabel('Wasserstein Distance')
    ax3.set_title('Wasserstein Distance by Component')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Component loadings heatmap
    ax4 = axes[1, 1]
    components_matrix = pca_results['components'][:len(components), :]
    feature_names = pca_results['feature_names']
    
    # Create heatmap
    im = ax4.imshow(components_matrix, cmap='RdBu_r', aspect='auto')
    ax4.set_xticks(range(len(feature_names)))
    ax4.set_xticklabels([name.replace('cut_', '') for name in feature_names], rotation=45, ha='right')
    ax4.set_yticks(range(len(components)))
    ax4.set_yticklabels([f'PC{i+1}' for i in range(len(components))])
    ax4.set_title('PCA Component Loadings')
    
    # Add colorbar
    plt.colorbar(im, ax=ax4, shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create additional detailed plots
    create_detailed_component_plots(pca_results, cfg, os.path.dirname(output_path), results)

def create_detailed_component_plots(pca_results: Dict[str, Any], cfg: DictConfig, 
                                  output_dir: str, results: Dict[str, Any]):
    """Create detailed plots for each PCA component."""
    
    pca1 = pca_results['pca1']
    pca2 = pca_results['pca2']
    
    for i in range(min(3, pca1.shape[1])):  # Plot first 3 components
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        pc1 = pca1[:, i]
        pc2 = pca2[:, i]
        
        # Histogram comparison
        ax1.hist(pc1, alpha=0.6, label=cfg.comparison.name1, bins=20, density=True)
        ax1.hist(pc2, alpha=0.6, label=cfg.comparison.name2, bins=20, density=True)
        ax1.set_xlabel(f'PC{i+1} Value')
        ax1.set_ylabel('Density')
        ax1.set_title(f'PC{i+1} Distribution Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot comparison
        data_to_plot = [pc1, pc2]
        labels = [cfg.comparison.name1, cfg.comparison.name2]
        ax2.boxplot(data_to_plot, labels=labels)
        ax2.set_ylabel(f'PC{i+1} Value')
        ax2.set_title(f'PC{i+1} Box Plot Comparison')
        ax2.grid(True, alpha=0.3)
        
        # Add Wasserstein distance as text
        wd = results['pca_wasserstein_distances'][i]['wasserstein_distance']
        fig.suptitle(f'PC{i+1} Analysis (Wasserstein Distance: {wd:.4f})', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'highs_pc{i+1}_detailed.png'), dpi=300, bbox_inches='tight')
        plt.close()

def print_pca_summary(results: Dict[str, Any]):
    """Print a summary of the PCA comparison results."""
    
    print("\n" + "="*60)
    print("HiGHS CUTTING PLANE PCA COMPARISON SUMMARY")
    print("="*60)
    
    print(f"Dataset 1: {results['dataset1_name']}")
    print(f"Dataset 2: {results['dataset2_name']}")
    print(f"Number of PCA components: {results['n_components']}")
    print(f"Total explained variance: {results['total_explained_variance']:.1%}")
    
    print(f"\nWeighted average Wasserstein distance: {results['weighted_average_wasserstein_distance']:.4f}")
    
    print(f"\nPer-component analysis:")
    for wd_info in results['pca_wasserstein_distances']:
        comp = wd_info['component']
        wd = wd_info['wasserstein_distance']
        var = wd_info['explained_variance']
        print(f"  PC{comp}: WD={wd:.4f}, Explained Variance={var:.1%}")
    
    # Interpretation
    avg_wd = results['weighted_average_wasserstein_distance']
    if avg_wd < 0.1:
        interpretation = "Very similar cutting plane usage patterns"
    elif avg_wd < 0.3:
        interpretation = "Moderately similar cutting plane usage patterns"
    elif avg_wd < 0.5:
        interpretation = "Somewhat different cutting plane usage patterns"
    else:
        interpretation = "Very different cutting plane usage patterns"
    
    print(f"\nInterpretation: {interpretation}")
    print("="*60)

if __name__ == "__main__":
    compare_highs_cutplanes()
