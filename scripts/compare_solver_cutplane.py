"""
Solver Cutting Plane Usage Distribution Comparison Script

This script compares the cutting plane usage patterns between two datasets using PCA analysis.
It processes the data by:
1. Normalizing cutting plane usage per instance
2. Standardizing each feature (cutting plane type)
3. Applying PCA to reduce dimensionality
4. Visualizing and quantitatively comparing the distributions
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from omegaconf import DictConfig, OmegaConf

import hydra

# Set matplotlib font properties to support Unicode
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Verdana']
plt.rcParams['axes.unicode_minus'] = False


def load_data(csv_file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file

    Args:
        csv_file_path: Path to the CSV file

    Returns:
        Loaded DataFrame
    """
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"File not found: {csv_file_path}")
    
    df = pd.read_csv(csv_file_path)
    
    # Verify that cut plane columns exist
    cut_columns = [col for col in df.columns if col.startswith('cut_')]
    if not cut_columns:
        raise ValueError(f"No cutting plane columns found in {csv_file_path}")
        
    return df


def prepare_cut_data(df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Prepare cutting plane data for analysis

    Args:
        df1: First DataFrame
        df2: Second DataFrame

    Returns:
        Tuple of (processed_df1, processed_df2, cut_plane_types)
    """
    # Get all cutting plane columns from both dataframes
    cut_columns1 = [col for col in df1.columns if col.startswith('cut_')]
    cut_columns2 = [col for col in df2.columns if col.startswith('cut_')]
    all_cut_columns = sorted(list(set(cut_columns1 + cut_columns2)))
    
    logging.info(f"Found {len(all_cut_columns)} cutting plane types: {', '.join(all_cut_columns)}")
    
    # Ensure all cutting plane columns exist in both dataframes
    for col in all_cut_columns:
        if col not in df1.columns:
            df1[col] = np.nan
        if col not in df2.columns:
            df2[col] = np.nan
    
    # Extract just the instance names and cutting plane columns
    cut_df1 = df1[['instance'] + all_cut_columns].copy()
    cut_df2 = df2[['instance'] + all_cut_columns].copy()
    
    # Fill NaN values with 0
    for col in all_cut_columns:
        cut_df1[col] = cut_df1[col].fillna(0)
        cut_df2[col] = cut_df2[col].fillna(0)
    
    return cut_df1, cut_df2, all_cut_columns


def normalize_per_instance(df: pd.DataFrame, cut_columns: List[str]) -> pd.DataFrame:
    """
    Normalize cutting plane usage per instance (convert to proportions)

    Args:
        df: DataFrame with cutting plane data
        cut_columns: List of cutting plane column names

    Returns:
        DataFrame with normalized cutting plane data
    """
    # Create a copy to avoid modifying the original
    norm_df = df.copy()
    
    # Calculate total cutting planes per instance
    norm_df['total_cuts'] = norm_df[cut_columns].sum(axis=1)
    
    # Handle cases with zero total cuts
    zero_cuts = norm_df['total_cuts'] == 0
    if zero_cuts.any():
        logging.warning(f"Found {zero_cuts.sum()} instances with zero cutting planes. Setting their values to 0.")
    
    # Normalize each cut type by the total cuts for that instance
    for col in cut_columns:
        # For instances with non-zero cuts, normalize
        norm_df.loc[~zero_cuts, col] = norm_df.loc[~zero_cuts, col] / norm_df.loc[~zero_cuts, 'total_cuts']
        # For instances with zero cuts, set to 0 (already done because 0/0 would give NaN, and we're handling that)
    
    # Drop the total column as it's no longer needed
    norm_df.drop(columns=['total_cuts'], inplace=True)
    
    return norm_df


def apply_pca(df1: pd.DataFrame, df2: pd.DataFrame, cut_columns: List[str], 
              n_components: int = 3, scale_method: str = 'standard') -> Tuple[pd.DataFrame, pd.DataFrame, PCA, Any]:
    """
    Apply PCA to the combined cutting plane data

    Args:
        df1: First normalized DataFrame
        df2: Second normalized DataFrame
        cut_columns: List of cutting plane column names
        n_components: Number of PCA components to retain
        scale_method: Scaling method, either 'standard' or 'minmax'

    Returns:
        Tuple of (pca_df1, pca_df2, pca_model, scaler)
    """
    # Combine data for fitting PCA and scaler
    X1 = df1[cut_columns].values
    X2 = df2[cut_columns].values
    X_combined = np.vstack([X1, X2])
    
    # Choose the scaling method
    if scale_method == 'standard':
        scaler = StandardScaler()
        logging.info("Using Standard Scaling (Z-score normalization)")
    elif scale_method == 'minmax':
        scaler = MinMaxScaler()
        logging.info("Using Min-Max Scaling to [0, 1] range")
    else:
        raise ValueError(f"Unknown scaling method: {scale_method}")
    
    # Fit and transform the combined data
    X_scaled = scaler.fit_transform(X_combined)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Split back into the two original datasets
    X1_pca = X_pca[:len(X1)]
    X2_pca = X_pca[len(X1):]
    
    # Create DataFrames with PCA results
    pca_cols = [f'PC{i+1}' for i in range(n_components)]
    pca_df1 = pd.DataFrame(X1_pca, columns=pca_cols)
    pca_df2 = pd.DataFrame(X2_pca, columns=pca_cols)
    
    # Add instance names
    pca_df1['instance'] = df1['instance'].values
    pca_df2['instance'] = df2['instance'].values
    
    # Log variance explained
    variance_explained = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_explained)
    for i, (var, cum_var) in enumerate(zip(variance_explained, cumulative_variance)):
        logging.info(f"PC{i+1}: {var:.4f} variance explained ({cum_var:.4f} cumulative)")
    
    return pca_df1, pca_df2, pca, scaler


def visualize_pca_results(pca_df1: pd.DataFrame, pca_df2: pd.DataFrame, 
                         name1: str, name2: str, pca_model: PCA,
                         output_path: str, cut_columns: List[str],
                         title: str = "Cutting Plane Usage PCA Comparison",
                         figsize: Tuple[int, int] = (14, 10)) -> None:
    """
    Visualize PCA results with multiple plots

    Args:
        pca_df1: First DataFrame with PCA results
        pca_df2: Second DataFrame with PCA results
        name1: Name of the first dataset
        name2: Name of the second dataset
        pca_model: Fitted PCA model
        output_path: Path to save the visualization
        cut_columns: List of cutting plane column names
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Create a 2x2 subplot layout
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # 1. Scatter plot of PC1 vs PC2
    axs[0, 0].scatter(pca_df1['PC1'], pca_df1['PC2'], alpha=0.7, label=name1)
    axs[0, 0].scatter(pca_df2['PC1'], pca_df2['PC2'], alpha=0.7, label=name2)
    axs[0, 0].set_title('PC1 vs PC2')
    axs[0, 0].set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]:.2%} variance)')
    axs[0, 0].set_ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]:.2%} variance)')
    axs[0, 0].legend()
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    # 2. If we have PC3, add PC1 vs PC3 plot
    if 'PC3' in pca_df1.columns:
        axs[0, 1].scatter(pca_df1['PC1'], pca_df1['PC3'], alpha=0.7, label=name1)
        axs[0, 1].scatter(pca_df2['PC1'], pca_df2['PC3'], alpha=0.7, label=name2)
        axs[0, 1].set_title('PC1 vs PC3')
        axs[0, 1].set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]:.2%} variance)')
        axs[0, 1].set_ylabel(f'PC3 ({pca_model.explained_variance_ratio_[2]:.2%} variance)')
        axs[0, 1].legend()
        axs[0, 1].grid(True, linestyle='--', alpha=0.7)
    
    # 3. Distribution of PC1 scores
    sns.histplot(pca_df1['PC1'], kde=True, label=name1, ax=axs[1, 0], alpha=0.5)
    sns.histplot(pca_df2['PC1'], kde=True, label=name2, ax=axs[1, 0], alpha=0.5)
    axs[1, 0].set_title('PC1 Score Distribution')
    axs[1, 0].set_xlabel('PC1 Score')
    axs[1, 0].set_ylabel('Frequency')
    axs[1, 0].legend()
    
    # 4. PCA Loading Plot (shows contribution of each cut plane type to PC1 and PC2)
    loadings = pca_model.components_.T
    # Only plot the top loadings if there are many cut types
    max_cuts_to_plot = min(15, len(cut_columns))
    if len(cut_columns) > max_cuts_to_plot:
        # Find the cut types with the largest overall loading magnitude
        loading_magnitudes = np.sum(loadings[:, :2]**2, axis=1)
        top_indices = np.argsort(loading_magnitudes)[-max_cuts_to_plot:]
        plot_loadings = loadings[top_indices, :]
        plot_cut_types = [cut_columns[i] for i in top_indices]
    else:
        plot_loadings = loadings
        plot_cut_types = cut_columns
    
    for i, cut_type in enumerate(plot_cut_types):
        axs[1, 1].arrow(0, 0, plot_loadings[i, 0], plot_loadings[i, 1], 
                       head_width=0.05, head_length=0.05, fc='blue', ec='blue')
        axs[1, 1].text(plot_loadings[i, 0]*1.15, plot_loadings[i, 1]*1.15, 
                      cut_type.replace('cut_', ''), color='blue', ha='center', va='center')
    
    axs[1, 1].set_xlim(-1, 1)
    axs[1, 1].set_ylim(-1, 1)
    axs[1, 1].set_xlabel('PC1')
    axs[1, 1].set_ylabel('PC2')
    axs[1, 1].set_title('PCA Loadings (Cut Type Contributions)')
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)
    # Add a unit circle
    circle = plt.Circle((0, 0), 1, fill=False, edgecolor='gray', linestyle='--')
    axs[1, 1].add_patch(circle)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logging.info(f"PCA visualization saved to {output_path}")


def calculate_wasserstein_distances(pca_df1: pd.DataFrame, pca_df2: pd.DataFrame, name1: str, name2: str) -> Dict[str, float]:
    """
    Calculate Wasserstein distances between PC scores of the two datasets

    Args:
        pca_df1: First DataFrame with PCA results
        pca_df2: Second DataFrame with PCA results
        name1: Name of the first dataset
        name2: Name of the second dataset

    Returns:
        Dictionary of Wasserstein distances for each PC
    """
    distances = {}
    
    for pc in [col for col in pca_df1.columns if col.startswith('PC')]:
        dist = stats.wasserstein_distance(pca_df1[pc], pca_df2[pc])
        distances[pc] = dist
        logging.info(f"Wasserstein distance for {pc} between {name1} and {name2}: {dist:.4f}")
    
    return distances


def save_pca_results(pca_df1: pd.DataFrame, pca_df2: pd.DataFrame, 
                    name1: str, name2: str, 
                    wasserstein_distances: Dict[str, float],
                    output_path: str) -> None:
    """
    Save PCA analysis results to CSV

    Args:
        pca_df1: First DataFrame with PCA results
        pca_df2: Second DataFrame with PCA results
        name1: Name of the first dataset
        name2: Name of the second dataset
        wasserstein_distances: Dictionary of Wasserstein distances
        output_path: Path to save results
    """
    # Add dataset column to identify source
    pca_df1['dataset'] = name1
    pca_df2['dataset'] = name2
    
    # Combine results
    combined_df = pd.concat([pca_df1, pca_df2], ignore_index=True)
    
    # Save to CSV
    combined_df.to_csv(output_path, index=False)
    
    # Also save the Wasserstein distances
    distances_df = pd.DataFrame(list(wasserstein_distances.items()), columns=['Principal_Component', 'Wasserstein_Distance'])
    distances_path = output_path.replace('.csv', '_distances.csv')
    distances_df.to_csv(distances_path, index=False)
    
    logging.info(f"PCA results saved to {output_path}")
    logging.info(f"Wasserstein distances saved to {distances_path}")


@hydra.main(config_path="../configs", config_name="compare_cutplane")
def compare_cutplanes(cfg: DictConfig) -> None:
    """
    Compare cutting plane usage patterns between two datasets using PCA

    Args:
        cfg: Hydra configuration object
    """
    # Set up logging
    log_level = getattr(logging, cfg.logging.level)
    logging.basicConfig(
        level=log_level,
        format=cfg.logging.format
    )
    
    logging.info("Initializing cutting plane usage pattern comparison...")
    
    # Ensure output directory exists
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    
    try:
        # Load data
        logging.info(f"Loading dataset 1: {cfg.paths.csv_file1}")
        df1 = load_data(cfg.paths.csv_file1)
        
        logging.info(f"Loading dataset 2: {cfg.paths.csv_file2}")
        df2 = load_data(cfg.paths.csv_file2)
        
        # Prepare cutting plane data
        logging.info("Extracting and preparing cutting plane data...")
        cut_df1, cut_df2, cut_columns = prepare_cut_data(df1, df2)
        
        # Normalize per instance
        logging.info("Normalizing cutting plane usage per instance...")
        norm_df1 = normalize_per_instance(cut_df1, cut_columns)
        norm_df2 = normalize_per_instance(cut_df2, cut_columns)
        
        # Apply PCA
        logging.info(f"Applying PCA with {cfg.pca.n_components} components using {cfg.pca.scaling} scaling...")
        pca_df1, pca_df2, pca_model, scaler = apply_pca(
            norm_df1, norm_df2, cut_columns, 
            n_components=cfg.pca.n_components,
            scale_method=cfg.pca.scaling
        )
        
        # Visualize results
        if cfg.visualization.enabled:
            logging.info("Generating PCA visualization...")
            vis_path = os.path.join(cfg.paths.output_dir, cfg.visualization.filename)
            visualize_pca_results(
                pca_df1, pca_df2,
                cfg.comparison.name1, cfg.comparison.name2,
                pca_model, vis_path, cut_columns,
                title=cfg.visualization.title,
                figsize=cfg.visualization.figure_size
            )
        
        # Calculate Wasserstein distances
        logging.info("Calculating Wasserstein distances between PC distributions...")
        wasserstein_distances = calculate_wasserstein_distances(
            pca_df1, pca_df2,
            cfg.comparison.name1, cfg.comparison.name2
        )
        
        # Save results
        if cfg.output.save_csv:
            logging.info("Saving PCA analysis results...")
            results_path = os.path.join(cfg.paths.output_dir, cfg.output.result_filename)
            save_pca_results(
                pca_df1, pca_df2,
                cfg.comparison.name1, cfg.comparison.name2,
                wasserstein_distances,
                results_path
            )
        
        logging.info("Comparison completed!")
        
        # Summary of key findings
        logging.info("\n---------- Key Findings ----------")
        logging.info(f"Total cutting plane types analyzed: {len(cut_columns)}")
        top_pc = sorted(wasserstein_distances.items(), key=lambda x: x[1], reverse=True)[0]
        logging.info(f"Largest distribution difference in {top_pc[0]}: {top_pc[1]:.4f}")
        total_variance = sum(pca_model.explained_variance_ratio_[:cfg.pca.n_components])
        logging.info(f"Total variance explained by {cfg.pca.n_components} components: {total_variance:.2%}")
        
        # Interpretation based on Wasserstein distance
        pc1_dist = wasserstein_distances.get('PC1', 0)
        if pc1_dist < 0.2:
            logging.info("The cutting plane usage patterns are very similar between the two datasets.")
        elif pc1_dist < 0.5:
            logging.info("The cutting plane usage patterns show moderate differences between the two datasets.")
        else:
            logging.info("The cutting plane usage patterns are substantially different between the two datasets.")
        
    except Exception as e:
        logging.error(f"Error during comparison: {str(e)}")
        raise


if __name__ == "__main__":
    compare_cutplanes()
