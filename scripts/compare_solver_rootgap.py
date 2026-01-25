"""
Solver Root Gap Distribution Comparison Script

This script compares the root gap distributions of two different datasets, calculates the Wasserstein
distance between them, and generates visualization charts to show the differences between distributions.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Tuple, Dict, List, Optional, Any
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime

# Add project root directory to system path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load CSV file data

    Args:
        file_path: CSV file path

    Returns:
        DataFrame containing data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    if 'root_gap' not in df.columns:
        raise ValueError(f"File {file_path} does not contain 'root_gap' column")
    
    return df


def prepare_data(df1: pd.DataFrame, df2: pd.DataFrame, 
                 sampling: str = "all", sample_size: int = 100, 
                 seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for comparison

    Args:
        df1: First dataset DataFrame
        df2: Second dataset DataFrame
        sampling: Sampling method, can be "all" or "sample"
        sample_size: Sample size
        seed: Random seed

    Returns:
        (data1, data2): Prepared root_gap values from two datasets
    """
    # Filter valid root_gap values
    data1 = df1['root_gap'].dropna().values
    data2 = df2['root_gap'].dropna().values
    
    # If sampling is enabled
    if sampling == "sample":
        np.random.seed(seed)
        if len(data1) > sample_size:
            data1 = np.random.choice(data1, size=sample_size, replace=False)
        if len(data2) > sample_size:
            data2 = np.random.choice(data2, size=sample_size, replace=False)
    
    return data1, data2


def calculate_wasserstein_distance(data1: np.ndarray, data2: np.ndarray) -> float:
    """
    Calculate the Wasserstein distance between two distributions

    Args:
        data1: First dataset
        data2: Second dataset

    Returns:
        Wasserstein distance
    """
    return stats.wasserstein_distance(data1, data2)


def calculate_statistics(data: np.ndarray) -> Dict[str, float]:
    """
    Calculate statistical information for the data

    Args:
        data: Input data

    Returns:
        Dictionary containing statistical information
    """
    return {
        "count": len(data),
        "mean": np.mean(data),
        "std": np.std(data),
        "min": np.min(data),
        "25%": np.percentile(data, 25),
        "50%": np.median(data),
        "75%": np.percentile(data, 75),
        "max": np.max(data)
    }


def visualize_comparison(data1: np.ndarray, data2: np.ndarray, 
                        name1: str, name2: str, 
                        output_path: str,
                        wasserstein_dist: float,
                        title: str = "Root Gap Distribution Comparison",
                        bins: int = 30,
                        figsize: Tuple[int, int] = (12, 8),
                        style: str = "seaborn-v0_8-darkgrid") -> None:
    """
    Visualize the comparison between two distributions

    Args:
        data1: First dataset
        data2: Second dataset
        name1: Name of the first dataset
        name2: Name of the second dataset
        output_path: Path to save the chart
        wasserstein_dist: Calculated Wasserstein distance
        title: Chart title
        bins: Number of bins for histogram
        figsize: Chart dimensions
        style: Chart style
    """
    plt.style.use(style)
    
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f"{title}\nWasserstein Distance: {wasserstein_dist:.4f}", fontsize=16)
    
    # Histogram comparison
    axs[0, 0].hist(data1, bins=bins, alpha=0.5, label=name1)
    axs[0, 0].hist(data2, bins=bins, alpha=0.5, label=name2)
    axs[0, 0].set_title('Histogram Comparison')
    axs[0, 0].set_xlabel('Root Gap (%)')
    axs[0, 0].set_ylabel('Frequency')
    axs[0, 0].legend()
    
    # Kernel density estimation comparison
    sns.kdeplot(data1, label=name1, ax=axs[0, 1])
    sns.kdeplot(data2, label=name2, ax=axs[0, 1])
    axs[0, 1].set_title('Kernel Density Estimation')
    axs[0, 1].set_xlabel('Root Gap (%)')
    axs[0, 1].set_ylabel('Density')
    axs[0, 1].legend()
    
    # Cumulative distribution function comparison
    axs[1, 0].hist(data1, bins=bins, alpha=0.5, density=True, cumulative=True, label=name1)
    axs[1, 0].hist(data2, bins=bins, alpha=0.5, density=True, cumulative=True, label=name2)
    axs[1, 0].set_title('Cumulative Distribution Function')
    axs[1, 0].set_xlabel('Root Gap (%)')
    axs[1, 0].set_ylabel('Cumulative Probability')
    axs[1, 0].legend()
    
    # Box plot comparison
    box_data = [data1, data2]
    axs[1, 1].boxplot(box_data, labels=[name1, name2])
    axs[1, 1].set_title('Box Plot Comparison')
    axs[1, 1].set_ylabel('Root Gap (%)')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path)
    plt.close()
    
    logging.info(f"Comparison chart saved to {output_path}")


def save_results(stats1: Dict[str, float], 
                stats2: Dict[str, float], 
                wasserstein_dist: float,
                name1: str,
                name2: str,
                output_path: str) -> None:
    """
    Save comparison results to a CSV file

    Args:
        stats1: Statistical information for the first dataset
        stats2: Statistical information for the second dataset
        wasserstein_dist: Calculated Wasserstein distance
        name1: Name of the first dataset
        name2: Name of the second dataset
        output_path: Output file path
    """
    # Create results DataFrame
    results = pd.DataFrame({
        "Statistic": list(stats1.keys()) + ["Wasserstein_Distance"],
        name1: list(stats1.values()) + [wasserstein_dist],
        name2: list(stats2.values()) + [wasserstein_dist]
    })
    
    # Save to CSV
    results.to_csv(output_path, index=False)
    logging.info(f"Comparison results saved to {output_path}")


@hydra.main(config_path="../configs", config_name="compare_rootgap")
def compare_rootgap(cfg: DictConfig) -> None:
    """
    Compare the root_gap distributions of two datasets

    Args:
        cfg: Hydra configuration object
    """
    # Set up logging
    log_level = getattr(logging, cfg.logging.level)
    logging.basicConfig(
        level=log_level,
        format=cfg.logging.format
    )
    
    logging.info("Initializing root_gap distribution comparison...")
    
    # Ensure output directory exists
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    
    try:
        # Load data
        logging.info(f"Loading dataset 1: {cfg.paths.csv_file1}")
        df1 = load_data(cfg.paths.csv_file1)
        
        logging.info(f"Loading dataset 2: {cfg.paths.csv_file2}")
        df2 = load_data(cfg.paths.csv_file2)
        
        # Prepare data
        logging.info("Preparing data for comparison...")
        data1, data2 = prepare_data(
            df1, df2, 
            sampling=cfg.comparison.sampling,
            sample_size=cfg.comparison.sample_size,
            seed=cfg.seed
        )
        
        # Calculate statistics
        stats1 = calculate_statistics(data1)
        stats2 = calculate_statistics(data2)
        
        # Print statistics
        if cfg.output.verbose:
            logging.info(f"{cfg.comparison.name1} statistics:")
            for key, value in stats1.items():
                logging.info(f"  {key}: {value}")
            
            logging.info(f"{cfg.comparison.name2} statistics:")
            for key, value in stats2.items():
                logging.info(f"  {key}: {value}")
        
        # Calculate Wasserstein distance
        logging.info("Calculating Wasserstein distance...")
        wasserstein_dist = calculate_wasserstein_distance(data1, data2)
        logging.info(f"Wasserstein distance: {wasserstein_dist:.4f}")
        
        # Visualize comparison
        if cfg.visualization.enabled:
            logging.info("Generating visualization comparison charts...")
            vis_path = os.path.join(cfg.paths.output_dir, cfg.visualization.filename)
            visualize_comparison(
                data1, data2,
                cfg.comparison.name1, cfg.comparison.name2,
                vis_path,
                wasserstein_dist,
                title=cfg.visualization.title,
                bins=cfg.visualization.bins,
                figsize=cfg.visualization.figure_size,
                style=cfg.visualization.style
            )
        
        # Save results
        if cfg.output.save_csv:
            logging.info("Saving comparison results...")
            results_path = os.path.join(cfg.paths.output_dir, cfg.output.result_filename)
            save_results(
                stats1, stats2, 
                wasserstein_dist,
                cfg.comparison.name1, cfg.comparison.name2,
                results_path
            )
        
        logging.info("Comparison completed!")
        
        # Summary of key findings
        logging.info("\n---------- Key Findings ----------")
        mean_diff = abs(stats1["mean"] - stats2["mean"])
        std_diff = abs(stats1["std"] - stats2["std"])
        median_diff = abs(stats1["50%"] - stats2["50%"])
        
        logging.info(f"Mean difference: {mean_diff:.2f}%")
        logging.info(f"Standard deviation difference: {std_diff:.2f}%")
        logging.info(f"Median difference: {median_diff:.2f}%")
        logging.info(f"Wasserstein distance: {wasserstein_dist:.4f}")
        
        if wasserstein_dist < 0.5:
            logging.info("The two distributions are very similar.")
        elif wasserstein_dist < 1.0:
            logging.info("The two distributions have some differences, but overall structure is similar.")
        elif wasserstein_dist < 2.0:
            logging.info("The two distributions have significant differences.")
        else:
            logging.info("The two distributions are substantially different.")
        
    except Exception as e:
        logging.error(f"Error during comparison: {str(e)}")
        raise


if __name__ == "__main__":
    compare_rootgap()
