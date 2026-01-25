import json
import logging
import multiprocessing as mp
import os
import os.path as path
import sys
from functools import partial
from typing import Union, Dict, List, Tuple, Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from omegaconf import DictConfig, OmegaConf
from scipy.stats import entropy
from tqdm import tqdm

# Add project root directory to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common import instance2graph

# Feature definition, used for similarity comparison
FEATURES = {
    "coef_dens": float,         # Coefficient density
    "var_degree_mean": float,   # Variable node degree mean
    "var_degree_std": float,    # Variable node degree standard deviation
    "cons_degree_mean": float,  # Constraint node degree mean
    "cons_degree_std": float,   # Constraint node degree standard deviation
    "lhs_mean": float,          # Left-hand side coefficient mean
    "lhs_std": float,           # Left-hand side coefficient standard deviation
    "rhs_mean": float,          # Right-hand side coefficient mean
    "rhs_std": float,           # Right-hand side coefficient standard deviation
    "clustering": float,        # Clustering coefficient
    "modularity": float,        # Modularity
}


def js_div(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Calculate the Jensen-Shannon divergence between two distributions
    
    Args:
        x1: Samples from the first distribution
        x2: Samples from the second distribution
        
    Returns:
        float: Jensen-Shannon divergence value
    """
    # Merge samples from both distributions
    x = np.hstack([x1, x2])
    
    # If standard deviation is too small, consider the distributions the same
    if x.std() < 1e-10:
        return 0.0
        
    # Calculate the histogram of the combined distribution
    M, bins = np.histogram(x, bins=5, density=True)
    
    # Calculate the histograms of the two distributions
    P, _ = np.histogram(x1, bins=bins)
    Q, _ = np.histogram(x2, bins=bins)
    
    # Calculate JS divergence
    return (entropy(P, M) + entropy(Q, M)) / 2


def compute_jsdiv(features1: List[np.ndarray], features2: List[np.ndarray], num_samples: int = 1000) -> Tuple[float, Dict[str, float]]:
    """
    Calculate the similarity between two sets of instance features
    
    Args:
        features1: First set of instance feature vectors
        features2: Second set of instance feature vectors
        num_samples: Number of samples for random sampling
        
    Returns:
        Tuple[float, Dict[str, float]]: Overall similarity score and feature similarity scores
    """
    # Ensure features1 and features2 are 2D arrays
    features1_array = np.vstack(features1)
    features2_array = np.vstack(features2)
    
    # Check if the number of features matches the expected FEATURES
    if features1_array.shape[1] != len(FEATURES) or features2_array.shape[1] != len(FEATURES):
        logging.warning(f"Feature count mismatch: features1 has {features1_array.shape[1]} features, features2 has {features2_array.shape[1]} features, expected {len(FEATURES)} features")
    
    # Random sampling (using replace=True allows duplicate sampling)
    sample_indices1 = np.random.choice(list(range(len(features1_array))), 
                                     min(num_samples, len(features1_array)), 
                                     replace=True)
    sample_indices2 = np.random.choice(list(range(len(features2_array))), 
                                     min(num_samples, len(features2_array)), 
                                     replace=True)
    
    f1 = features1_array[sample_indices1]
    f2 = features2_array[sample_indices2]
    
    # Calculate feature similarity
    meta_results = {}
    for i in range(min(f1.shape[1], f2.shape[1], len(FEATURES))):
        feature_name = list(FEATURES.keys())[i]
        try:
            jsdiv = js_div(f1[:, i], f2[:, i])
            # Convert JS divergence to similarity score 
            meta_results[feature_name] = round(1 - jsdiv / np.log(2), 3)
        except Exception as e:
            logging.warning(f"Error computing similarity for feature {feature_name}: {str(e)}")
            meta_results[feature_name] = 0.0
    
    # Calculate overall similarity score (average of all feature similarities)
    score = sum(meta_results.values()) / len(meta_results)
    return score, meta_results


def extract_feature_vector(feature_dict: Dict) -> np.ndarray:
    """
    Extract feature vector from feature dictionary defined in FEATURES
    
    Args:
        feature_dict: Dictionary containing instance features
        
    Returns:
        np.ndarray: Feature vector
    """
    # Check if feature dictionary contains all features defined in FEATURES
    if not all(key in feature_dict for key in FEATURES.keys()):
        logging.warning(f"Feature dictionary missing some features: {set(FEATURES.keys()) - set(feature_dict.keys())}")
    
    # Extract feature values in the order defined in FEATURES
    feature_vector = []
    for key in FEATURES.keys():
        if key in feature_dict:
            feature_vector.append(feature_dict[key])
        else:
            # If a feature is missing, set it to 0
            feature_vector.append(0.0)
            
    return np.array(feature_vector)


def compute_features(samples_dir: str, num_workers: int = 1) -> List[np.ndarray]:
    """
    Compute features for all instances in a directory
    
    Args:
        samples_dir: Directory containing instance files
        num_workers: Number of parallel processes for computation
        
    Returns:
        List[np.ndarray]: List of feature vectors for all instances
    """
    # Get all files in the directory
    samples = os.listdir(samples_dir)
    if not samples:
        logging.error(f"Directory {samples_dir} is empty")
        return []
    
    # Parallel compute instance features
    func = partial(compute_features_, data_dir=samples_dir)
    with mp.Pool(num_workers) as pool:
        feature_dicts = list(tqdm(pool.imap(func, samples), total=len(
            samples), desc="Computing instance features"))
    
    # Convert feature dictionaries to feature vectors
    features = [extract_feature_vector(feat_dict) for feat_dict in feature_dicts if feat_dict is not None]
    
    if not features:
        logging.error(f"Unable to extract features from instances in directory {samples_dir}")
        return []
    
    return features


def compute_features_(file: str, data_dir: str) -> Dict:
    """
    Compute features for a single instance
    
    Args:
        file: Instance file name
        data_dir: Directory containing instance files
        
    Returns:
        Dict: Feature dictionary for the instance
    """
    try:
        sample_path = path.join(data_dir, file)
        _, features = instance2graph(sample_path, compute_features=True)
        return features
    except Exception as e:
        logging.error(f"Error processing file {file}: {str(e)}")
        return None


def compare_instance_sets(cfg: DictConfig) -> Dict[str, Any]:
    """
    Compare the structural similarity of two sets of instances
    
    Args:
        cfg: Hydra configuration object
        
    Returns:
        Dict[str, Any]: Dictionary containing similarity results
    """
    set1_dir = cfg.paths.set1_dir
    set2_dir = cfg.paths.set2_dir
    num_workers = cfg.compute.num_workers
    num_samples = cfg.compute.num_samples
    
    # If num_workers is set to null, use all available CPUs
    if num_workers is None:
        num_workers = mp.cpu_count()
        
    logging.info(f"Calculating features for set1: {set1_dir}")
    features1 = compute_features(set1_dir, num_workers)
    if not features1:
        logging.error(f"Unable to extract features from directory {set1_dir}")
        return None
    
    logging.info(f"Calculating features for set2: {set2_dir}")
    features2 = compute_features(set2_dir, num_workers)
    if not features2:
        logging.error(f"Unable to extract features from directory {set2_dir}")
        return None
    
    logging.info("Calculating similarity...")
    score, feature_scores = compute_jsdiv(features1, features2, num_samples)
    
    results = {
        "overall_similarity": score,
        "feature_similarity": feature_scores,
        "set1_count": len(os.listdir(set1_dir)),
        "set2_count": len(os.listdir(set2_dir)),
        "set1_dir": set1_dir,
        "set2_dir": set2_dir
    }
    
    return results


def plot_similarity_heatmap(results: Dict[str, Any], output_path: str) -> None:
    """
    Plot similarity heatmap of features
    
    Args:
        results: Dictionary containing similarity results
        output_path: Output file path
    """
    # Extract feature similarity data
    feature_scores = results["feature_similarity"]
    features = list(feature_scores.keys())
    scores = list(feature_scores.values())
    
    # Create heatmap data
    data = np.array([scores])
    
    # Plot heatmap
    plt.figure(figsize=(12, 4))
    ax = sns.heatmap(data, annot=True, fmt=".3f", cmap="YlGnBu", 
                   xticklabels=features, yticklabels=["Similarity"],
                   vmin=0, vmax=1, cbar_kws={"label": "Similarity Score"})
    
    # Set title and labels
    set1_name = os.path.basename(results["set1_dir"])
    set2_name = os.path.basename(results["set2_dir"])
    plt.title(f"{set1_name} with {set2_name} feature similarity (Overall: {results['overall_similarity']:.3f})")
    plt.xlabel("Feature")
    plt.xticks(rotation=45, ha="right")
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


@hydra.main(version_base=None, config_path="../configs", config_name="instance_similarity")
def main(cfg: DictConfig):
    """
    Compare the structural similarity of two sets of MILP instances using a configuration file
    
    Args:
        cfg: Hydra configuration object
    """
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level),
        format=cfg.logging.format
    )
    
    # Print configuration
    logging.info(f"=== Current configuration ===\n{OmegaConf.to_yaml(cfg)}")
    
    # Ensure output directory exists
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    
    # Calculate similarity
    results = compare_instance_sets(cfg)
    if results is None:
        return
    
    # Print summary
    logging.info("\n=== Result Summary ===")
    logging.info(f"Set1: {results['set1_dir']} (Instance count: {results['set1_count']})")
    logging.info(f"Set2: {results['set2_dir']} (Instance count: {results['set2_count']})")
    logging.info(f"Overall similarity score: {results['overall_similarity']:.4f}")
    logging.info("\nFeature similarity scores:")
    for feature, score in results['feature_similarity'].items():
        logging.info(f"  {feature}: {score:.4f}")
    
    # Save results
    if cfg.output.save_results:
        result_path = os.path.join(cfg.paths.output_dir, cfg.output.result_filename)
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"\nResults saved to: {result_path}")
    
    # Generate visualization
    if cfg.output.save_plots:
        plot_path = os.path.join(cfg.paths.output_dir, cfg.output.plot_filename)
        plot_similarity_heatmap(results, plot_path)
        logging.info(f"Similarity heatmap saved to: {plot_path}")
    
    return results


if __name__ == "__main__":
    main()
