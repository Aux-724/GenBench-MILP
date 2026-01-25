"""
Compare the structure similarity of MILP instances between two directories

Use the instance_similarity.yaml configuration file to manage all parameters
"""

import os
import sys
import json
import logging
import hydra
from omegaconf import DictConfig, OmegaConf

# Add the project root directory to the system path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the instance similarity function
from metrics.instance_similarity import compute_features, compute_jsdiv

def get_instance_files(directory):
    """Get all MILP instance files in the directory"""
    instance_files = []
    for file in os.listdir(directory):
        if file.endswith('.lp') or file.endswith('.mps'):
            instance_files.append(os.path.join(directory, file))
    return instance_files

def setup_logging(cfg):
    """Set up the logging system"""
    log_level = getattr(logging, cfg.logging.level)
    log_format = cfg.logging.format
    logging.basicConfig(level=log_level, format=log_format)
    return logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="instance_similarity", version_base=None)
def main(cfg: DictConfig):
    # Set up the logging system
    logger = setup_logging(cfg)
    
    # Print configuration information
    logger.info("Loading configuration information...")
    print(OmegaConf.to_yaml(cfg))
    
    # Read the directory paths from the configuration file
    original_dir = cfg.paths.set1_dir
    generated_dir = cfg.paths.set2_dir
    output_dir = cfg.paths.output_dir
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Original instance directory: {original_dir}")
    logger.info(f"Generated instance directory: {generated_dir}")
    logger.info(f"Result output directory: {output_dir}")
    
    # Get instance files
    logger.info(f"Loading original instances: {original_dir}")
    original_instances = get_instance_files(original_dir)
    logger.info(f"Found {len(original_instances)} original instances")
    
    logger.info(f"Loading generated instances: {generated_dir}")
    generated_instances = get_instance_files(generated_dir)
    logger.info(f"Found {len(generated_instances)} generated instances")
    
    # Calculate instance similarity
    logger.info("Calculating instance feature similarity...")
    
    # Calculate instance features
    logger.info("Extracting original instance features...")
    original_features = compute_features(original_dir, num_workers=cfg.compute.num_workers)
    
    logger.info("Extracting generated instance features...")
    generated_features = compute_features(generated_dir, num_workers=cfg.compute.num_workers)
    
    # Calculate similarity
    logger.info("Calculating the structure similarity of instance sets...")
    similarity_score, feature_similarities = compute_jsdiv(
        original_features, 
        generated_features,
        num_samples=cfg.compute.num_samples
    )
    
    # Create result dictionary (only contains similarity information)
    similarity_results = {
        "overall_similarity": similarity_score,
        "feature_similarities": feature_similarities,
        "meta": {
            "set1_dir": original_dir,
            "set2_dir": generated_dir,
            "set1_count": len(original_instances),
            "set2_count": len(generated_instances),
            "num_samples": cfg.compute.num_samples
        }
    }
    
    # Generate a file name with comparison information
    result_filename = cfg.output.result_filename
    if hasattr(cfg.output, 'include_comparison_info') and cfg.output.include_comparison_info:
        # Extract the directory name (last level directory)
        set1_name = os.path.basename(os.path.normpath(original_dir))
        set2_name = os.path.basename(os.path.normpath(generated_dir))
        # Build a new file name, format: original file name prefix_set1 name_vs_set2 name.extension
        name_parts = os.path.splitext(result_filename)
        result_filename = f"{name_parts[0]}_{set1_name}_vs_{set2_name}{name_parts[1]}"
        
    # Save the similarity results
    similarity_path = os.path.join(output_dir, result_filename)
    with open(similarity_path, 'w') as f:
        json.dump(similarity_results, f, indent=4)
    
    logger.info(f"Instance structure similarity results have been saved to: {similarity_path}")
    
    # Save the heatmap (if needed)
    if cfg.output.save_plots:
        plot_filename = cfg.output.plot_filename
        if hasattr(cfg.output, 'include_comparison_info') and cfg.output.include_comparison_info:
            # Add comparison information to the chart
            name_parts = os.path.splitext(plot_filename)
            plot_filename = f"{name_parts[0]}_{set1_name}_vs_{set2_name}{name_parts[1]}"
            
        # Generate and save the heatmap
        plot_path = os.path.join(output_dir, plot_filename)
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Prepare heatmap data
            features = list(feature_similarities.keys())
            similarities = list(feature_similarities.values())
            
            # Create heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                [similarities], 
                annot=True, 
                cmap='YlGnBu', 
                vmin=0, 
                vmax=1, 
                xticklabels=features, 
                yticklabels=[f"{set1_name} vs {set2_name}"],
                cbar_kws={'label': 'Similarity score'}
            )
            plt.title(f"Instance feature similarity ({set1_name} vs {set2_name})\nOverall similarity: {similarity_score:.4f}")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300)
            plt.close()
            
            logger.info(f"Feature similarity heatmap has been saved to: {plot_path}")
        except Exception as e:
            logger.warning(f"Failed to generate heatmap: {str(e)}")

if __name__ == "__main__":
    main()
