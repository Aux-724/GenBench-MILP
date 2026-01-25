"""
Calculate the feasibility and boundedness ratio of MILP instances in a single dataset.

This script uses FeasibleRatioMetric to calculate the feasibility and boundedness ratio of MILP instances in a specified directory,
which can be used to evaluate the quality of a set of instances.
All parameters are managed through the configuration file (configs/feasible_ratio.yaml).
"""

import os
import glob
import logging
import datetime
from typing import Dict, List, Any
from pathlib import Path
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

# Import FeasibleRatioMetric class
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(project_root))
from metrics.feasible_ratio import FeasibleRatioMetric

# Set logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FeasibleRatioCalculation")


def get_instances(directory: str) -> List[str]:
    """
    Get all .lp files in the directory
    
    Args:
        directory: Instance directory path
        
    Returns:
        List of instance file paths
    """
    if not os.path.exists(directory):
        logger.error(f"Directory does not exist: {directory}")
        return []
    
    # Recursively find all .lp files
    instances = glob.glob(os.path.join(directory, "**", "*.lp"), recursive=True)
    
    return instances


def calculate_feasibility(
    instances_dir: str, 
    output_file: str, 
    time_limit: int = 300, 
    threads: int = 4,
    parallel: bool = True,
    n_jobs: int = None
) -> Dict[str, Any]:
    """
    Calculate the feasibility and boundedness ratio of MILP instances in a specified directory.
    
    Args:
        instances_dir: Instance directory path
        output_file: Output file path
        time_limit: Time limit for solving each instance (in seconds)
        threads: Number of threads used for solving
        parallel: Whether to enable parallel processing
        n_jobs: Number of parallel jobs
        
    Returns:
        Feasible ratio result
    """
    logger.info(f"Calculating feasibility and boundedness ratio of MILP instances in the following directory:")
    logger.info(f"Instance directory: {instances_dir}")
    
    # Create metric instance from parameters
    cfg_dict = {
        "time_limit": time_limit,
        "threads": threads,
        "parallel": parallel,
        "n_jobs": n_jobs
    }
    metric = FeasibleRatioMetric.from_config(cfg_dict)
    
    # Get instances from directory
    instances = get_instances(instances_dir)
    
    logger.info(f"Found {len(instances)} instances")
    
    # Calculate feasible ratio
    logger.info(f"Calculating feasible ratio for {instances_dir}...")
    result = metric.calculate(instances)
    
    # Prepare result report
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("MILP instance feasibility and boundedness ratio report\n")
        f.write("====================\n\n")
        
        # Overall results
        f.write(f"Instance directory: {instances_dir}\n")
        f.write(f"Total instances: {result['details']['total_count']}\n")
        f.write(f"Feasible and bounded instances: {result['details']['feasible_count']}\n")
        f.write(f"Feasible and bounded ratio: {result['value']:.2f}%\n\n")
        
        # Detailed results
        f.write("Instance results\n")
        f.write("---------\n")
        for instance_result in result['details']['instance_results']:
            status = "Feasible and bounded" if instance_result['feasible'] else "Infeasible or unbounded"
            f.write(f"{instance_result['instance']}: {status}\n")
    
    logger.info(f"Results saved to: {output_file}")
    
    return result


@hydra.main(config_path="../configs", config_name="feasible_ratio", version_base=None)
def main(cfg: DictConfig):
    """Main function, manages all configurations through Hydra"""
    # Print used configuration
    logger.info(f"Using configuration: \n{OmegaConf.to_yaml(cfg)}")
    
    # Check if single_dir is present in configuration, if not, use original_dir from compare
    instances_dir = cfg.get('single', {}).get('dir', cfg.compare.original_dir)
    
    # Process relative paths, if path is not absolute, use project root as base
    if not os.path.isabs(instances_dir):
        project_root = Path(__file__).resolve().parent.parent
        instances_dir = os.path.join(project_root, instances_dir)
    
    # Ensure path exists, if not, output error message and provide suggestions
    if not os.path.exists(instances_dir):
        logger.error(f"Instance directory does not exist: {instances_dir}")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error(f"Please check the path settings in the configuration file or ensure the directory exists")
    
    # Generate output file name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Process output directory path
    output_dir = str(cfg.get('single', {}).get('output_dir', cfg.compare.output_dir)).replace("{now:%Y-%m-%d_%H-%M-%S}", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    
    # Process relative paths, if output directory is not absolute, use project root as base
    if not os.path.isabs(output_dir):
        project_root = Path(__file__).resolve().parent.parent
        output_dir = os.path.join(project_root, output_dir)
        
    os.makedirs(output_dir, exist_ok=True)
    # Generate output file name
    output_file = os.path.join(output_dir, f"feasibility_results_{timestamp}.txt")
    
    time_limit = cfg.time_limit
    threads = cfg.threads
    
    # Record parallel configuration
    parallel_config = "Parallel" if cfg.get("parallel", True) else "Sequential"
    n_jobs = cfg.get("n_jobs", "Auto(CPU-1)")
    logger.info(f"Processing mode: {parallel_config}, number of jobs: {n_jobs}")
    
    # Calculate feasible ratio
    result = calculate_feasibility(
        instances_dir=instances_dir,
        output_file=output_file,
        time_limit=time_limit,
        threads=threads,
        parallel=cfg.get("parallel", True),
        n_jobs=cfg.get("n_jobs", None)
    )
    
    # Print summary results
    print("\nFeasible ratio calculation summary:")
    print(f"Instance directory: {instances_dir}")
    print(f"Total instances: {result['details']['total_count']}")
    print(f"Feasible and bounded instances: {result['details']['feasible_count']}")
    print(f"Feasible and bounded ratio: {result['value']:.2f}%")
    print(f"Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
