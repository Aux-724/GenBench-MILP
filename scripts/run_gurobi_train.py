"""
Use a single training set for parameter tuning and evaluate performance on a test set.

This script is suitable for parameter tuning and testing on a single instance set, without comparing multiple instance sets.
If you need to compare the tuning effects of different instance sets, please use run_gurobi_compare.py.

This script works in the following steps:
1. Use the training set to tune Gurobi parameters
2. Evaluate the tuned parameter configuration on the test set
3. Compare with default parameters, calculate performance improvement
4. Save detailed results

"""

import os
import sys
import json
import logging
import hydra
import numpy as np
from datetime import datetime
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import parameter tuning module
from metrics.gurobi_param_tuning import GurobiParamTuning

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
)
logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="gurobi_train", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main function, managed by Hydra.
    
    Args:
        cfg: Hydra configuration object
    """
    # Print configuration information
    logger.info("Gurobi parameter tuning configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg.paths.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output results will be saved to: {output_dir}")
    
    # Save configuration
    with open(output_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    
    # Create parameter tuner
    tuner = GurobiParamTuning(cfg)
    
    # Execute training-test workflow
    logger.info("Starting parameter tuning process...")
    
    # Use custom instance set for tuning
    logger.info(f"Using custom instance set for parameter tuning: {cfg.paths.custom_instances_dir}")
    results = tuner.run(
        tuning_mode="custom",
        repeat=cfg.evaluation.repeat,
        evaluate_on_test=True
    )
    
    # Save tuning results
    results_file = output_dir / "gurobi_tuning_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Extract key results and save as summary
    if "evaluation_results" in results:
        # Extract default and optimized performance
        default_time = results["evaluation_results"]["Default"]["overall_mean_time"]
        best_time = results["evaluation_results"]["Best"]["overall_mean_time"]
        
        # Calculate performance improvement
        if default_time > 0:
            improvement = (default_time - best_time) / default_time * 100
        else:
            improvement = 0.0
        
        # Create summary
        summary = {
            "default_avg_time": default_time,
            "optimized_avg_time": best_time,
            "improvement_percent": improvement,
            "best_config": results["configs"]["Best"],
            "timestamp": timestamp,
            "training_instances": cfg.paths.training_instances_dir,
            "test_instances": cfg.paths.test_instances_dir,
            "tuning_trials": cfg.tuning.n_trials
        }
        
        # Save summary
        summary_file = output_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        logger.info("\nTuning result summary:")
        logger.info(f"Default average solve time: {default_time:.4f} seconds")
        logger.info(f"Tuned average solve time: {best_time:.4f} seconds")
        logger.info(f"Performance improvement: {improvement:.2f}%")
        logger.info(f"Tuned parameter configuration: {summary['best_config']}")
        
        # Generate result visualization
        if hasattr(tuner, "plot_comparison_results"):
            try:
                plot_path = output_dir / "gurobi_tuning_plot.png"
                configs_to_plot = {
                    "Default": results["configs"]["Default"],
                    "Optimized": results["configs"]["Best"]
                }
                tuner.plot_comparison_results(
                    configs_to_plot,
                    results["evaluation_results"],
                    plot_path
                )
                logger.info(f"Performance comparison plot saved to: {plot_path}")
            except Exception as e:
                logger.warning(f"Failed to generate visualization: {e}")
    else:
        logger.error("Tuning process did not return evaluation results.")
    
    logger.info(f"All results saved to: {output_dir}")
    return results


if __name__ == "__main__":
    main()
