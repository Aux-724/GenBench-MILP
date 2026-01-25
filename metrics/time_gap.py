"""
Solving Time Gap metric implementation.

This metric calculates the time gap between generated MILP instances and training instances,
helping evaluate if the generated instances have similar computational complexity to the original instances.

Formula: Gap = |(Generated instance solve time - Training instance solve time) / Training instance solve time| × 100%
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import glob

from utils.common import solve_instance

class TimeGapMetric:
    """
    Solving Time Gap metric.
    
    This metric calculates the time gap between generated MILP instances and training instances,
    helping evaluate if the generated instances have similar computational complexity to the original instances.
    
    Formula: Gap = |(Generated instance solve time - Training instance solve time) / Training instance solve time| × 100%
    """
    
    def __init__(self, cfg: Optional[DictConfig] = None):
        """
        Initialize the solving time gap metric.
        
        Args:
            cfg: Hydra configuration object, if None, use default configuration
        """
        self.name = "Solving Time Gap"
        self.params = {}
        
        # Get project root directory (more reliable way)
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # If no configuration provided, use default configuration
        if cfg is None:
            # Try loading configuration file using Hydra
            try:
                cfg = OmegaConf.load(os.path.join(script_dir, "configs/time_gap.yaml"))
            except:
                # If loading fails, create default configuration
                cfg = {
                    "solve": {
                        "time_limit": 1000,
                        "threads": 1
                    },
                    "paths": {
                        "training_instances_dir": os.path.join(script_dir, "data/raw"),
                        "generated_instances_dir": os.path.join(script_dir, "data/generated"),
                        "output_dir": os.path.join(script_dir, "outputs/time_gap")
                    },
                    "output": {
                        "save_results": True,
                        "result_filename": "time_gap_results.json",
                        "save_plots": True,
                        "plot_filename": "time_gap_plot.png"
                    },
                    "logging": {
                        "level": "INFO",
                        "format": "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
                    }
                }
                cfg = OmegaConf.create(cfg)
        
        # Save configuration parameters
        self.params["time_limit"] = cfg.solve.time_limit
        self.params["threads"] = cfg.solve.threads
        self.params["training_instances_dir"] = cfg.paths.training_instances_dir
        self.params["generated_instances_dir"] = cfg.paths.generated_instances_dir
        self.params["output_dir"] = cfg.paths.output_dir
        self.params["save_results"] = cfg.output.save_results
        self.params["result_filename"] = cfg.output.result_filename
        self.params["save_plots"] = cfg.output.save_plots
        self.params["plot_filename"] = cfg.output.plot_filename
        
        # Set logging
        log_level = getattr(logging, cfg.logging.level)
        logging.basicConfig(
            level=log_level,
            format=cfg.logging.format
        )
        
        # Create output directory
        if self.params["save_results"] or self.params["save_plots"]:
            os.makedirs(self.params["output_dir"], exist_ok=True)
    
    def calculate(self, 
                 training_instances: Optional[List[str]] = None, 
                 generated_instances: Optional[List[str]] = None, 
                 **kwargs) -> Dict[str, Any]:
        """
        Calculate the time gap between training and generated MILP instances.
        
        Args:
            training_instances: List of paths to training MILP instances. If None, uses default directory.
            generated_instances: List of paths to generated MILP instances. If None, uses default directory.
            **kwargs: Additional parameters.
            
        Returns:
            Dictionary containing the time gap metric.
        """
        # Update parameters (if provided)
        temp_params = self.params.copy()
        temp_params.update(kwargs)
        
        # If no instance paths provided, load from default directories
        if training_instances is None:
            training_dir = temp_params["training_instances_dir"]
            training_instances = self._get_instances_from_dir(training_dir)
            logging.info(f"Loading training instances from directory: {training_dir}，found {len(training_instances)} instances")
        
        if generated_instances is None:
            generated_dir = temp_params["generated_instances_dir"]
            generated_instances = self._get_instances_from_dir(generated_dir)
            logging.info(f"Loading generated instances from directory: {generated_dir}，found {len(generated_instances)} instances")
        
        # Solve training instances
        training_times = []
        training_results = []
        
        logging.info("Begin solving training instances...")
        batch_size = 50  # Process 50 instances at a time
        for i, instance_path in enumerate(training_instances):
            try:
                # Only output information for the first instance or last instance in each batch
                if i % batch_size == 0 or i == len(training_instances) - 1:
                    logging.info(f"Training instance progress [{i+1}/{len(training_instances)}]: currently processing {os.path.basename(instance_path)}")
                solve_time, obj_value, status = self._solve_instance(
                    instance_path, 
                    time_limit=temp_params["time_limit"],
                    threads=temp_params["threads"]
                )
                if solve_time is not None:
                    training_times.append(solve_time)
                    training_results.append({
                        "instance": os.path.basename(instance_path),
                        "solve_time": solve_time,
                        "obj_value": obj_value,
                        "status": status
                    })
            except Exception as e:
                logging.error(f"Error solving training instance {instance_path}: {str(e)}")
        
        # Solve generated instances
        generated_times = []
        generated_results = []
        
        logging.info("Begin solving generated instances...")
        batch_size = 50  # Process 50 instances at a time
        for i, instance_path in enumerate(generated_instances):
            try:
                # Only output information for the first instance or last instance in each batch
                if i % batch_size == 0 or i == len(generated_instances) - 1:
                    logging.info(f"Generated instance progress [{i+1}/{len(generated_instances)}]: currently processing {os.path.basename(instance_path)}")
                solve_time, obj_value, status = self._solve_instance(
                    instance_path, 
                    time_limit=temp_params["time_limit"],
                    threads=temp_params["threads"]
                )
                if solve_time is not None:
                    generated_times.append(solve_time)
                    generated_results.append({
                        "instance": os.path.basename(instance_path),
                        "solve_time": solve_time,
                        "obj_value": obj_value,
                        "status": status
                    })
            except Exception as e:
                logging.error(f"Error solving generated instance {instance_path}: {str(e)}")
        
        # Calculate time gap statistics
        if not training_times or not generated_times:
            return {
                "name": self.name,
                "value": None,
                "details": {
                    "error": "No valid solve times found"
                }
            }
        
        # Calculate average solve time
        training_mean_time = np.mean(training_times)
        generated_mean_time = np.mean(generated_times)
        
        # Calculate absolute time gap
        abs_time_gap = abs(generated_mean_time - training_mean_time)
        
        # Calculate relative time gap (percentage)
        rel_time_gap = (abs_time_gap / max(training_mean_time, 1e-10)) * 100
        
        # Calculate time gap for each instance
        instance_gaps = []
        for gen_result in generated_results:
            gen_time = gen_result["solve_time"]
            # Calculate relative gap, avoid denominator being 0 and ensure accuracy of relative gap calculation
            rel_gap = (abs(gen_time - training_mean_time) / max(training_mean_time, 1e-10)) * 100
            instance_gaps.append({
                "instance": gen_result["instance"],
                "solve_time": gen_time,
                "gap_percent": rel_gap
            })
        
        # Sort by time gap
        instance_gaps.sort(key=lambda x: x["gap_percent"])
        
        results = {
            "name": self.name,
            "value": rel_time_gap,
            "details": {
                "absolute_time_gap": abs_time_gap,
                "relative_time_gap_percent": rel_time_gap,
                "training_mean_time": training_mean_time,
                "generated_mean_time": generated_mean_time,
                "training_min_time": min(training_times),
                "training_max_time": max(training_times),
                "generated_min_time": min(generated_times),
                "generated_max_time": max(generated_times),
                "training_count": len(training_times),
                "generated_count": len(generated_times),
                "training_results": training_results,
                "generated_results": generated_results,
                "instance_gaps": instance_gaps
            }
        }
        
        # Save results
        if self.params.get("save_results", False):
            result_path = os.path.join(self.params["output_dir"], self.params["result_filename"])
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logging.info(f"Results saved to: {result_path}")
        
        # Create time gap plot
        if self.params.get("save_plots", False):
            self._create_time_gap_plot(training_times, generated_times, instance_gaps)
        
        return results
    
    def _create_time_gap_plot(self, training_times: List[float], generated_times: List[float], instance_gaps: List[Dict]):
        """
        Create a visualization chart for time gap.
        
        Args:
            training_times: List of solve times for training instances
            generated_times: List of solve times for generated instances
            instance_gaps: List of time gap statistics for each instance
        """
        try:
            plt.figure(figsize=(12, 10))
            
            # Create a 2x2 subplot layout
            plt.subplot(2, 2, 1)
            plt.boxplot([training_times, generated_times], labels=['Training Set', 'Generated Set'])
            plt.title('Solving Time Distribution')
            plt.ylabel('Time (seconds)')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Compare solving times for each instance
            plt.subplot(2, 2, 2)
            training_indices = range(len(training_times))
            generated_indices = range(len(generated_times))
            
            plt.scatter(training_indices, sorted(training_times), color='blue', label='Training Set')
            plt.scatter(generated_indices, sorted(generated_times), color='red', label='Generated Set')
            plt.legend()
            plt.title('Sorted Solving Time')
            plt.xlabel('Instance Index')
            plt.ylabel('Time (seconds)')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Instance time gap percentage
            plt.subplot(2, 2, 3)
            instance_names = [item["instance"] for item in instance_gaps]
            gap_values = [item["gap_percent"] for item in instance_gaps]
            
            plt.bar(range(len(instance_names)), gap_values, color='green')
            plt.xticks(range(len(instance_names)), instance_names, rotation=45, ha='right')
            plt.title('Instance Time Gap Percentage')
            plt.ylabel('Gap Percentage (%)')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Solving time histogram
            plt.subplot(2, 2, 4)
            bins = np.linspace(0, max(max(training_times), max(generated_times)), 20)
            plt.hist(training_times, bins=bins, alpha=0.5, label='Training Set', color='blue')
            plt.hist(generated_times, bins=bins, alpha=0.5, label='Generated Set', color='red')
            plt.legend()
            plt.title('Solving Time Distribution')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Frequency')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.params["output_dir"], self.params["plot_filename"])
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logging.info(f"Plot saved to: {plot_path}")
            plt.close()
            
        except Exception as e:
            logging.error(f"Error creating plot: {str(e)}")
    
    def _solve_instance(self, 
                       instance_path: str, 
                       time_limit: Optional[int] = None,
                       threads: int = 4) -> Tuple[Optional[float], Optional[float], str]:
        """
        Solve MILP instance and return solving time, optimal objective value, and solving status.
        
        Args:
            instance_path: MILP instance file path.
            time_limit: Time limit (seconds). None means no limit.
            threads: Number of threads used.
            
        Returns:
            Tuple of solving time (seconds), optimal objective value, and solving status. Returns None if solving fails.
        """
        try:
            # Use the solve_instance function from the project
            # Correctly pass time_limit and threads parameters
            results = solve_instance(
                instance_path,
                time_limit=time_limit,
                threads=threads
            )
            
            # Check if solving time exists - this is a key indicator of successful solving
            if results["solving_time"] is None:
                logging.warning(f"Error solving {instance_path}: No valid solving time")
                return None, None, results.get("status_name", "UNKNOWN")
            
            # Extract relevant information from the results
            solve_time = results["solving_time"]
            obj_value = results["obj"] if results.get("is_feasible", False) else None
            status_str = results["status_name"]
            
            return solve_time, obj_value, status_str
        
        except Exception as e:
            logging.error(f"Error solving {instance_path}: {str(e)}")
            return None, None, f"ERROR: {str(e)}"
    

    
    def _get_instances_from_dir(self, directory: str) -> List[str]:
        """
        Get all MILP instance file paths from the directory.
        
        Args:
            directory: Directory path.
            
        Returns:
            List of instance file paths.
        """
        # Supported file extensions
        extensions = ['.lp', '.mps', '.mps.gz']
        
        if not os.path.exists(directory):
            logging.warning(f"Directory does not exist: {directory}")
            return []
            
        instances = []
        for ext in extensions:
            pattern = os.path.join(directory, f'*{ext}')
            instances.extend(glob.glob(pattern))
        
        if not instances:
            logging.warning(f"No instance files found in directory: {directory}")
            
        return sorted(instances)
