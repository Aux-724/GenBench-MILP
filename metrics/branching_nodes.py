"""
Branching nodes metric implementation.

This metric calculates the number of branching nodes generated during the Gurobi
solving process of MILP instances, and compares the number of branching nodes
between generated instances and training instances.

Formula: Relative Error = |(Generated instance branching nodes - Training instance branching nodes) / Training instance branching nodes| × 100%
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
import gurobipy as gp
from gurobipy import GRB
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm

class BranchingNodesMetric:
    """
    Branching nodes metric.
    
    This metric calculates the number of branching nodes generated during the Gurobi
    solving process of MILP instances, and compares the number of branching nodes
    between generated instances and training instances.
    
    Formula: Relative Error = |(Generated instance branching nodes - Training instance branching nodes) / Training instance branching nodes| × 100%
    """
    
    def __init__(self, cfg: Optional[DictConfig] = None):
        """
        Initialize the branching nodes metric.
        
        Args:
            cfg: Hydra configuration object, if None, use default configuration
        """
        self.name = "Branching Nodes"
        self.params = {}
        
        # If no configuration is provided, use default configuration
        if cfg is None:
            # Try to load configuration file using relative path
            try:
                # Calculate configuration file path (relative to current module)
                module_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(module_dir)
                config_path = os.path.join(project_root, "configs/branching_nodes.yaml")
                cfg = OmegaConf.load(config_path)
            except:
                # If loading fails, create default configuration
                cfg = {
                    "solve": {
                        "time_limit": 30,  # Default 30 seconds, avoid spending too much time on complex instances
                        "threads": 1,      # Use 1 thread per Gurobi instance to avoid conflict with parallel processing
                        "parallel": {
                            "enabled": True,
                            "processes": max(1, multiprocessing.cpu_count() - 1)  # Default use CPU core count - 1 processes
                        }
                    },
                    "paths": {
                        "training_instances_dir": os.path.join(os.getcwd(), "data/raw"),
                        "generated_instances_dir": os.path.join(os.getcwd(), "data/generated"),
                        "output_dir": os.path.join(os.getcwd(), "data/branching_nodes")
                    },
                    "output": {
                        "save_results": True,
                        "result_filename": "branching_nodes_results.json",
                        "save_plots": True,
                        "plot_filename": "branching_nodes_plot.png"
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
        self.params["parallel_enabled"] = cfg.solve.parallel.enabled
        self.params["parallel_processes"] = cfg.solve.parallel.processes
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
    
    def get_branching_nodes(self, instance_path: str) -> Optional[int]:
        """
        Get the number of branching nodes generated during the Gurobi solving process of a MILP instance.
        
        Args:
            instance_path: Path to the MILP instance file
            
        Returns:
            Number of branching nodes generated during the Gurobi solving process, or None if an error occurs
        """
        # Try to load and solve the instance
        try:
            # Disable Gurobi logging output
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.start()
            
            # Load model
            model = gp.read(instance_path, env=env)
            
            # Set solving parameters
            if self.params["time_limit"] is not None:
                model.setParam("TimeLimit", self.params["time_limit"])
            model.setParam("Threads", self.params["threads"])
            
            # Add interrupt heuristics to find feasible solutions faster
            model.setParam("Heuristics", 0.5)  # Set to half of the default value to reduce heuristic search time
            model.setParam("MIPFocus", 1)      # Focus on finding feasible solutions
            
            # Solve model
            model.optimize()
            
            # Get the number of branching nodes
            node_count = model.NodeCount
            
            # Ensure the node count is greater than 0
            if node_count > 0:
                logging.debug(f"Instance {os.path.basename(instance_path)} has {node_count} branching nodes")
                return node_count
            else:
                logging.warning(f"Instance {os.path.basename(instance_path)} has 0 branching nodes, which may be an abnormal situation")
                return None
            
        except Exception as e:
            logging.error(f"Error processing instance {instance_path}: {str(e)}")
            return None
    
    def _process_instance_wrapper(self, args):
        """
        Process a single instance wrapper function for parallel processing.
        
        Args:
            args: Tuple containing instance path and other parameters
            
        Returns:
            (instance name, number of branching nodes) tuple, or None if processing fails
        """
        instance_path, index, total = args
        result = self.get_branching_nodes(instance_path)
        
        if result is not None:
            return (os.path.basename(instance_path), result)
        return None
    
    def calculate(self, 
                 training_instances: Optional[List[str]] = None, 
                 generated_instances: Optional[List[str]] = None, 
                 **kwargs) -> Dict[str, Any]:
        """
        Calculate the difference in the number of branching nodes between training instances and generated instances.
        
        Args:
            training_instances: List of training instance file paths, if None, use all .mps files in the training directory
            generated_instances: List of generated instance file paths, if None, use all .mps files in the generated directory
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing calculation results, main result stored in 'value' key, representing relative error percentage
        """
        logging.info("Solving...")
        
        # If no training instances provided, use default directory
        if training_instances is None:
            training_dir = self.params["training_instances_dir"]
            training_instances = glob.glob(os.path.join(training_dir, "*.mps")) + glob.glob(os.path.join(training_dir, "*.lp"))
            print(f"From {training_dir} found {len(training_instances)} training instances")
        
        # If no generated instances provided, use default directory
        if generated_instances is None:
            generated_dir = self.params["generated_instances_dir"]
            generated_instances = glob.glob(os.path.join(generated_dir, "*.mps")) + glob.glob(os.path.join(generated_dir, "*.lp"))
            print(f"From {generated_dir} found {len(generated_instances)} generated instances")
        
        # If no instances found, return error
        if not training_instances:
            raise ValueError(f"No training instances found in directory: {self.params['training_instances_dir']}\nPlease confirm that the directory contains .mps or .lp files.")
        
        if not generated_instances:
            raise ValueError(f"No generated instances found in directory: {self.params['generated_instances_dir']}\nPlease confirm that the directory contains .mps or .lp files.")
        
        # Calculate the number of branching nodes for each training instance
        print("\n===== Start calculating training instance branching node count =====")
        print(f"Total {len(training_instances)} training instances to process...")
        
        # Decide whether to use parallel processing
        if self.params["parallel_enabled"] and len(training_instances) > 10:
            print(f"Enable parallel processing, using {self.params['parallel_processes']} processes...")
            
            # Prepare process pool and parameters
            with Pool(processes=self.params["parallel_processes"]) as pool:
                # Prepare parameter list: (instance path, index, total count)
                args_list = [(path, i, len(training_instances)) for i, path in enumerate(training_instances)]
                
                # Use tqdm to display progress bar
                results = list(tqdm(
                    pool.imap(self._process_instance_wrapper, args_list),
                    total=len(args_list),
                    desc="Processing training instances"
                ))
                
                # Filter out successful results
                training_node_counts = [r for r in results if r is not None]
                successful_count = len(training_node_counts)
                failed_count = len(training_instances) - successful_count
        else:
            # Serial processing
            training_node_counts = []
            successful_count = 0
            failed_count = 0
            
            # Progress bar display
            for i, inst_path in enumerate(tqdm(training_instances, desc="Processing training instances")):
                # Process instance
                node_count = self.get_branching_nodes(inst_path)
                
                if node_count is not None:  # Only consider successfully solved instances
                    training_node_counts.append((os.path.basename(inst_path), node_count))
                    successful_count += 1
                else:
                    failed_count += 1
                
                # Every 100 instances, output statistics
                if (i + 1) % 100 == 0 and successful_count > 0:
                    current_nodes = [nodes for _, nodes in training_node_counts]
                    print(f"Processed {i+1} instances, successful: {successful_count}, failed: {failed_count}")
                    print(f"Current average branching nodes: {np.mean(current_nodes):.2f}, max: {max(current_nodes)}, min: {min(current_nodes)}")
        
        print(f"\nTraining instance processing completed! Successful: {successful_count}, failed: {failed_count}")
        if training_node_counts:
            nodes_list = [nodes for _, nodes in training_node_counts]
            print(f"Training instance branching node statistics: average={np.mean(nodes_list):.2f}, median={np.median(nodes_list):.2f}, max={max(nodes_list)}, min={min(nodes_list)}")
        
        # Calculate the number of branching nodes for each generated instance
        print("\n===== Start calculating generated instance branching node count =====")
        print(f"Total {len(generated_instances)} generated instances to process...")
        
        # Decide whether to use parallel processing
        if self.params["parallel_enabled"] and len(generated_instances) > 10:
            print(f"Enable parallel processing, using {self.params['parallel_processes']} processes...")
            
            # Prepare process pool and parameters
            with Pool(processes=self.params["parallel_processes"]) as pool:
                # Prepare parameter list: (instance path, index, total count)
                args_list = [(path, i, len(generated_instances)) for i, path in enumerate(generated_instances)]
                
                # Use tqdm to display progress bar
                results = list(tqdm(
                    pool.imap(self._process_instance_wrapper, args_list),
                    total=len(args_list),
                    desc="Processing generated instances"
                ))
                
                # Filter out successful results
                generated_node_counts = [r for r in results if r is not None]
                successful_count = len(generated_node_counts)
                failed_count = len(generated_instances) - successful_count
        else:
            # Serial processing
            generated_node_counts = []
            successful_count = 0
            failed_count = 0
            
            # Progress bar display
            for i, inst_path in enumerate(tqdm(generated_instances, desc="Processing generated instances")):
                # Process instance
                node_count = self.get_branching_nodes(inst_path)
                
                if node_count is not None:  # Only consider successfully solved instances
                    generated_node_counts.append((os.path.basename(inst_path), node_count))
                    successful_count += 1
                else:
                    failed_count += 1
                
                # Every 100 instances, output statistics
                if (i + 1) % 100 == 0 and successful_count > 0:
                    current_nodes = [nodes for _, nodes in generated_node_counts]
                    print(f"Processed {i+1} instances, successful: {successful_count}, failed: {failed_count}")
                    print(f"Current average branching nodes: {np.mean(current_nodes):.2f}, max: {max(current_nodes)}, min: {min(current_nodes)}")
        
        print(f"\nGenerated instance processing completed! Successful: {successful_count}, failed: {failed_count}")
        if generated_node_counts:
            nodes_list = [nodes for _, nodes in generated_node_counts]
            print(f"Generated instance branching node statistics: average={np.mean(nodes_list):.2f}, median={np.median(nodes_list):.2f}, max={max(nodes_list)}, min={min(nodes_list)}")
            
        print("\n===== Branching node count calculation completed =====")
        
        # Prepare result data
        training_names = [name for name, _ in training_node_counts]
        training_nodes = [nodes for _, nodes in training_node_counts]
        generated_names = [name for name, _ in generated_node_counts]
        generated_nodes = [nodes for _, nodes in generated_node_counts]
        
        # Calculate total number of branching nodes
        training_total_nodes = sum(training_nodes) if training_nodes else 0
        generated_total_nodes = sum(generated_nodes) if generated_nodes else 0
        
        # Calculate average number of branching nodes (for display)
        training_mean_nodes = np.mean(training_nodes) if training_nodes else 0
        generated_mean_nodes = np.mean(generated_nodes) if generated_nodes else 0
        
        # Calculate relative error based on total number of branching nodes (not average)
        if training_total_nodes > 0:
            relative_error = abs((generated_total_nodes - training_total_nodes) / training_total_nodes) * 100
        else:
            relative_error = float('inf')
            logging.warning("Training instance total branching nodes is 0, cannot calculate relative error")
        
        # Prepare detailed results - no longer assuming one-to-one mapping between training and generated instances
        training_details = []
        for t_name, t_nodes in training_node_counts:
            training_details.append({
                "instance_name": t_name,
                "node_count": t_nodes
            })
            
        generated_details = []
        for g_name, g_nodes in generated_node_counts:
            generated_details.append({
                "instance_name": g_name,
                "node_count": g_nodes
            })
        
        # Organize results
        results = {
            "value": relative_error,
            "details": {
                "training_count": len(training_nodes),
                "generated_count": len(generated_nodes),
                "training_total_nodes": training_total_nodes,
                "generated_total_nodes": generated_total_nodes,
                "training_mean_nodes": training_mean_nodes,
                "generated_mean_nodes": generated_mean_nodes,
                "training_median_nodes": np.median(training_nodes) if training_nodes else 0,
                "generated_median_nodes": np.median(generated_nodes) if generated_nodes else 0,
                "training_std_nodes": np.std(training_nodes) if training_nodes else 0,
                "generated_std_nodes": np.std(generated_nodes) if generated_nodes else 0,
                "absolute_total_difference": abs(generated_total_nodes - training_total_nodes),
                "training_instances": training_details,
                "generated_instances": generated_details
            }
        }
        
        # Save results
        if self.params["save_results"]:
            result_path = os.path.join(self.params["output_dir"], self.params["result_filename"])
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logging.info(f"Results saved to {result_path}")
        
        # Generate charts
        if self.params["save_plots"] and training_nodes and generated_nodes:
            self._generate_plots(training_names, training_nodes, 
                               generated_names, generated_nodes,
                               results["value"])
        
        return results
    
    def _generate_plots(self, 
                       training_names: List[str],
                       training_nodes: List[float],
                       generated_names: List[str],
                       generated_nodes: List[float],
                       relative_error: float) -> None:
        """
        Generate a plot comparing the number of branching nodes between training and generated instances.
        
        Args:
            training_names: List of training instance names
            training_nodes: List of training instance branching node counts
            generated_names: List of generated instance names
            generated_nodes: List of generated instance branching node counts
            relative_error: Relative error percentage
        """
        # 1. 绘制分布对比图
        plt.figure(figsize=(12, 8))
        
        # Set title and axis labels
        plt.title(f"Branching Node Count Distribution Comparison (Relative Error: {relative_error:.2f}%)", fontsize=14)
        plt.xlabel("Branching Node Count", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        
        # Calculate the number of bins, using Sturges formula as a reference, but ensure at least 5 bins
        n_bins = max(5, int(np.log2(len(training_nodes) + len(generated_nodes)) + 1))
        
        # Draw histograms
        plt.hist(training_nodes, bins=n_bins, alpha=0.5, label='Training Instances')
        plt.hist(generated_nodes, bins=n_bins, alpha=0.5, label='Generated Instances')
        
        # Add legend and statistics
        training_stats = f"Training Instances: Mean={np.mean(training_nodes):.1f}, Median={np.median(training_nodes):.1f}, Standard Deviation={np.std(training_nodes):.1f}"
        generated_stats = f"Generated Instances: Mean={np.mean(generated_nodes):.1f}, Median={np.median(generated_nodes):.1f}, Standard Deviation={np.std(generated_nodes):.1f}"
        
        plt.figtext(0.1, 0.01, training_stats, fontsize=10)
        plt.figtext(0.5, 0.01, generated_stats, fontsize=10)
        
        # Add legend and grid
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save histogram
        plt.tight_layout()
        plot_path = os.path.join(self.params["output_dir"], self.params["plot_filename"])
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        logging.info(f"Distribution comparison plot saved to {plot_path}")
        
        # Generate boxplot for comparison
        plt.figure(figsize=(10, 6))
        plt.title("Branching Node Count Distribution Comparison", fontsize=14)
        plt.boxplot([training_nodes, generated_nodes], labels=['Training Instances', 'Generated Instances'])
        plt.ylabel("Branching Node Count", fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add statistics for each set
        plt.figtext(0.1, 0.01, f"Training Instances: {len(training_nodes)}", fontsize=10)
        plt.figtext(0.5, 0.01, f"Generated Instances: {len(generated_nodes)}", fontsize=10)
        
        # Save boxplot
        plt.tight_layout()
        boxplot_path = os.path.join(self.params["output_dir"], "branching_nodes_boxplot.png")
        plt.savefig(boxplot_path, dpi=300)
        plt.close()
        
        logging.info(f"Boxplot saved to {boxplot_path}")
        
        # Add cumulative distribution function (CDF) plot
        plt.figure(figsize=(10, 6))
        plt.title("Branching Node Count Cumulative Distribution Function", fontsize=14)
        
        # Calculate and plot CDF
        sorted_training = np.sort(training_nodes)
        sorted_generated = np.sort(generated_nodes)
        y_training = np.arange(1, len(sorted_training) + 1) / len(sorted_training)
        y_generated = np.arange(1, len(sorted_generated) + 1) / len(sorted_generated)
        
        plt.plot(sorted_training, y_training, marker='.', linestyle='none', alpha=0.5, label='Training Instances')
        plt.plot(sorted_generated, y_generated, marker='.', linestyle='none', alpha=0.5, label='Generated Instances')
        
        # Add smooth curve
        from scipy.interpolate import make_interp_spline
        if len(sorted_training) > 5:
            x_smooth = np.linspace(min(sorted_training), max(sorted_training), 100)
            y_smooth = make_interp_spline(sorted_training, y_training)(x_smooth)
            plt.plot(x_smooth, y_smooth, linewidth=2, label='Training Instances (smooth)')
            
        if len(sorted_generated) > 5:
            x_smooth = np.linspace(min(sorted_generated), max(sorted_generated), 100)
            y_smooth = make_interp_spline(sorted_generated, y_generated)(x_smooth)
            plt.plot(x_smooth, y_smooth, linewidth=2, label='Generated Instances (smooth)')
            
        plt.xlabel("Branching Node Count", fontsize=12)
        plt.ylabel("Cumulative Probability", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save CDF plot
        plt.tight_layout()
        cdf_path = os.path.join(self.params["output_dir"], "branching_nodes_cdf.png")
        plt.savefig(cdf_path, dpi=300)
        plt.close()
        
        logging.info(f"CDF plot saved to {cdf_path}")
