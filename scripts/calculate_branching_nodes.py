"""
Calculate branching nodes for a single instance set

Usage:
    python calculate_branching_nodes.py

Configuration file:
    The script uses configs/branching_nodes.yaml to control all parameters
    You can override configuration values via command line arguments, e.g.:
    python calculate_branching_nodes.py paths.instances_dir=/path/to/instances
    
Main configuration parameters:
    paths.instances_dir: Directory containing the instance set
    paths.output_dir: Output directory
    solve.time_limit: Solve time limit (seconds)
    solve.parallel.enabled: Whether to use parallel processing
    solve.parallel.processes: Maximum number of processes
"""

import os
import glob
import time
import json
import logging
import numpy as np
import multiprocessing
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import get_original_cwd
import gurobipy as gp
import matplotlib.pyplot as plt
import pandas as pd

def setup_logger(log_level, log_format):
    """Set up logger"""
    level = getattr(logging, log_level)
    logging.basicConfig(level=level, format=log_format)
    return logging.getLogger(__name__)

def get_status_string(status):
    """Convert Gurobi status code to string"""
    status_map = {
        gp.GRB.LOADED: "LOADED",
        gp.GRB.OPTIMAL: "OPTIMAL",
        gp.GRB.INFEASIBLE: "INFEASIBLE",
        gp.GRB.INF_OR_UNBD: "INF_OR_UNBD",
        gp.GRB.UNBOUNDED: "UNBOUNDED",
        gp.GRB.CUTOFF: "CUTOFF",
        gp.GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        gp.GRB.NODE_LIMIT: "NODE_LIMIT",
        gp.GRB.TIME_LIMIT: "TIME_LIMIT",
        gp.GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        gp.GRB.INTERRUPTED: "INTERRUPTED",
        gp.GRB.NUMERIC: "NUMERIC",
        gp.GRB.SUBOPTIMAL: "SUBOPTIMAL",
        gp.GRB.INPROGRESS: "INPROGRESS",
        gp.GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT"
    }
    return status_map.get(status, f"UNKNOWN({status})")

def solve_instance(args):
    """Solve a single instance and return the number of branching nodes."""
    instance_path, time_limit = args
    try:
        # Disable Gurobi logging
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        
        # Load model
        model = gp.read(instance_path, env=env)
        
        # Set solve parameters
        model.setParam("TimeLimit", time_limit)
        model.setParam("Threads", 1)  # Each instance uses 1 thread, due to parallel processing
        
        # Record solve start time
        start_time = time.time()
        
        # Solve model
        model.optimize()
        
        # Record solve end time
        solve_time = time.time() - start_time
        
        # Get model information
        node_count = model.NodeCount
        status = model.Status
        status_str = get_status_string(status)
        
        # Get constraint and variable information
        num_vars = model.NumVars
        num_integer_vars = model.NumIntVars
        num_binary_vars = model.NumBinVars
        num_constraints = model.NumConstrs
        
        # Get objective value and bound (if available)
        obj_val = None
        obj_bound = None
        gap = None
        
        if status == gp.GRB.OPTIMAL:
            obj_val = float(model.ObjVal)
            obj_bound = float(model.ObjBound)
            gap = 0.0
        elif status == gp.GRB.TIME_LIMIT and model.SolCount > 0:
            obj_val = float(model.ObjVal)
            obj_bound = float(model.ObjBound)
            gap = abs(obj_val - obj_bound) / (1e-10 + abs(obj_val)) * 100
            
        return {
            "instance_name": os.path.basename(instance_path),
            "node_count": float(node_count),
            "status": status_str,
            "solve_time": solve_time,
            "obj_val": obj_val,
            "obj_bound": obj_bound,
            "gap": gap,
            "num_vars": num_vars,
            "num_integer_vars": num_integer_vars,
            "num_binary_vars": num_binary_vars,
            "num_constraints": num_constraints
        }
    except Exception as e:
        print(f"Error processing instance {instance_path}: {str(e)}")
        return {
            "instance_name": os.path.basename(instance_path),
            "node_count": None,
            "status": "ERROR",
            "error": str(e)
        }

def generate_plots(stats, output_dir):
    """Generate various visualization charts."""
    node_counts = [r["node_count"] for r in stats["instance_details"] if r["node_count"] is not None]
    solve_times = [r["solve_time"] for r in stats["instance_details"] if "solve_time" in r and r["solve_time"] is not None]
    
    # Create result chart directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Node count histogram
    plt.figure(figsize=(10, 6))
    plt.hist(node_counts, bins=50, alpha=0.75)
    plt.title('Node count distribution')
    plt.xlabel('Node count')
    plt.ylabel('Instance count')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, "node_count_histogram.png"), dpi=300)
    plt.close()
    
    # 2. Node count boxplot
    plt.figure(figsize=(8, 6))
    plt.boxplot(node_counts, vert=True, notch=True, patch_artist=True)
    plt.title('Node count boxplot')
    plt.ylabel('Node count')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, "node_count_boxplot.png"), dpi=300)
    plt.close()
    
    # 3. Solve time histogram
    if solve_times:
        plt.figure(figsize=(10, 6))
        plt.hist(solve_times, bins=50, alpha=0.75)
        plt.title('Solve time distribution')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Instance count')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, "solve_time_histogram.png"), dpi=300)
        plt.close()
    
    # 4. Node count vs Solve time scatter plot
    if solve_times:
        plt.figure(figsize=(10, 6))
        plt.scatter(node_counts, solve_times, alpha=0.5)
        plt.title('Node count vs Solve time')
        plt.xlabel('Node count')
        plt.ylabel('Solve time (seconds)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, "node_count_vs_time_scatter.png"), dpi=300)
        plt.close()
    
    # 5. Solve status distribution pie chart
    status_counts = stats["status_distribution"]
    if status_counts:
        plt.figure(figsize=(10, 8))
        plt.pie(status_counts.values(), labels=status_counts.keys(), autopct='%1.1f%%', shadow=True, startangle=90)
        plt.axis('equal')
        plt.title('Solve status distribution')
        plt.savefig(os.path.join(plots_dir, "status_distribution_pie.png"), dpi=300)
        plt.close()
    
    # 6. Node count cumulative distribution function (CDF)
    plt.figure(figsize=(10, 6))
    sorted_nodes = np.sort(node_counts)
    y = np.arange(1, len(sorted_nodes) + 1) / len(sorted_nodes)
    plt.plot(sorted_nodes, y)
    plt.title('Node count cumulative distribution function (CDF)')
    plt.xlabel('Node count')
    plt.ylabel('Cumulative probability')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, "node_count_cdf.png"), dpi=300)
    plt.close()

def export_to_csv(stats, output_dir):
    """Export instance details to CSV file."""
    csv_path = os.path.join(output_dir, "instance_details.csv")
    df = pd.DataFrame(stats["instance_details"])
    df.to_csv(csv_path, index=False)
    print(f"Instance details exported to: {csv_path}")

@hydra.main(config_path="../configs", config_name="branching_nodes")
def main(cfg: DictConfig):
    """Main function."""
    # Set up logging
    logger = setup_logger(cfg.logging.level, cfg.logging.format)
    
    # Get absolute paths for instances and output directories
    cwd = get_original_cwd()
    instances_dir = os.path.abspath(os.path.join(cwd, cfg.paths.instances_dir 
                                            if hasattr(cfg.paths, 'instances_dir') 
                                            else cfg.paths.generated_instances_dir))
    output_dir = os.path.abspath(cfg.paths.output_dir)
    
    # Check if instances directory exists
    if not os.path.isdir(instances_dir):
        logger.error(f"Error: Instances directory does not exist: {instances_dir}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset-specific subdirectory
    dataset_name = os.path.basename(instances_dir)
    result_dir = os.path.join(output_dir, f"branching_nodes_{dataset_name}")
    os.makedirs(result_dir, exist_ok=True)
    
    logger.info(f"Calculating branching nodes for dataset: {instances_dir}")
    logger.info(f"Results will be saved to: {result_dir}")
    
    # Get all LP files
    lp_files = sorted(glob.glob(os.path.join(instances_dir, "*.lp")))
    mps_files = sorted(glob.glob(os.path.join(instances_dir, "*.mps")))
    all_files = lp_files + mps_files
    
    if not all_files:
        logger.error(f"Error: No LP or MPS files found in directory {instances_dir}")
        return
    
    # Limit number of instances
    max_instances = cfg.get('max_instances', 0)
    if max_instances > 0 and max_instances < len(all_files):
        logger.info(f"Note: Limiting processing to {max_instances}/{len(all_files)} instances")
        all_files = all_files[:max_instances]
    
    logger.info(f"Found {len(all_files)} instances to process")
    
    # Prepare solve parameters
    time_limit = cfg.solve.time_limit
    logger.info(f"Solve time limit: {time_limit} seconds")
    solve_args = [(instance_file, time_limit) for instance_file in all_files]
    
    # Process all instances
    start_time = time.time()
    
    if cfg.solve.parallel.enabled and len(all_files) > 1:
        num_processes = min(cfg.solve.parallel.processes, len(all_files))
        logger.info(f"Using {num_processes} processes to parallelly process instances")
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap(solve_instance, solve_args), total=len(all_files), desc="Processing instances"))
    else:
        logger.info("Using single-threaded sequential processing")
        results = []
        for arg in tqdm(solve_args, desc="Processing instances"):
            results.append(solve_instance(arg))
    
    total_time = time.time() - start_time
    
    # Filter out failed instances
    valid_results = [r for r in results if r.get("node_count") is not None]
    node_counts = [r["node_count"] for r in valid_results]
    
    # Calculate status distribution
    status_counts = {}
    for r in valid_results:
        status = r["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Calculate statistics
    stats = {}
    
    if node_counts:
        stats = {
            "dataset_name": dataset_name,
            "total_instances": len(all_files),
            "valid_instances": len(valid_results),
            "total_nodes": float(sum(node_counts)),
            "mean_nodes": float(np.mean(node_counts)),
            "median_nodes": float(np.median(node_counts)),
            "max_nodes": float(np.max(node_counts)),
            "min_nodes": float(np.min(node_counts)),
            "std_nodes": float(np.std(node_counts)),
            "status_distribution": status_counts,
            "total_processing_time": total_time,
            "average_processing_time": total_time / len(all_files),
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "parameters": {
                "time_limit": time_limit,
                "parallel": cfg.solve.parallel.enabled,
                "processes": cfg.solve.parallel.processes if cfg.solve.parallel.enabled else 1,
                "max_instances": max_instances
            },
            "instance_details": valid_results
        }
        
        # Print statistics
        logger.info("\nStatistics:")
        logger.info(f"  Total instances: {stats['total_instances']}")
        logger.info(f"  Valid instances: {stats['valid_instances']}")
        logger.info(f"  Total nodes: {stats['total_nodes']}")
        logger.info(f"  Mean nodes: {stats['mean_nodes']:.2f}")
        logger.info(f"  Median nodes: {stats['median_nodes']:.2f}")
        logger.info(f"  Max nodes: {stats['max_nodes']}")
        logger.info(f"  Min nodes: {stats['min_nodes']}")
        logger.info(f"  Node count standard deviation: {stats['std_nodes']:.2f}")
        logger.info(f"  Total processing time: {stats['total_processing_time']:.2f} seconds")
        
        logger.info("\n  Solve status distribution:")
        for status, count in stats['status_distribution'].items():
            logger.info(f"    {status}: {count} instances ({count/stats['valid_instances']*100:.2f}%)")
    else:
        stats = {
            "dataset_name": dataset_name,
            "total_instances": len(all_files),
            "valid_instances": 0,
            "error": "Unable to get valid node count statistics",
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "parameters": {
                "time_limit": time_limit,
                "parallel": cfg.solve.parallel.enabled,
                "processes": cfg.solve.parallel.processes if cfg.solve.parallel.enabled else 1,
                "max_instances": max_instances
            }
        }
        logger.error("\nError: Unable to get valid node count statistics")
    
    # Save JSON results
    results_filename = cfg.output.result_filename
    output_path = os.path.join(result_dir, results_filename)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"\nResults saved to: {output_path}")
    
    # Generate visualization plots
    if node_counts and cfg.output.save_plots:
        try:
            logger.info("\nGenerating visualization plots...")
            generate_plots(stats, result_dir)
            logger.info(f"Plots saved to: {os.path.join(result_dir, 'plots')}")
        except Exception as e:
            logger.error(f"Error generating plots: {str(e)}")
    
    # Export CSV
    if valid_results:
        try:
            export_to_csv(stats, result_dir)
        except Exception as e:
            logger.error(f"Error exporting CSV: {str(e)}")
    
    logger.info("\nCompleted")

if __name__ == "__main__":
    main()
