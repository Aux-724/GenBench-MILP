#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calculate solving time statistics for a single set of MILP instances.

This script uses the TimeGapMetric class to solve MILP instances in a specified directory
and generate solving time statistics and visualization charts.

Uses Hydra to manage configuration parameters, sharing the time_gap.yaml configuration file.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import hydra
import random
import tqdm
from datetime import datetime
from typing import Dict, List, Any, Optional
from omegaconf import DictConfig, OmegaConf

# Add project root to system path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import TimeGapMetric class from metrics
from metrics.time_gap import TimeGapMetric


def create_solving_time_visualization(solving_times: List[float], output_path: str):
    """
    Create solving time visualization chart.
    
    Args:
        solving_times: List of solving times
        output_path: Output path
    """
    plt.figure(figsize=(10, 8))
    
    # Create a 2x2 subplot layout
    plt.subplot(2, 2, 1)
    plt.boxplot(solving_times, labels=['Solving Time'])
    plt.title('Solving Time Distribution')
    plt.ylabel('Time (seconds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Sorted solving times
    plt.subplot(2, 2, 2)
    indices = range(len(solving_times))
    plt.scatter(indices, sorted(solving_times), color='blue')
    plt.title('Sorted Solving Time')
    plt.xlabel('Instance Index')
    plt.ylabel('Time (seconds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Solving time histogram
    plt.subplot(2, 2, 3)
    plt.hist(solving_times, bins=20, color='green', alpha=0.7)
    plt.title('Solving Time Histogram')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Cumulative distribution function
    plt.subplot(2, 2, 4)
    sorted_times = np.sort(solving_times)
    cumulative = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
    plt.step(sorted_times, cumulative, where='post', color='purple')
    plt.title('Cumulative Distribution Function')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Cumulative Probability')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_html_report(results: List[Dict], stats: Dict, output_path: str):
    """
    Generate HTML format solving time report.
    
    Args:
        results: Solving results list
        stats: Statistics information
        output_path: Report output path
    """
    # Create DataFrame to display results
    df = pd.DataFrame([
        {
            "Instance Name": r["instance_name"],
            "Solving Time (s)": r["solving_time"] if r["solving_time"] is not None else "N/A",
            "Solving Status": r["status_name"],
            "Is Feasible": "Yes" if r.get("is_feasible", False) else "No",
            "Objective Value": r.get("obj", "N/A") if r.get("is_feasible", False) else "N/A",
            "Nodes": r.get("num_nodes", "N/A"),
            "Solutions": r.get("num_sols", "N/A")
        }
        for r in results
    ])
    
    # Generate HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Solving Time Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .summary {{ background-color: #eef; padding: 15px; border-radius: 5px; }}
            .error {{ color: red; }}
            .success {{ color: green; }}
        </style>
    </head>
    <body>
        <h1>Solving Time Report</h1>
        <div class="summary">
            <h2>Statistics Summary</h2>
            <p>Instance count: <strong>{stats['total_count']}</strong></p>
            <p>Solved instance count: <strong>{stats['solved_count']}</strong></p>
            <p>Mean solving time: <strong>{stats['mean_time']:.4f} seconds</strong></p>
            <p>Median solving time: <strong>{stats['median_time']:.4f} seconds</strong></p>
            <p>Shortest solving time: <strong>{stats['min_time']:.4f} seconds</strong></p>
            <p>Longest solving time: <strong>{stats['max_time']:.4f} seconds</strong></p>
            <p>Solving time standard deviation: <strong>{stats['std_time']:.4f} seconds</strong></p>
            <p>Feasible instance count: <strong>{stats['feasible_count']}</strong></p>
            <p>Infeasible instance count: <strong>{stats['infeasible_count']}</strong></p>
            <p>Error instance count: <strong>{stats['error_count']}</strong></p>
        </div>
        
        <h2>Detailed Solving Results</h2>
        {df.to_html(index=False)}
        
        <h2>Solving Time Distribution</h2>
        <img src="solving_time_plot.png" alt="Solving Time Distribution" style="max-width: 100%;">
        
        <footer>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </footer>
    </body>
    </html>
    """
    
    # Write HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


# Configure without Hydra changing working directory
@hydra.main(config_path="../configs", config_name="time_gap", version_base=None)
def main(cfg: DictConfig):
    """Main function"""
    # Get original project root directory (need to save before Hydra changes working directory)
    original_cwd = hydra.utils.get_original_cwd()
    
    # Print configuration information
    print(f"=== Current configuration ===\n{OmegaConf.to_yaml(cfg)}\n")
    
    # Use instances_dir parameter from config (if exists), otherwise use training_instances_dir
    instances_dir_param = cfg.paths.get('instances_dir', cfg.paths.training_instances_dir)
    instances_dir = os.path.join(original_cwd, instances_dir_param)
    
    # Set output directory (using timestamp to avoid overwriting previous results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(original_cwd, f"{cfg.paths.output_dir}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get other configuration parameters
    skip_solve = cfg.get('calculate_instance', {}).get('skip_solve', False)
    max_instances = cfg.get('calculate_instance', {}).get('max_instances', None)
    
    print(f"=== MILP instance solving time analysis ===" )
    print(f"Original working directory: {original_cwd}" )
    print(f"Instance directory: {instances_dir}" )
    print(f"Output directory: {output_dir}" )
    print(f"Solving time limit: {cfg.solve.time_limit if cfg.solve.time_limit else 'no limit'}" )
    print(f"Threads: {cfg.solve.threads}")
    print(f"Skip solving: {skip_solve}")
    if max_instances:
        print(f"Maximum solving instances: {max_instances}")
    
    # Create a configuration object specifically for this script
    from copy import deepcopy
    exec_cfg = deepcopy(cfg)
    
    # Override path settings to ensure correct working even after Hydra changes working directory
    exec_cfg.paths.training_instances_dir = instances_dir
    exec_cfg.paths.generated_instances_dir = instances_dir  # In this script, these directories are the same
    exec_cfg.paths.output_dir = output_dir
    
    # Create TimeGapMetric instance
    metric = TimeGapMetric(exec_cfg)
    
    # Get instance list
    instances = metric._get_instances_from_dir(instances_dir)
    if not instances:
        print(f"Error: No MILP instance files found in directory {instances_dir}")
        return
    
    # If specified maximum instances, sample randomly
    if max_instances and len(instances) > max_instances:
        random_seed = cfg.get('calculate_instance', {}).get('random_seed', 42)
        random.seed(random_seed)
        instances = random.sample(instances, max_instances)
        print(f"Randomly sampled {max_instances} instances (total {len(instances)} instances found)")
    else:
        print(f"Found {len(instances)} MILP instances")
    
    # Store solving results
    results = []
    solving_times = []
    
    # Use skip_solve parameter from configuration file
    if not skip_solve:
        print("\nSolving instances...")
        # Use tqdm to create progress bar, without detailed instance information
        for i, instance_path in enumerate(tqdm.tqdm(instances, desc="Solving Progress", ncols=100)):
            instance_name = os.path.basename(instance_path)
            
            try:
                solve_time, obj_value, status = metric._solve_instance(
                    instance_path, 
                    time_limit=cfg.solve.time_limit,
                    threads=cfg.solve.threads
                )
                
                # Parse status information
                is_feasible = False
                is_error = False
                
                if status is not None and "optimal" in status.lower():
                    is_feasible = True
                elif status is not None and "feasible" in status.lower():
                    is_feasible = True
                elif status is not None and "error" in status.lower():
                    is_error = True
                
                # Collect results
                result = {
                    "instance_name": instance_name,
                    "instance_path": instance_path,
                    "solving_time": solve_time,
                    "obj": obj_value,
                    "status_name": status,
                    "is_feasible": is_feasible,
                    "is_error": is_error
                }
                
                results.append(result)
                
                if solve_time is not None:
                    solving_times.append(solve_time)
                
            except Exception as e:
                print(f"Error: Solving {instance_name} failed: {str(e)}")
                results.append({
                    "instance_name": instance_name,
                    "instance_path": instance_path,
                    "solving_time": None,
                    "obj": None,
                    "status_name": f"ERROR: {str(e)}",
                    "is_feasible": False,
                    "is_error": True
                })
    
    # Calculate statistics
    total_count = len(results)
    solved_count = len([r for r in results if r["solving_time"] is not None])
    feasible_count = len([r for r in results if r.get("is_feasible", False)])
    infeasible_count = len([r for r in results if not r.get("is_feasible", False) and not r.get("is_error", False)])
    error_count = len([r for r in results if r.get("is_error", False)])
    
    # Calculate solving time statistics
    if solving_times:
        mean_time = np.mean(solving_times)
        median_time = np.median(solving_times)
        min_time = np.min(solving_times)
        max_time = np.max(solving_times)
        std_time = np.std(solving_times)
    else:
        mean_time = median_time = min_time = max_time = std_time = 0.0
    
    # Aggregate statistics
    stats = {
        "total_count": total_count,
        "solved_count": solved_count,
        "mean_time": mean_time,
        "median_time": median_time,
        "min_time": min_time,
        "max_time": max_time,
        "std_time": std_time,
        "feasible_count": feasible_count,
        "infeasible_count": infeasible_count,
        "error_count": error_count
    }
    
    # Print statistics summary
    print("\n=== Solving Statistics Summary ===")
    print(f"Instance count: {total_count}")
    print(f"Solved instance count: {solved_count}")
    print(f"Mean solving time: {mean_time:.4f} seconds")
    print(f"Median solving time: {median_time:.4f} seconds")
    print(f"Shortest solving time: {min_time:.4f} seconds")
    print(f"Longest solving time: {max_time:.4f} seconds")
    print(f"Solving time standard deviation: {std_time:.4f} seconds")
    print(f"Feasible instance count: {feasible_count}")
    print(f"Infeasible instance count: {infeasible_count}")
    print(f"Error instance count: {error_count}")
    
    # Create visualization
    if solving_times and cfg.output.save_plots:
        plot_path = os.path.join(output_dir, "solving_time_plot.png")
        print(f"\nGenerating solving time distribution plot...")
        create_solving_time_visualization(solving_times, plot_path)
        print(f"Chart saved to: {plot_path}")
    
    # Save JSON results
    if cfg.output.save_results:
        json_path = os.path.join(output_dir, "solving_time_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "results": results,
                "stats": stats
            }, f, indent=2, ensure_ascii=False)
        print(f"Detailed results saved to: {json_path}")
    
    # Generate HTML report (if enabled in configuration)
    if cfg.get('calculate_instance', {}).get('html_report', True):
        html_path = os.path.join(output_dir, "solving_time_report.html")
        print(f"\nGenerating HTML report...")
        generate_html_report(results, stats, html_path)
        print(f"HTML report saved to: {html_path}")
    
    return {"stats": stats}

if __name__ == "__main__":
    main()
