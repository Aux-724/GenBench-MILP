"""
HiGHS solver behavior metrics evaluation script.

This script uses HiGHS to solve a set of MILP instances and collects key solver behavior metrics, including:
1. Cut types and counts
2. Heuristic method success counts
3. Root node Gap

These metrics can be used to compare the characteristics of different MILP instance sets and their impact on solver behavior.
"""

import os
import sys
import glob
import logging
import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import tempfile
import json
import highspy

# Add the project root directory to the system path to ensure the project modules can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the project metrics parsing module
from metrics.highs_solver_info import HiGHSSolverInfoMetric

@hydra.main(config_path="../configs", config_name="highs_solver_info", version_base=None)
def evaluate_highs_solver(cfg: DictConfig):
    """
    Use HiGHS to solve a set of MILP instances and collect key solver behavior metrics.
    
    Args:
        cfg: Hydra configuration object
    """
    # Set the log level
    log_level = getattr(logging, cfg.logging.level)
    logging.basicConfig(
        level=log_level,
        format=cfg.logging.format
    )
    
    logging.info("Initializing HiGHS solver behavior metrics evaluation...")
    
    # Get the instance directory path
    instances_dir = cfg.paths.instances_dir
    logging.info(f"Instance directory: {instances_dir}")
    
    # Check if the instance directory exists
    if not os.path.exists(instances_dir):
        logging.error(f"Instance directory does not exist: {instances_dir}")
        return
    
    # Get all instance files
    instance_files = []
    for ext in ['*.mps', '*.lp', '*.mps.gz', '*.lp.gz']:
        instance_files.extend(glob.glob(os.path.join(instances_dir, ext)))
    
    if not instance_files:
        logging.error(f"No instance files found in {instances_dir}")
        return
    
    logging.info(f"Found {len(instance_files)} instances to solve")
    
    # Create output directory
    output_dir = cfg.paths.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Create log directory
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize the metrics extractor
    metrics_extractor = HiGHSSolverInfoMetric(cfg)
    
    # Solve all instances
    results = []
    failed_instances = []
    
    for instance_file in tqdm(instance_files, desc="Solving instances"):
        instance_name = os.path.splitext(os.path.basename(instance_file))[0]
        if instance_name.endswith('.mps'):
            instance_name = instance_name[:-4]
        
        log_file = os.path.join(log_dir, f"{instance_name}.log")
        
        logging.info(f"Solving instance: {instance_name}")
        
        # Solve the instance
        solve_result = solve_with_highs(instance_file, log_file, cfg)
        
        if solve_result['success']:
            # Extract metrics from log file
            metrics = metrics_extractor.extract_solver_info(log_file, instance_name)
            
            # Add solve information
            metrics.update({
                'solve_success': True,
                'solve_time': solve_result.get('solve_time', 0.0),
                'status': solve_result.get('status', 'Unknown')
            })
            
            results.append(metrics)
            logging.info(f"Successfully processed {instance_name}")
        else:
            logging.error(f"Failed to solve {instance_name}: {solve_result.get('error', 'Unknown error')}")
            failed_instances.append(instance_name)
            
            # Add failed instance with empty metrics
            empty_metrics = metrics_extractor._get_empty_metrics(instance_name)
            empty_metrics.update({
                'solve_success': False,
                'solve_time': 0.0,
                'status': 'Failed'
            })
            results.append(empty_metrics)
    
    # Create DataFrame from results
    df = pd.DataFrame(results)
    
    # Save results
    if cfg.output.save_results:
        result_file = os.path.join(output_dir, cfg.output.result_filename)
        metrics_extractor.save_results(df, result_file)
        
        # Save configuration
        config_file = os.path.join(output_dir, "config.yaml")
        with open(config_file, 'w') as f:
            OmegaConf.save(cfg, f)
    
    # Create summary plots
    if cfg.output.save_plots:
        plot_dir = os.path.join(output_dir, "plots")
        metrics_extractor.create_summary_plots(df, plot_dir)
    
    # Print summary
    print_summary(df, failed_instances)
    
    logging.info("HiGHS solver behavior metrics evaluation completed!")

def solve_with_highs(instance_file: str, log_file: str, cfg: DictConfig) -> Dict[str, Any]:
    """
    Solve a single instance using HiGHS Python API and return basic solve information.
    
    Args:
        instance_file: Path to the instance file
        log_file: Path to save the log file
        cfg: Configuration object
        
    Returns:
        Dictionary containing basic solve information
    """
    try:
        # Create HiGHS solver instance
        h = highspy.Highs()
        
        # Set solver options
        h.setOptionValue("log_to_console", cfg.highs.get('log_to_console', False))
        h.setOptionValue("log_file", log_file)
        h.setOptionValue("time_limit", cfg.solve.time_limit)
        h.setOptionValue("threads", cfg.solve.threads)
        h.setOptionValue("random_seed", cfg.solve.seed)
        h.setOptionValue("mip_abs_gap", cfg.solve.mip_gap)
        h.setOptionValue("mip_rel_gap", cfg.solve.mip_gap)
        
        # Set HiGHS specific options
        if hasattr(cfg, 'highs'):
            h.setOptionValue("presolve", cfg.highs.get('presolve', 'choose'))
            h.setOptionValue("log_dev_level", cfg.highs.get('log_dev_level', 2))
            h.setOptionValue("mip_report_level", cfg.highs.get('mip_report_level', 2))
            
            if 'mip_heuristic_effort' in cfg.highs:
                h.setOptionValue("mip_heuristic_effort", cfg.highs.mip_heuristic_effort)
            if 'mip_max_nodes' in cfg.highs and cfg.highs.mip_max_nodes > 0:
                h.setOptionValue("mip_max_nodes", cfg.highs.mip_max_nodes)
        
        # Read the problem
        start_time = datetime.now()
        status = h.readModel(instance_file)
        
        if status != highspy.HighsStatus.kOk:
            return {
                'success': False,
                'solve_time': 0.0,
                'status': 'Read Error',
                'error': f'Failed to read model: {status}'
            }
        
        # Solve the problem
        status = h.run()
        end_time = datetime.now()
        
        solve_time = (end_time - start_time).total_seconds()
        
        # Get solution information
        model_status = h.getModelStatus()
        
        # Determine if solve was successful
        success = status == highspy.HighsStatus.kOk
        
        # Map model status to string
        status_map = {
            highspy.HighsModelStatus.kOptimal: 'Optimal',
            highspy.HighsModelStatus.kInfeasible: 'Infeasible',
            highspy.HighsModelStatus.kUnbounded: 'Unbounded',
            highspy.HighsModelStatus.kTimeLimit: 'Time limit',
            highspy.HighsModelStatus.kIterationLimit: 'Iteration limit',
            highspy.HighsModelStatus.kUnknown: 'Unknown'
        }
        
        # Handle additional status types that might exist
        try:
            if hasattr(highspy.HighsModelStatus, 'kBoundedButNotOptimal'):
                status_map[highspy.HighsModelStatus.kBoundedButNotOptimal] = 'Feasible'
        except:
            pass
        
        status_str = status_map.get(model_status, 'Unknown')
        
        return {
            'success': success,
            'solve_time': solve_time,
            'status': status_str,
            'model_status': model_status,
            'highs_status': status
        }
        
    except Exception as e:
        return {
            'success': False,
            'solve_time': 0.0,
            'status': 'Error',
            'error': str(e)
        }



def print_summary(df: pd.DataFrame, failed_instances: List[str]):
    """
    Print a summary of the evaluation results.
    
    Args:
        df: DataFrame containing results
        failed_instances: List of failed instance names
    """
    print("\n" + "="*60)
    print("HiGHS SOLVER BEHAVIOR METRICS EVALUATION SUMMARY")
    print("="*60)
    
    total_instances = len(df)
    successful_instances = len(df[df['solve_success'] == True])
    
    print(f"Total instances: {total_instances}")
    print(f"Successfully solved: {successful_instances}")
    print(f"Failed instances: {len(failed_instances)}")
    
    if successful_instances > 0:
        print(f"\nSolve time statistics:")
        print(f"  Mean: {df[df['solve_success'] == True]['solve_time'].mean():.2f}s")
        print(f"  Median: {df[df['solve_success'] == True]['solve_time'].median():.2f}s")
        print(f"  Max: {df[df['solve_success'] == True]['solve_time'].max():.2f}s")
        
        # Status distribution
        print(f"\nStatus distribution:")
        status_counts = df['status'].value_counts()
        for status, count in status_counts.items():
            print(f"  {status}: {count}")
        
        # Cut plane usage summary
        cut_columns = [col for col in df.columns if col.startswith('cut_')]
        if cut_columns:
            print(f"\nCutting plane usage (total across all instances):")
            cut_totals = df[cut_columns].sum().sort_values(ascending=False)
            for cut_type, total in cut_totals.items():
                if total > 0:
                    print(f"  {cut_type}: {total}")
        
        # Heuristic success summary
        heur_columns = [col for col in df.columns if col.startswith('heur_')]
        if heur_columns:
            print(f"\nHeuristic success (total across all instances):")
            heur_totals = df[heur_columns].sum().sort_values(ascending=False)
            for heur_type, total in heur_totals.items():
                if total > 0:
                    print(f"  {heur_type}: {total}")
        
        # Root gap statistics
        if 'root_gap' in df.columns:
            valid_gaps = df[df['root_gap'] > 0]['root_gap']
            if not valid_gaps.empty:
                print(f"\nRoot gap statistics:")
                print(f"  Mean: {valid_gaps.mean():.2f}%")
                print(f"  Median: {valid_gaps.median():.2f}%")
                print(f"  Max: {valid_gaps.max():.2f}%")
    
    if failed_instances:
        print(f"\nFailed instances:")
        for instance in failed_instances[:10]:  # Show first 10
            print(f"  {instance}")
        if len(failed_instances) > 10:
            print(f"  ... and {len(failed_instances) - 10} more")
    
    print("="*60)

if __name__ == "__main__":
    evaluate_highs_solver()
