"""
SCIP solver behavior metrics evaluation script.

This script uses SCIP to solve a set of MILP instances and collects key solver behavior metrics, including:
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
import subprocess
import tempfile

# Add the project root directory to the system path to ensure the project modules can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the project metrics parsing module
from metrics.scip_solver_info import SCIPSolverInfoMetric

@hydra.main(config_path="../configs", config_name="scip_solver_info", version_base=None)
def evaluate_scip_solver(cfg: DictConfig):
    """
    Use SCIP to solve a set of MILP instances and collect key solver behavior metrics.
    
    Args:
        cfg: Hydra configuration object
    """
    # Set the log level
    log_level = getattr(logging, cfg.logging.level)
    logging.basicConfig(
        level=log_level,
        format=cfg.logging.format
    )
    
    logging.info("Initializing SCIP solver behavior metrics evaluation...")
    
    # Get the instance directory path
    instances_dir = cfg.paths.instances_dir
    logging.info(f"Instance directory: {instances_dir}")
    
    # Get the output directory path
    try:
        if hasattr(cfg, 'hydra') and hasattr(cfg.hydra, 'run') and hasattr(cfg.hydra.run, 'dir'):
            output_dir = cfg.hydra.run.dir
        else:
            # Use the output directory defined in the configuration file as a backup, and add a timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(cfg.paths.output_dir, f"scip_experiment_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        # If any error occurs, use the output directory in the configuration and add a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(cfg.paths.output_dir, f"scip_experiment_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Output directory: {output_dir}")
    
    # Create the SCIP log directory
    scip_logs_dir = os.path.join(output_dir, "scip_logs")
    os.makedirs(scip_logs_dir, exist_ok=True)
    logging.info(f"SCIP log directory: {scip_logs_dir}")
    
    # Get the instance file paths
    instance_files = []
    for ext in ["*.mps", "*.lp", "*.mps.gz", "*.lp.gz"]:
        instance_files.extend(glob.glob(os.path.join(instances_dir, ext)))
    
    if not instance_files:
        logging.error(f"No MPS or LP instance files found in directory {instances_dir}")
        return
    
    logging.info(f"Found {len(instance_files)} instance files")
    
    # Initialize the results list
    results = []
    
    # Initialize the SCIPSolverInfoMetric object, for later log parsing
    solver_metric = SCIPSolverInfoMetric(cfg)
    
    # Iterate through each instance, using tqdm to create a progress bar
    logging.info("Starting to process instances...")
    for instance_file in tqdm(instance_files, desc="SCIP Solving progress", ncols=100, colour="blue"):
        instance_name = os.path.basename(instance_file).split(".")[0]
        
        # Construct a unique log file path
        log_file = os.path.join(scip_logs_dir, f"{instance_name}.log")
        
        try:
            # Solve the instance using SCIP
            instance_result = solve_with_scip(instance_file, log_file, cfg)
            
            # Parse the log file to extract metrics
            log_metrics = solver_metric.analyze_log_file(log_file)
            
            # Combine results
            instance_result.update(log_metrics)
            
            # Add the instance name
            instance_result["instance"] = instance_name
            
            # Add the results to the list
            results.append(instance_result)
            
        except Exception as e:
            logging.error(f"Error processing instance {instance_name}: {str(e)}")
            results.append({
                "instance": instance_name,
                "error": str(e)
            })
    
    # Save the results
    if results:
        # Determine the results save path
        results_path = os.path.join(output_dir, cfg.output.result_filename)
        
        # Convert the results to a DataFrame
        results_df = pd.DataFrame(results)
        
        # Save the results to a CSV file
        results_df.to_csv(results_path, index=False)
        logging.info(f"Results saved to {results_path}")
        
        # Generate plots
        if cfg.output.save_plots:
            plot_path = os.path.join(output_dir, cfg.output.plot_filename)
            solver_metric.plot_results(results_df, plot_path)
        
        # Create result summary information
        summary_info = {}
        summary_info['Instance count'] = len(results)
        summary_info['Dataset path'] = cfg.paths.instances_dir
        summary_info['Evaluation time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary_info['Solver'] = 'SCIP'
        
        # Calculate the total number of each cut type
        cut_columns = [col for col in results_df.columns if col.startswith('cut_')]
        if cut_columns:
            logging.info("Cut plane statistics:")
            cut_totals = results_df[cut_columns].sum().sort_values(ascending=False)
            summary_info['Cut plane statistics'] = {}
            for cut_type, count in cut_totals.items():
                if count > 0:
                    logging.info(f"  {cut_type}: {count}")
                    summary_info['Cut plane statistics'][cut_type] = float(count)
        
        # Calculate the total number of successful attempts for each heuristic method
        heur_columns = [col for col in results_df.columns if col.startswith('heur_')]
        if heur_columns:
            logging.info("Heuristic method success statistics:")
            heur_totals = results_df[heur_columns].sum().sort_values(ascending=False)
            summary_info['Heuristic method success statistics'] = {}
            for heur_type, count in heur_totals.items():
                if count > 0:
                    logging.info(f"  {heur_type}: {count}")
                    summary_info['Heuristic method success statistics'][heur_type] = int(count)
        
        # Calculate the average root node Gap
        if 'root_gap' in results_df.columns:
            mean_gap = results_df['root_gap'].dropna().mean()
            logging.info(f"Average root node Gap: {mean_gap:.2f}%")
            summary_info['Average root node Gap'] = float(f"{mean_gap:.2f}")
        
        # Save summary information to a JSON file
        import json
        summary_path = os.path.join(output_dir, "scip_solver_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_info, f, ensure_ascii=False, indent=4)
        logging.info(f"Summary information saved to {summary_path}")
        
        # Generate a summary text file for easier reading
        summary_txt_path = os.path.join(output_dir, "scip_solver_summary.txt")
        with open(summary_txt_path, 'w', encoding='utf-8') as f:
            f.write(f"SCIP Solver behavior analysis summary\n")
            f.write(f"====================================\n\n")
            f.write(f"Dataset path: {cfg.paths.instances_dir}\n")
            f.write(f"Evaluation time: {summary_info['Evaluation time']}\n")
            f.write(f"Instance count: {summary_info['Instance count']}\n")
            f.write(f"Solver: {summary_info['Solver']}\n\n")
            
            if 'Average root node Gap' in summary_info:
                f.write(f"Average root node Gap: {summary_info['Average root node Gap']}%\n\n")
            
            if 'Cut plane statistics' in summary_info:
                f.write(f"Cut plane statistics:\n")
                for cut_type, count in summary_info['Cut plane statistics'].items():
                    f.write(f"  {cut_type}: {count}\n")
                f.write("\n")
            
            if 'Heuristic method success statistics' in summary_info:
                f.write(f"Heuristic method success statistics:\n")
                for heur_type, count in summary_info['Heuristic method success statistics'].items():
                    f.write(f"  {heur_type}: {count}\n")
        
        logging.info(f"Summary text saved to {summary_txt_path}")
        logging.info(f"Summary: Successfully processed {len(results)} instances with SCIP")

def solve_with_scip(instance_file: str, log_file: str, cfg: DictConfig) -> Dict[str, Any]:
    """
    Solve a single instance using SCIP and return basic solve information.
    
    Args:
        instance_file: Path to the instance file
        log_file: Path to save the log file
        cfg: Configuration object
        
    Returns:
        Dictionary containing basic solve information
    """
    result = {}
    
    try:
        # Create SCIP settings file
        settings_content = create_scip_settings(cfg)
        
        # Create temporary settings file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.set', delete=False) as f:
            f.write(settings_content)
            settings_file = f.name
        
        try:
            # Construct SCIP command
            # SCIP command format: scip -f instance_file -l log_file -s settings_file -q
            cmd = [
                'scip',
                '-f', instance_file,
                '-l', log_file,
                '-s', settings_file,
                '-q'  # Quiet mode (reduce console output)
            ]
            
            # Run SCIP
            start_time = datetime.now()
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=cfg.solve.time_limit + 10  # Add 10 seconds buffer
            )
            end_time = datetime.now()
            
            # Calculate actual solve time
            actual_solve_time = (end_time - start_time).total_seconds()
            result['actual_solve_time'] = actual_solve_time
            
            # Check if SCIP ran successfully
            if process.returncode == 0:
                result['scip_status'] = 'Success'
            else:
                result['scip_status'] = 'Error'
                result['scip_error'] = process.stderr
                logging.warning(f"SCIP returned non-zero exit code for {instance_file}: {process.returncode}")
                logging.warning(f"SCIP stderr: {process.stderr}")
        
        finally:
            # Clean up temporary settings file
            if os.path.exists(settings_file):
                os.unlink(settings_file)
    
    except subprocess.TimeoutExpired:
        result['scip_status'] = 'Timeout'
        logging.warning(f"SCIP timeout for instance {instance_file}")
    except Exception as e:
        result['scip_status'] = 'Exception'
        result['scip_error'] = str(e)
        logging.error(f"Exception running SCIP on {instance_file}: {str(e)}")
    
    return result

def create_scip_settings(cfg: DictConfig) -> str:
    """
    Create SCIP settings content based on configuration.
    
    Args:
        cfg: Configuration object
        
    Returns:
        String containing SCIP settings
    """
    settings_lines = []
    
    # Time and resource limits
    settings_lines.append(f"limits/time = {cfg.scip.limits_time}")
    settings_lines.append(f"limits/gap = {cfg.scip.limits_gap}")
    
    # Threading and randomization
    settings_lines.append(f"parallel/maxnthreads = {cfg.scip.parallel_maxnthreads}")
    settings_lines.append(f"randomization/randomseedshift = {cfg.scip.randomization_randomseedshift}")
    
    # Presolving settings
    settings_lines.append(f"presolving/maxrounds = {cfg.scip.presolving_maxrounds}")
    settings_lines.append(f"presolving/maxrestarts = {cfg.scip.presolving_maxrestarts}")
    
    # Separation settings
    settings_lines.append(f"separating/maxrounds = {cfg.scip.separating_maxrounds}")
    settings_lines.append(f"separating/maxroundsroot = {cfg.scip.separating_maxroundsroot}")
    
    # Display settings
    settings_lines.append(f"display/verblevel = {cfg.scip.display_verblevel}")
    settings_lines.append(f"display/freq = {cfg.scip.display_freq}")
    
    # Heuristics emphasis
    if cfg.scip.heuristics_emphasis == "off":
        settings_lines.append("heuristics/emphasis = off")
    elif cfg.scip.heuristics_emphasis == "fast":
        settings_lines.append("heuristics/emphasis = fast")
    elif cfg.scip.heuristics_emphasis == "aggressive":
        settings_lines.append("heuristics/emphasis = aggressive")
    else:
        settings_lines.append("heuristics/emphasis = default")
    
    return '\n'.join(settings_lines)

if __name__ == "__main__":
    evaluate_scip_solver()
