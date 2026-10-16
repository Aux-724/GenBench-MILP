"""
Gurobi solver behavior metrics evaluation script.

This script uses Gurobi to solve a set of MILP instances and collects key solver behavior metrics, including:
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
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Optional
from tqdm import tqdm

# Add the project root directory to the system path to ensure the project modules can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the project metrics parsing module
from metrics.solver_info import SolverInfoMetric
from metrics.gurobi_callbacks import HeuristicCallback


def solve_instance_for_solver_info(
    instance_file: str,
    gurobi_logs_dir: str,
    gurobi_params: Dict[str, Any],
    gurobi_cfg: Dict[str, Any],
    metric_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Solve one instance and parse its Gurobi behavior log."""
    instance_name = os.path.basename(instance_file).split(".")[0]
    log_file = os.path.join(gurobi_logs_dir, f"{instance_name}.log")
    model = None
    env = None

    try:
        solver_metric = SolverInfoMetric(OmegaConf.create(metric_cfg))

        env = gp.Env(empty=True)
        for param, value in gurobi_params.items():
            env.setParam(param, value)
        env.start()

        model = gp.read(instance_file, env)
        model.setParam(GRB.Param.LogFile, log_file)
        model.setParam(GRB.Param.LogToConsole, gurobi_cfg["log_to_console"])
        model.setParam(GRB.Param.OutputFlag, gurobi_cfg["output_flag"])
        model.setParam(GRB.Param.LogFile, log_file)
        model.setParam(GRB.Param.Heuristics, gurobi_cfg["heuristics"])

        try:
            if hasattr(GRB.Param, "LogFileAppend"):
                model.setParam(GRB.Param.LogFileAppend, gurobi_cfg["log_file_append"])
        except Exception as exc:
            logging.debug(f"Failed to set LogFileAppend: {str(exc)}")

        try:
            if hasattr(GRB.Param, "DisplayInterval"):
                model.setParam(GRB.Param.DisplayInterval, gurobi_cfg["display_interval"])
        except Exception as exc:
            logging.debug(f"Failed to set DisplayInterval: {str(exc)}")

        callbacks_active = False
        heuristic_stats = {"success_count": {}}
        if gurobi_cfg.get("use_callback", False):
            try:
                callback_log_file = os.path.join(gurobi_logs_dir, f"{instance_name}_heuristics.log")
                heuristic_callback = HeuristicCallback(callback_log_file)
                model.optimize(heuristic_callback)
                callbacks_active = True
                heuristic_stats = heuristic_callback.get_heuristic_stats()

                if heuristic_stats["success_count"]:
                    logging.debug(
                        "Obtained %s heuristic success entries from the callback",
                        len(heuristic_stats["success_count"]),
                    )
                    for method, count in heuristic_stats["success_count"].items():
                        logging.debug("  - %s: %s successful calls", method, count)
            except Exception as exc:
                logging.error(f"Error executing callback function: {str(exc)}")
                callbacks_active = False

        if not callbacks_active:
            model.optimize()

        model.dispose()
        env.dispose()
        model = None
        env = None

        instance_result = solver_metric.analyze_log_file(log_file)

        if callbacks_active and heuristic_stats["success_count"]:
            for key in list(instance_result.keys()):
                if key.startswith("heur_"):
                    del instance_result[key]

            for method, count in heuristic_stats["success_count"].items():
                instance_result[f"heur_{method}"] = count

            instance_result["solutions_found"] = heuristic_stats["solutions_found"]
            if heuristic_stats["last_solution_method"]:
                instance_result["last_solution_method"] = heuristic_stats["last_solution_method"]
            instance_result["heuristic_data_source"] = "callback"
        else:
            instance_result["heuristic_data_source"] = "log_parsing"

        instance_result["instance"] = instance_name
        return instance_result

    except Exception as exc:
        logging.error(f"Error processing instance {instance_name}: {str(exc)}")
        return {
            "instance": instance_name,
            "error": str(exc),
        }
    finally:
        if model is not None:
            try:
                model.dispose()
            except Exception:
                pass
        if env is not None:
            try:
                env.dispose()
            except Exception:
                pass


@hydra.main(config_path="../configs", config_name="solver_info")
def evaluate_solver(cfg: DictConfig):
    """
    Use Gurobi to solve a set of MILP instances and collect key solver behavior metrics.
    
    Args:
        cfg: Hydra configuration object
    """
    # Set the log level
    log_level = getattr(logging, cfg.logging.level)
    logging.basicConfig(
        level=log_level,
        format=cfg.logging.format
    )
    
    logging.info("Initializing solver behavior metrics evaluation...")
    
    # Get the instance directory path
    instances_dir = cfg.paths.instances_dir
    logging.info(f"Instance directory: {instances_dir}")
    
    # Get the output directory path
    # First try to get it from cfg.hydra.run.dir
    try:
        if hasattr(cfg, 'hydra') and hasattr(cfg.hydra, 'run') and hasattr(cfg.hydra.run, 'dir'):
            output_dir = cfg.hydra.run.dir
        else:
            # Use the output directory defined in the configuration file as a backup, and add a timestamp
            # Create an output directory with a timestamp to ensure each experiment can be distinguished
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(cfg.paths.output_dir, f"experiment_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        # If any error occurs, use the output directory in the configuration and add a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(cfg.paths.output_dir, f"experiment_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Output directory: {output_dir}")
    
    # Create the Gurobi log directory
    gurobi_logs_dir = os.path.join(output_dir, "gurobi_logs")
    os.makedirs(gurobi_logs_dir, exist_ok=True)
    logging.info(f"Gurobi log directory: {gurobi_logs_dir}")
    
    # Get the instance file paths
    instance_files = []
    for ext in ["*.mps", "*.lp", "*.mps.gz", "*.lp.gz"]:
        instance_files.extend(glob.glob(os.path.join(instances_dir, ext)))
    instance_files = sorted(instance_files)
    
    if not instance_files:
        logging.error(f"No MPS or LP instance files found in directory {instances_dir}")
        return
    
    logging.info(f"Found {len(instance_files)} instance files")
    
    # Initialize the results list
    results = []
    
    # Initialize the SolverInfoMetric object, for later log parsing
    solver_metric = SolverInfoMetric(cfg)
    
    # Set the Gurobi core parameters
    gurobi_params = {
        "Threads": cfg.solve.threads,
        "Seed": cfg.solve.seed,
        "MIPGap": cfg.solve.mip_gap,
        "TimeLimit": cfg.solve.time_limit,
        "LogToConsole": cfg.gurobi.log_to_console,
        "OutputFlag": cfg.gurobi.output_flag,
        "Presolve": cfg.gurobi.presolve,
        "MIPFocus": cfg.gurobi.mip_focus,
        "Heuristics": cfg.gurobi.heuristics,
        "Cuts": cfg.gurobi.cuts
    }

    gurobi_cfg = OmegaConf.to_container(cfg.gurobi, resolve=True)
    metric_cfg = OmegaConf.to_container(cfg, resolve=True)
    metric_cfg["paths"]["output_dir"] = output_dir
    parallel_enabled = bool(OmegaConf.select(cfg, "solve.parallel.enabled", default=False))
    requested_workers = int(OmegaConf.select(cfg, "solve.parallel.processes", default=1))
    max_workers = max(1, min(requested_workers, len(instance_files), 150))

    logging.info("Starting to process instances...")
    if parallel_enabled and max_workers > 1:
        logging.info("Using %s parallel solver-info workers", max_workers)
        if int(cfg.solve.threads) != 1:
            logging.warning(
                "solve.threads=%s with %s workers can exceed the intended solver budget; "
                "GenBench-MILP comparison runs should use solve.threads=1.",
                cfg.solve.threads,
                max_workers,
            )
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_instance = {
                executor.submit(
                    solve_instance_for_solver_info,
                    instance_file,
                    gurobi_logs_dir,
                    gurobi_params,
                    gurobi_cfg,
                    metric_cfg,
                ): instance_file
                for instance_file in instance_files
            }
            for future in tqdm(
                as_completed(future_to_instance),
                total=len(future_to_instance),
                desc="Solving progress",
                ncols=100,
                colour="green",
            ):
                instance_file = future_to_instance[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    instance_name = os.path.basename(instance_file).split(".")[0]
                    logging.error(f"Worker failed for instance {instance_name}: {str(exc)}")
                    results.append({"instance": instance_name, "error": str(exc)})
    else:
        for instance_file in tqdm(instance_files, desc="Solving progress", ncols=100, colour="green"):
            results.append(
                solve_instance_for_solver_info(
                    instance_file,
                    gurobi_logs_dir,
                    gurobi_params,
                    gurobi_cfg,
                    metric_cfg,
                )
            )
    
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
        heur_columns = [col for col in results_df.columns if col.startswith('heur_') and col != 'heuristic_data_source']
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
        summary_path = os.path.join(output_dir, "solver_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_info, f, ensure_ascii=False, indent=4)
        logging.info(f"Summary information saved to {summary_path}")
        
        # Generate a summary text file for easier reading
        summary_txt_path = os.path.join(output_dir, "solver_summary.txt")
        with open(summary_txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Solver behavior analysis summary\n")
            f.write(f"=========================\n\n")
            f.write(f"Dataset path: {cfg.paths.instances_dir}\n")
            f.write(f"Evaluation time: {summary_info['Evaluation time']}\n")
            f.write(f"Instance count: {summary_info['Instance count']}\n\n")
            
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
        logging.info(f"Summary: Successfully processed {len(results)} instances")
    
    logging.info("Solver behavior metrics evaluation completed")

if __name__ == "__main__":
    evaluate_solver()
