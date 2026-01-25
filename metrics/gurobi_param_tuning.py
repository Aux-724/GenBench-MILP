"""
Gurobi hyperparameter tuning metrics implementation.

This module provides a generic function for using SMAC to optimize Gurobi solver parameters, supporting:
1. Using training set for parameter tuning
2. Using test set to evaluate tuning parameters
3. Comparing tuning results from different instance sets
4. Support for SMAC v1.x (SMAC4HPO)

"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import glob
import pickle
from pathlib import Path
from tqdm import tqdm
import statistics

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario

import gurobipy as gp
from gurobipy import GRB


class GurobiParamTuning:
    """
    Gurobi hyperparameter tuning class.
    
    Uses SMAC3 to optimize Gurobi solver parameters, evaluating the impact of different source instance sets on tuning performance.
    """
    
    def __init__(self, cfg: Optional[DictConfig] = None):
        """
        Initialize Gurobi hyperparameter tuning class.
        
        Args:
            cfg: Hydra configuration object, if None, use default configuration
        """
        self.name = "Gurobi Parameter Tuning"
        self.params = {}
        
        # If no configuration is provided, use default configuration
        if cfg is None:
            # Try to load configuration file using Hydra
            try:
                cfg = OmegaConf.load(os.path.join(os.getcwd(), "configs/gurobi_param_tuning.yaml"))
            except:
                # If loading fails, create default configuration
                cfg = {
                    "tuning": {
                        "n_trials": 200,
                        "random_seed": 42,
                        "smac_output_dir": os.path.join(os.getcwd(), "outputs/smac3")
                    },
                    "solve": {
                        "time_limit": 300,
                        "threads": 1
                    },
                    "paths": {
                        "training_instances_dir": os.path.join(os.getcwd(), "data/raw/independent_set"),
                        "generated_instances_dir": os.path.join(os.getcwd(), "data/acm-milp_mis_eta0.1"),
                        "test_instances_dir": os.path.join(os.getcwd(), "data/gurobi_test/independent_set"),
                        "output_dir": os.path.join(os.getcwd(), "outputs/gurobi_tuning")
                    },
                    "param_space": {
                        "Heuristics": {"type": "float", "range": [0.0, 1.0]},
                        "MIPFocus": {"type": "integer", "values": [0, 1, 2, 3]},
                        "VarBranch": {"type": "integer", "values": [-1, 0, 1, 2, 3]},
                        "BranchDir": {"type": "integer", "values": [-1, 0, 1]},
                        "Presolve": {"type": "integer", "values": [-1, 0, 1, 2]},
                        "PrePasses": {"type": "integer", "values": list(range(-1, 21))},
                        "Cuts": {"type": "integer", "values": [-1, 0, 1, 2, 3]},
                        "Method": {"type": "integer", "values": [-1, 0, 1, 2, 3, 4, 5]}
                    },
                    "output": {
                        "save_results": True,
                        "result_filename": "gurobi_tuning_results.json",
                        "save_plots": True,
                        "plot_filename": "gurobi_tuning_plot.png"
                    },
                    "logging": {
                        "level": "INFO",
                        "format": "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
                    }
                }
                cfg = OmegaConf.create(cfg)
        
        # Save configuration parameters
        self.params["n_trials"] = cfg.tuning.n_trials
        self.params["random_seed"] = cfg.tuning.random_seed
        self.params["smac_output_dir"] = cfg.tuning.smac_output_dir
        self.params["time_limit"] = cfg.solve.time_limit
        self.params["threads"] = cfg.solve.threads
        # Handle necessary paths
        self.params["test_instances_dir"] = cfg.paths.test_instances_dir
        
        # Handle optional paths
        if hasattr(cfg.paths, "training_instances_dir"):
            self.params["training_instances_dir"] = cfg.paths.training_instances_dir
        else:
            self.params["training_instances_dir"] = None
            
        if hasattr(cfg.paths, "generated_instances_dir"):
            self.params["generated_instances_dir"] = cfg.paths.generated_instances_dir
        else:
            self.params["generated_instances_dir"] = None
            
        # Handle custom instance set path
        if hasattr(cfg.paths, "custom_instances_dir"):
            self.params["custom_instances_dir"] = cfg.paths.custom_instances_dir
        self.params["output_dir"] = cfg.paths.output_dir
        self.params["param_space"] = cfg.param_space
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
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.params["save_results"] or self.params["save_plots"]:
            # Add timestamp subdirectory to output directory
            self.params["output_dir"] = os.path.join(self.params["output_dir"], f"run_{timestamp}")
            self.params["smac_output_dir"] = os.path.join(self.params["smac_output_dir"], f"run_{timestamp}")
            os.makedirs(self.params["output_dir"], exist_ok=True)
            os.makedirs(self.params["smac_output_dir"], exist_ok=True)
            logging.info(f"Results will be saved to: {self.params['output_dir']}")
    
    def _get_instances_from_dir(self, directory: str) -> List[str]:
        """
        Get a list of MILP instance file paths from a directory.
        
        Args:
            directory: Directory path
            
        Returns:
            List of MILP instance file paths
        """
        if not os.path.exists(directory):
            logging.warning(f"Directory does not exist: {directory}")
            return []
        
        # Support multiple common MILP file formats
        extensions = ["*.mps", "*.lp", "*.mps.gz", "*.lp.gz"]
        instance_files = []
        
        for ext in extensions:
            pattern = os.path.join(directory, ext)
            instance_files.extend(glob.glob(pattern))
        
        return sorted(instance_files)
    
    def _solve_instance_with_params(self, instance_path: str, params: Dict[str, Any], seed: int = 42) -> Tuple[float, float, str]:
        """
        Solve a MILP instance with given parameters and return solve time.
        
        Args:
            instance_path: Path to the MILP instance file
            params: Dictionary of Gurobi parameters
            seed: Random seed used for reproducibility
            
        Returns:
            (solve time, objective value, status) tuple
        """
        try:
            # Create Gurobi environment and model
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)  # Close output
            env.setParam("TimeLimit", self.params["time_limit"])
            env.setParam("Threads", self.params["threads"])
            env.setParam("Seed", seed)  # Set random seed for reproducibility
            
            # Set hyperparameters
            for param_name, param_value in params.items():
                try:
                    env.setParam(param_name, param_value)
                except Exception as e:
                    logging.warning(f"Unable to set parameter {param_name}={param_value}: {e}")
            
            env.start()
            model = gp.read(instance_path, env=env)
            
            # Solve model
            start_time = time.time()
            model.optimize()
            solve_time = time.time() - start_time
            
            # Get results
            status_map = {
                GRB.OPTIMAL: "OPTIMAL",
                GRB.INFEASIBLE: "INFEASIBLE",
                GRB.UNBOUNDED: "UNBOUNDED",
                GRB.INF_OR_UNBD: "INF_OR_UNBD",
                GRB.TIME_LIMIT: "TIME_LIMIT"
            }
            
            status = status_map.get(model.status, f"UNKNOWN({model.status})")
            obj_value = model.objVal if model.status == GRB.OPTIMAL or (model.status == GRB.TIME_LIMIT and model.SolCount > 0) else float('inf')
            
            return solve_time, obj_value, status
        except Exception as e:
            logging.error(f"Error solving instance {instance_path}: {str(e)}")
            return float('inf'), float('inf'), f"ERROR: {str(e)}"
    
    def _create_config_space(self) -> ConfigurationSpace:
        """
        Create SMAC configuration space, supporting multiple parameter types
        
        Returns:
            SMAC configuration space object
        """
        cs = ConfigurationSpace(seed=self.params["random_seed"])
        
        # Add hyperparameters
        param_space = self.params["param_space"]
        
        # Handle PrePasses - use discrete options instead of continuous integers, more suitable and efficient
        if "PrePasses" in param_space:
            param_config = param_space["PrePasses"]
            if param_config["type"] == "integer":
                cs.add_hyperparameter(CategoricalHyperparameter(
                    "PrePasses", 
                    choices=list(range(-1, 21))
                ))
        
        # Handle Heuristics - use float hyperparameter for better precision
        if "Heuristics" in param_space:
            param_config = param_space["Heuristics"]
            if param_config["type"] == "float":
                cs.add_hyperparameter(UniformFloatHyperparameter(
                    "Heuristics", 
                    lower=param_config["range"][0], 
                    upper=param_config["range"][1]
                ))
        
        # Handle other parameters (categorical or integer variables)
        for param_name, param_config in param_space.items():
            if param_name in ["Heuristics", "PrePasses"]:
                continue
                
            if param_config["type"] == "integer":
                # For most integer parameters, using categorical variables is more suitable
                cs.add_hyperparameter(CategoricalHyperparameter(
                    param_name, 
                    choices=param_config["values"]
                ))
            elif param_config["type"] == "float":
                # For float parameters
                cs.add_hyperparameter(UniformFloatHyperparameter(
                    param_name, 
                    lower=param_config["range"][0], 
                    upper=param_config["range"][1]
                ))
        
        return cs
    
    def _create_objective_function(self, instances: List[str], repeat: int = 1) -> callable:
        """
        Create SMAC objective function
        
        Args:
            instances: List of instance file paths for evaluation
            repeat: Number of times to solve each instance, to improve result stability
            
        Returns:
            Objective function (for SMAC optimization)
        """
        def objective_function(config, seed=None):
            # Convert configuration to dictionary and ensure parameter types are correct
            config_dict = {}
            for param_name, param_value in config.items():
                if param_name == "Heuristics" or param_name.endswith("Value"):
                    # Float parameters
                    config_dict[param_name] = float(param_value)
                else:
                    # Integer parameters
                    config_dict[param_name] = int(param_value)
            
            # Calculate average solve time for all instances
            all_times = []
            
            # Use tqdm to create progress bar
            progress_bar = tqdm(
                instances, 
                desc=f"Evaluating configuration", 
                unit="instance",
                leave=False
            )
            
            for instance_path in progress_bar:
                instance_times = []
                
                for run in range(repeat):
                    # Use different seed for each run
                    run_seed = (self.params["random_seed"] + run) if seed is None else (seed + run)
                    
                    # Use configuration to solve instance
                    solve_time, _, _ = self._solve_instance_with_params(
                        instance_path, 
                        config_dict,
                        seed=run_seed
                    )
                    instance_times.append(solve_time)
                
                # Calculate average time for this instance
                if repeat > 1:
                    avg_time = statistics.mean(instance_times)
                else:
                    avg_time = instance_times[0]
                
                all_times.append(avg_time)
            
            # Return mean solve time across all instances
            mean_time = statistics.mean(all_times)
            return mean_time
            
        return objective_function
    
    def tune_parameters(self, scenario_name: str, instance_files: List[str]) -> Dict[str, Any]:
        """
        Use SMAC1.x (SMAC4HPO) to tune Gurobi parameters
        """
        config_space = self._create_config_space()
        scenario_dir = os.path.join(self.params["smac_output_dir"], scenario_name)
        os.makedirs(scenario_dir, exist_ok=True)
        # Create SMAC scenario configuration
        scenario_config = {
            "run_obj": "quality",
            "runcount-limit": self.params["n_trials"],
            "cs": config_space,
            "deterministic": "false",
            "output_dir": self.params["smac_output_dir"],
            "n_jobs": 64  # Use 64 parallel jobs to speed up evaluation, suitable for 128-core system
        }
        
        logging.info(f"SMAC parallel setting: Using {scenario_config['n_jobs']} parallel jobs")
        scenario = Scenario(scenario_config)
        objective_function = self._create_objective_function(instance_files)
        logging.info(f"Begin {scenario_name} scenario parameter tuning...")
        print(f"\n{'-'*50}")
        print(f"Begin {scenario_name} scenario parameter tuning (up to {self.params['n_trials']} evaluations)")
        print(f"{'-'*50}")
        # Create SMAC object and optimize
        smac = SMAC4HPO(
            scenario=scenario,
            rng=np.random.RandomState(self.params["random_seed"]),
            tae_runner=objective_function,
            run_id=0  # Set run ID
        )
        
        logging.info("Begin parallel parameter tuning, please wait...")
        incumbent = smac.optimize()
        best_config = incumbent.get_dictionary()
        for param in best_config:
            if param != "Heuristics":
                best_config[param] = int(best_config[param])
            else:
                best_config[param] = float(best_config[param])
        logging.info(f"{scenario_name} scenario best configuration: {best_config}")
        config_path = os.path.join(scenario_dir, "best_config.json")
        with open(config_path, 'w') as f:
            json.dump(best_config, f, indent=4)
        return best_config
    
    def evaluate_config(self, config_name: str, config: Dict[str, Any], test_instances: List[str], repeat: int = 1) -> Dict[str, Any]:
        """
        Evaluate parameter configuration on test instances
        
        Args:
            config_name: Configuration name
            config: Parameter configuration dictionary
            test_instances: List of test instance file paths
            repeat: Number of times to evaluate
            
        Returns:
            Evaluation results dictionary
        """
        logging.info(f"Begin evaluating configuration {config_name}...")
        
        all_results = []
        for run in range(repeat):
            logging.info(f"Running {run+1}/{repeat}...")
            
            # Solve each test instance (using tqdm progress bar)
            instance_results = []
            solve_times = []
            
            # Use tqdm to create progress bar
            progress_bar = tqdm(
                test_instances, 
                desc=f"Evaluating configuration {config_name} (run {run+1}/{repeat})", 
                unit="instance"
            )
            
            for instance_path in progress_bar:
                solve_time, obj_value, status = self._solve_instance_with_params(instance_path, config)
                
                # Record results
                instance_results.append({
                    "instance": os.path.basename(instance_path),
                    "solve_time": solve_time,
                    "obj_value": obj_value,
                    "status": status
                })
                
                if solve_time < float('inf'):
                    solve_times.append(solve_time)
                    
                # Update progress bar description, show current average solve time
                current_mean = np.mean(solve_times) if solve_times else 0.0
                progress_bar.set_postfix({"Average time": f"{current_mean:.4f}s"})
            
            # Calculate statistics
            mean_time = np.mean(solve_times) if solve_times else float('inf')
            std_time = np.std(solve_times) if solve_times else 0
            
            # Add to results list
            all_results.append({
                "run": run + 1,
                "mean_solve_time": mean_time,
                "std_solve_time": std_time,
                "instance_results": instance_results
            })
        
        # Calculate overall statistics
        mean_times = [r["mean_solve_time"] for r in all_results]
        overall_mean = np.mean(mean_times)
        overall_std = np.std(mean_times)
        
        evaluation_result = {
            "config_name": config_name,
            "config": config,
            "overall_mean_time": overall_mean,
            "overall_std_time": overall_std,
            "run_results": all_results
        }
        
        logging.info(f"Evaluation of configuration {config_name} completed, average solve time: {overall_mean:.4f}s")
        
        return evaluation_result
    
    def run(self, tuning_mode: str = "both", repeat: int = 1, evaluate_on_test: bool = True) -> Dict[str, Any]:
        """
        Run hyperparameter tuning experiment
        
        Args:
            tuning_mode: Tuning mode, optional:
                - "training": Use training set for tuning
                - "generated": Use generated set for tuning
                - "both": Use both training and generated sets for tuning
                - "custom": Use custom instance set for tuning (specified by params["custom_instances_dir"])
            repeat: Number of evaluations
            evaluate_on_test: Whether to evaluate on test set
            
        Returns:
            Experiment result dictionary
        """
        start_time = datetime.now()
        logging.info(f"Begin Gurobi parameter tuning task, time: {start_time}")
        
        # Step 1: Get instance files
        # Get training instances (if configured)
        training_instances = []
        if self.params["training_instances_dir"]:
            training_instances = self._get_instances_from_dir(self.params["training_instances_dir"])
            logging.info(f"Found {len(training_instances)} training instances")
        
        # Get generated instances (if configured)
        generated_instances = []
        if self.params["generated_instances_dir"]:
            generated_instances = self._get_instances_from_dir(self.params["generated_instances_dir"])
            logging.info(f"Found {len(generated_instances)} generated instances")
        
        # Step 2: Get test instances
        test_instances = self._get_instances_from_dir(self.params["test_instances_dir"])
        logging.info(f"Found {len(test_instances)} test instances")
        
        # Step 3: Get custom instances (if configured)
        custom_instances = []
        if "custom_instances_dir" in self.params and self.params["custom_instances_dir"]:
            custom_instances = self._get_instances_from_dir(self.params["custom_instances_dir"])
            logging.info(f"Found {len(custom_instances)} Instances")
        logging.info(f"Found {len(generated_instances)} Instances")
        logging.info(f"Found {len(test_instances)} Instances")
        
        # Step 4: Execute hyperparameter tuning
        if tuning_mode == "training":
            # Use training set for tuning
            config_train = self.tune_parameters("train_scenario", training_instances)
            best_config = config_train
        elif tuning_mode == "generated":
            # Use generated set for tuning
            config_generated = self.tune_parameters("generated_scenario", generated_instances)
            best_config = config_generated
        elif tuning_mode == "both":
            # Use both training and generated sets for tuning
            config_train = self.tune_parameters("train_scenario", training_instances)
            config_generated = self.tune_parameters("generated_scenario", generated_instances)
            best_config = config_train
        elif tuning_mode == "custom":
            # Use custom instance set for tuning
            if "custom_instances_dir" not in self.params or not self.params["custom_instances_dir"]:
                logging.error("Custom instance directory not specified")
                return {"error": "Custom instance directory not specified"}
                
            config_custom = self.tune_parameters("custom_scenario", custom_instances)
            best_config = config_custom
        
        # Step 5: Get default configuration
        config_default = {
            "Heuristics": 0.05,  # Gurobi default value
            "MIPFocus": 0, 
            "VarBranch": -1,
            "BranchDir": 0,
            "Presolve": -1,
            "PrePasses": -1,
            "Cuts": -1,
            "Method": -1
        }
        
        # Step 6: Evaluate all configurations on test set
        evaluation_results = {}
        
        # Evaluate default configuration
        evaluation_results["Default"] = self.evaluate_config(
            "Default", config_default, test_instances, repeat
        )
        
        # Evaluate best configuration
        evaluation_results["Best"] = self.evaluate_config(
            "Best", best_config, test_instances, repeat
        )
        
        # Calculate performance improvement percentage (relative to default configuration)
        default_time = evaluation_results["Default"]["overall_mean_time"]
        
        for config_name in ["Best"]:
            config_time = evaluation_results[config_name]["overall_mean_time"]
            if default_time > 0 and config_time > 0:
                improvement = (default_time - config_time) / default_time * 100
                evaluation_results[config_name]["improvement_percent"] = improvement
        
        # Generate result summary
        summary = {
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_training_instances": len(training_instances),
            "n_generated_instances": len(generated_instances),
            "n_test_instances": len(test_instances),
            "tuning_trials": self.params["n_trials"],
            "evaluation_repeats": repeat,
            "configs": {
                "Default": config_default,
                "Best": best_config
            },
            "evaluation_results": {
                name: {
                    "overall_mean_time": result["overall_mean_time"],
                    "overall_std_time": result["overall_std_time"],
                    "improvement_percent": result.get("improvement_percent", 0)
                }
                for name, result in evaluation_results.items()
            }
        }
        
        # Save results (using timestamp directory)
        if self.params["save_results"]:
            # Save summary
            summary_path = os.path.join(self.params["output_dir"], "summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=4)
            
            # Save detailed evaluation results
            results_path = os.path.join(self.params["output_dir"], self.params["result_filename"])
            with open(results_path, 'w') as f:
                json.dump(evaluation_results, f, indent=4)
                
            print(f"\nResults saved to: {self.params['output_dir']}")
        
        # Draw result charts
        if self.params["save_plots"]:
            self._plot_results(evaluation_results)
        
        logging.info("Gurobi parameter tuning task completed")
        
        return summary
    
    def _plot_results(self, evaluation_results: Dict[str, Any]) -> None:
        """
        Draw evaluation result charts
        
        Args:
            evaluation_results: Evaluation results dictionary
        """
        # Extract data
        config_names = list(evaluation_results.keys())
        mean_times = [evaluation_results[name]["overall_mean_time"] for name in config_names]
        std_times = [evaluation_results[name]["overall_std_time"] for name in config_names]
        
        # Calculate improvement percentage
        default_time = evaluation_results["Default"]["overall_mean_time"]
        improvements = []
        for name in config_names:
            if name == "Default":
                improvements.append(0)
            else:
                config_time = evaluation_results[name]["overall_mean_time"]
                if default_time > 0 and config_time > 0:
                    improvement = (default_time - config_time) / default_time * 100
                else:
                    improvement = 0
                improvements.append(improvement)
        
        # Create chart
        plt.figure(figsize=(15, 10))
        
        # Mean solve time bar chart
        plt.subplot(2, 1, 1)
        bars = plt.bar(config_names, mean_times, yerr=std_times, capsize=5)
        
        # Add value labels
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{mean_times[i]:.2f}s",
                    ha='center', va='bottom')
        
        plt.title("Comparison of Average Solve Time")
        plt.ylabel("Solve Time (s)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Improvement percentage bar chart
        plt.subplot(2, 1, 2)
        bars = plt.bar(config_names[1:], improvements[1:])
        
        # Add value labels
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{improvements[i+1]:.2f}%",
                    ha='center', va='bottom')
        
        plt.title("Performance Improvement Percentage Relative to Default Configuration")
        plt.ylabel("Improvement Percentage (%)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save chart
        plt.tight_layout()
        plt.savefig(os.path.join(self.params["output_dir"], self.params["plot_filename"]))
        plt.close()


@hydra.main(config_path="../configs", config_name="gurobi_param_tuning", version_base=None)
def main(cfg: DictConfig):
    """
    Main function, runs Gurobi parameter tuning task using Hydra
    
    Args:
        cfg: Hydra configuration object
    """
    # Print configuration information
    print(OmegaConf.to_yaml(cfg))
    
    # Create tuning object
    tuner = GurobiParamTuning(cfg)
    
    # Run tuning
    results = tuner.run(
        tuning_mode=cfg.tuning.mode, 
        repeat=cfg.evaluation.repeat,
        evaluate_on_test=cfg.evaluation.test_evaluation
    )
    
    # Print summary
    if "improvements" in results:
        print("\nPerformance Improvement Summary:")
        for config_name, improvement in results["improvements"].items():
            print(f"{config_name}: {improvement:.2f}%")
    
    return results
if __name__ == "__main__":
    main()
