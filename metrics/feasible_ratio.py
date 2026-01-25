"""Implementation of the Feasible Ratio metric.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import gurobipy as gp
from gurobipy import GRB
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed


class FeasibleRatioMetric:
    """
    Calculate the feasible and bounded ratio metric.
    
    This metric measures the percentage of generated MILP instances that are both feasible and bounded.
    It quantifies the ratio of instances that satisfy all constraints and have a finite optimal solution.
    """
    
    def __init__(self, time_limit: int = 300, threads: int = 64, parallel: bool = True, n_jobs: int = None):
        """
        Initialize the feasible ratio metric.
        
        Args:
            time_limit: Time limit for solving each instance (in seconds)
            threads: Number of threads used for solving
            parallel: Whether to enable parallel processing
            n_jobs: Number of parallel jobs, set to None for CPU core count - 1
        """
        self.time_limit = time_limit
        self.threads = threads
        self.parallel = parallel
        self.n_jobs = n_jobs if n_jobs is not None else max(1, mp.cpu_count() - 1)
        self.name = "Feasible Ratio"
    
    def _get_instance_name(self, instance_path: str) -> str:
        """
        Extract instance name from instance path.
        
        Args:
            instance_path: Full path to the instance file
            
        Returns:
            Instance file name with extension (e.g., instance_1.lp)
        """
        return os.path.basename(instance_path)
    
    def calculate(self, instances: List[str]) -> Dict[str, Any]:
        """
        Calculate the feasible and bounded ratio metric.
        
        Args:
            instances: List of instance file paths
            
        Returns:
            Dictionary containing the feasible and bounded ratio metric results
        """
        total_count = len(instances)
        instance_results = []
        
        if self.parallel and total_count > 1:
            # Parallel processing of instances
            logging.info(f"Using parallel calculation mode, number of parallel tasks: {self.n_jobs}")
            # Use process pool for parallel processing
            results = self._parallel_check_feasibility(instances)
            
            # Merge results
            feasible_count = sum(1 for result in results if result[1])
            instance_results = [
                {
                    "instance": self._get_instance_name(instance_path),
                    "path": instance_path,
                    "feasible": is_feasible
                }
                for instance_path, is_feasible in results
            ]
        else:
            # Sequential processing of instances
            logging.info("Using sequential calculation mode")
            feasible_count = 0
            
            # Create Gurobi environment and disable all output
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.start()
            
            # Add progress bar
            for instance_path in tqdm(instances, desc="Checking instance feasibility and boundedness", ncols=100):
                # Check the feasibility of the instance
                is_feasible = self._check_feasibility(instance_path, env)
                instance_name = self._get_instance_name(instance_path)
                
                if is_feasible:
                    feasible_count += 1
                
                instance_results.append({
                    "instance": instance_name,
                    "path": instance_path,
                    "feasible": is_feasible
                })
        
        # Calculate feasibility ratio
        if total_count > 0:
            feasibility_ratio = (feasible_count / total_count) * 100
        else:
            feasibility_ratio = 0.0
        
        # Return results
        return {
            "name": self.name,
            "value": feasibility_ratio,
            "details": {
                "feasible_count": feasible_count,
                "total_count": total_count,
                "instance_results": instance_results
            }
        }
    
    def _check_feasibility(self, instance_path: str, env: Optional[gp.Env] = None) -> bool:
        """
        Check the feasibility and boundedness of an instance by solving it.
        
        Args:
            instance_path: Path to the instance file
            env: Optional Gurobi environment, if provided, use it, otherwise create a new one
            
        Returns:
            True if the instance is feasible and bounded, otherwise False
        """
        try:
            # If no environment is provided, create a new Gurobi environment
            if env is None:
                env = gp.Env(empty=True)
                env.setParam("OutputFlag", 0)
                env.start()
            
            # Load model using the created environment
            model = gp.read(instance_path, env=env)
            
            # Set solving parameters
            if self.time_limit is not None:
                model.setParam("TimeLimit", self.time_limit)
            model.setParam("Threads", self.threads)
            model.setParam("LogToConsole", 0)
            
            # Solve the model
            model.optimize()
            
            # Get the status code
            status = model.Status
            
            # Determine feasibility based on status code
            # Status code description:
            # GRB.OPTIMAL (2) - Found optimal solution
            # GRB.INFEASIBLE (3) - Problem has no solution
            # GRB.UNBOUNDED (5) - Problem is unbounded
            # GRB.SUBOPTIMAL (12) - Found suboptimal solution
            # GRB.TIME_LIMIT (9) - Reached time limit
            
            is_feasible = False
            
            if status == GRB.OPTIMAL:
                is_feasible = True
            elif status == GRB.INFEASIBLE or status == GRB.INF_OR_UNBD:
                is_feasible = False
            elif status == GRB.UNBOUNDED:
                # Unbounded problem although has feasible solution, but now requires "feasible and bounded", so consider it as not feasible
                is_feasible = False
            elif status == GRB.SUBOPTIMAL:
                # Found suboptimal solution, problem is feasible
                is_feasible = True
            elif status == GRB.TIME_LIMIT:
                # Time limit reached, cannot determine boundedness, strictly consider it as not feasible
                is_feasible = False
            
            return is_feasible
        
        except Exception as e:
            # Record specific error information
            logging.error(f"Error checking instance {instance_path}: {e}")
            return False
            
    def _parallel_check_feasibility(self, instances: List[str]) -> List[Tuple[str, bool]]:
        """
        Parallel check the feasibility of multiple instances.
        
        Args:
            instances: List of instance file paths
            
        Returns:
            List of tuples containing instance paths and feasibility results
        """
        results = []
        
        # Dynamically adjust chunk size based on number of instances
        total_instances = len(instances)
        chunksize = max(1, min(100, total_instances // (self.n_jobs * 2)))
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # Use map method and apply chunksize for better performance
            futures_iterator = executor.map(self._check_feasibility, instances, chunksize=chunksize)
            
            # Keep results in the same order as input
            results_iter = zip(instances, tqdm(futures_iterator, total=len(instances), desc="Parallel checking instance feasibility and boundedness", ncols=100))
            
            for instance_path, is_feasible in results_iter:
                try:
                    results.append((instance_path, is_feasible))
                except Exception as e:
                    logging.error(f"Error checking instance {instance_path}: {e}")
                    results.append((instance_path, False))
        
        return results
    
    @staticmethod
    def from_config(cfg: dict) -> 'FeasibleRatioMetric':
        """
        Create a feasible ratio metric instance from configuration
        
        Args:
            cfg: Configuration dictionary
        
        Returns:
            Feasible ratio metric instance
        """
        time_limit = cfg.get("time_limit", 300)
        threads = cfg.get("threads", 64)
        parallel = cfg.get("parallel", True)
        n_jobs = cfg.get("n_jobs", None)
        
        return FeasibleRatioMetric(
            time_limit=time_limit,
            threads=threads,
            parallel=parallel,
            n_jobs=n_jobs
        )
