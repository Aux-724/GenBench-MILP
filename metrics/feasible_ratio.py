"""Implementation of the Feasible Ratio metric.
"""

import os
import logging
import time
import numpy as np
from collections import Counter
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
            instance_results = self._parallel_check_feasibility_details(instances)
        else:
            # Sequential processing of instances
            logging.info("Using sequential calculation mode")
            
            try:
                # Create Gurobi environment and disable all output. If the
                # environment itself cannot start, preserve that as a solver
                # error for every requested instance.
                env = gp.Env(empty=True)
                env.setParam("OutputFlag", 0)
                env.start()
            except Exception as exc:
                logging.error(f"Error starting Gurobi environment: {exc}")
                instance_results = [
                    self._error_result(instance_path, exc, 0.0)
                    for instance_path in instances
                ]
            else:
                # Add progress bar
                for instance_path in tqdm(instances, desc="Checking instance feasibility and boundedness", ncols=100):
                    instance_results.append(self._check_feasibility_details(instance_path, env))
                env.dispose()

        feasible_count = sum(1 for result in instance_results if result["feasible"])
        status_counts = Counter(result.get("status", "UNKNOWN") for result in instance_results)
        error_counts = Counter(result.get("error", "") for result in instance_results if result.get("error"))
        
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
                "timeout_count": status_counts.get("TIME_LIMIT", 0),
                "error_count": sum(error_counts.values()),
                "status_counts": dict(status_counts),
                "error_counts": dict(error_counts),
                "instance_results": instance_results
            }
        }

    @staticmethod
    def _status_name(status_code: int) -> str:
        names = [
            "LOADED",
            "OPTIMAL",
            "INFEASIBLE",
            "INF_OR_UNBD",
            "UNBOUNDED",
            "CUTOFF",
            "ITERATION_LIMIT",
            "NODE_LIMIT",
            "TIME_LIMIT",
            "SOLUTION_LIMIT",
            "INTERRUPTED",
            "NUMERIC",
            "SUBOPTIMAL",
            "INPROGRESS",
            "USER_OBJ_LIMIT",
            "WORK_LIMIT",
            "MEM_LIMIT",
        ]
        status_names = {getattr(GRB, name): name for name in names if hasattr(GRB, name)}
        return status_names.get(status_code, f"STATUS_{status_code}")

    def _error_result(self, instance_path: str, error: Exception, runtime_sec: float) -> Dict[str, Any]:
        return {
            "instance": self._get_instance_name(instance_path),
            "path": instance_path,
            "feasible": False,
            "status": "ERROR",
            "status_code": "",
            "runtime_sec": round(runtime_sec, 6),
            "sol_count": "",
            "error": str(error),
        }

    @staticmethod
    def _safe_model_attr(model: gp.Model, attr: str, default: Any = "") -> Any:
        try:
            return getattr(model, attr)
        except Exception:
            return default

    def _check_feasibility_details(self, instance_path: str, env: Optional[gp.Env] = None) -> Dict[str, Any]:
        """
        Check feasibility and boundedness and preserve solver status details.
        """
        start_time = time.monotonic()
        owned_env = False
        model = None

        try:
            # If no environment is provided, create a new Gurobi environment
            if env is None:
                env = gp.Env(empty=True)
                env.setParam("OutputFlag", 0)
                env.start()
                owned_env = True

            # Load model using the created environment
            model = gp.read(instance_path, env=env)

            # Set solving parameters
            if self.time_limit is not None:
                model.setParam("TimeLimit", self.time_limit)
            model.setParam("Threads", self.threads)
            model.setParam("LogToConsole", 0)

            # Solve the model
            model.optimize()

            status_code = model.Status
            status = self._status_name(status_code)

            # The metric is feasible-and-bounded. A time limit with an incumbent
            # is still not certified bounded, so it is not counted as valid here.
            is_feasible = status_code in (GRB.OPTIMAL, GRB.SUBOPTIMAL)

            return {
                "instance": self._get_instance_name(instance_path),
                "path": instance_path,
                "feasible": is_feasible,
                "status": status,
                "status_code": status_code,
                "runtime_sec": round(time.monotonic() - start_time, 6),
                "sol_count": self._safe_model_attr(model, "SolCount", ""),
                "error": "",
            }

        except Exception as e:
            # Preserve solver/environment errors separately from mathematical
            # infeasibility so summaries cannot report license failures as 0%.
            logging.error(f"Error checking instance {instance_path}: {e}")
            return self._error_result(instance_path, e, time.monotonic() - start_time)

        finally:
            if model is not None:
                model.dispose()
            if owned_env and env is not None:
                env.dispose()

    def _check_feasibility(self, instance_path: str, env: Optional[gp.Env] = None) -> bool:
        """
        Check the feasibility and boundedness of an instance by solving it.

        Args:
            instance_path: Path to the instance file
            env: Optional Gurobi environment, if provided, use it, otherwise create a new one

        Returns:
            True if the instance is feasible and bounded, otherwise False
        """
        return self._check_feasibility_details(instance_path, env)["feasible"]
            
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

    def _parallel_check_feasibility_details(self, instances: List[str]) -> List[Dict[str, Any]]:
        """
        Parallel check feasibility while retaining per-instance solver statuses.
        """
        results = []

        # Dynamically adjust chunk size based on number of instances
        total_instances = len(instances)
        chunksize = max(1, min(100, total_instances // (self.n_jobs * 2)))

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures_iterator = executor.map(self._check_feasibility_details, instances, chunksize=chunksize)
            for result in tqdm(
                futures_iterator,
                total=len(instances),
                desc="Parallel checking instance feasibility and boundedness",
                ncols=100,
            ):
                results.append(result)

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
