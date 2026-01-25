"""
Gurobi callback function implementation module.

This module contains callback functions for capturing more detailed information
from the Gurobi optimization process, particularly for启发式方法 and branch rules.
"""

import os
import gurobipy as gp
from gurobipy import GRB
from typing import Dict, List, Any, Optional, Set

class HeuristicCallback:
    """
    Gurobi callback function for tracking Heuristic used in the optimization process.
    
    This callback records the number of times Heuristic is successfully used to find new solutions.
    It directly retrieves Heuristic information from the MIPSOL callback point, which is more accurate than log parsing.
    """
    
    def __init__(self, log_file_path: str = None):
        """
        Initialize Heuristic callback.
        
        Args:
            log_file_path: Log file path, used to append Heuristic information.
        """
        # Heuristic success count, records the number of times each method successfully finds new solutions
        self.heuristic_success_count = {}
        
        # Number of solutions found and best objective value
        self.solutions_found = 0
        self.best_objective = float('inf')
        self.last_solution_method = None
        
        # Log file path
        self.log_file_path = log_file_path
        
        # Initialize callback log
        if self.log_file_path:
            with open(self.log_file_path, 'w') as f:
                f.write("===== Gurobi Heuristic Callback Log =====\n")
                f.write("\nTime\t\tMethod\t\tObjective\n")
    
    def __call__(self, model, where):
        """
        Gurobi callback function.
        
        Args:
            model: Gurobi model object
            where: Callback trigger location code
        """
        try:
            # Record callback trigger location code (only first time or debug mode)
            if self.log_file_path and not hasattr(self, "_logged_where_codes"):
                self._logged_where_codes = set()
                with open(self.log_file_path, 'a') as f:
                    f.write(f"Callback trigger point code: {where}\n")
                self._logged_where_codes.add(where)
            
            # MIP solution found during the solving process
            if where == GRB.Callback.MIPSOL:
                self.solutions_found += 1
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                
                # Try to detect the source of the solution (Heuristic)
                try:
                    solution_method = model.cbGet(GRB.Callback.MIPSOL_METHOD)
                    method_name = self._get_method_name(solution_method)
                except Exception as e:
                    # If cannot get method information, use generic name
                    if self.log_file_path:
                        with open(self.log_file_path, 'a') as f:
                            f.write(f"Unable to get solution method information: {str(e)}\n")
                    method_name = "FoundHeuristic"
                
                # Update Heuristic success count
                self.heuristic_success_count[method_name] = self.heuristic_success_count.get(method_name, 0) + 1
                self.last_solution_method = method_name
                
                # Update best objective value
                if obj < self.best_objective:
                    self.best_objective = obj
                
                # Record to log file - open in append mode
                if self.log_file_path:
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(self.log_file_path, 'a') as f:
                        f.write(f"{timestamp}\t{method_name}\t{obj}\n")
            
            # Try to get information during MIP node processing
            elif where == GRB.Callback.MIPNODE:
                try:
                    # Try to get node information
                    nodecount = model.cbGet(GRB.Callback.MIPNODE_NODCNT)
                    if nodecount % 100 == 0 and self.log_file_path: # Reduce write frequency
                        with open(self.log_file_path, 'a') as f:
                            f.write(f"Processing node: {nodecount}\n")
                    
                    # Check if Heuristic was applied
                    if hasattr(model, "_last_heuristic") and model._last_heuristic:
                        heur_name = model._last_heuristic
                        self.heuristic_success_count[heur_name] = self.heuristic_success_count.get(heur_name, 0) + 1
                except Exception:
                    pass  # Ignore node processing errors
        
        except Exception as e:
            # Callback errors are usually swallowed by Gurobi, we need to record them
            if self.log_file_path:
                with open(self.log_file_path, 'a') as f:
                    f.write(f"CALLBACK ERROR: {str(e)}\n")
    
    def _get_method_name(self, method_code: int) -> str:
        """
        Convert Gurobi method code to method name.
        
        Args:
            method_code: Gurobi method code
            
        Returns:
            Method name
        """
        # Reference: https://www.gurobi.com/documentation/9.5/refman/cb_codes.html
        method_map = {
            GRB.FEASRELAX: "Feasibility Relaxation",
            1: "Primal Simplex",
            2: "Dual Simplex",
            3: "Barrier",
            4: "Concurrent",
            5: "Deterministic Concurrent",
            GRB.METHOD_AUTO: "Automatic",
            GRB.METHOD_PRIMAL: "Primal Simplex",
            GRB.METHOD_DUAL: "Dual Simplex",
            GRB.METHOD_BARRIER: "Barrier",
            GRB.METHOD_CONCURRENT: "Concurrent",
            GRB.METHOD_DETERMINISTIC: "Deterministic Concurrent",
            GRB.METHOD_NETWORK: "Network Simplex",
            GRB.METHOD_BENDERS: "Benders",
            # MIP methods
            GRB.MIPSOL_BRANCHING: "Branching",
            GRB.MIPSOL_MPEHEURISTIC: "MPE Heuristic",
            GRB.MIPSOL_NODLPOPT: "Node LP Optimization",
            GRB.MIPSOL_OBJBND: "Objective Bound",
            GRB.MIPSOL_OBJPRI: "Objective Priority",
            GRB.MIPSOL_OBJRNG: "Objective Range",
            GRB.MIPSOL_OBJVAL: "Objective Value",
            GRB.MIPSOL_SOLCNT: "Solution Count",
            GRB.MIPSOL_POOLCNT: "Pool Count",
            GRB.MIPSOL_POOLOBJVAL: "Pool Objective Value",
            GRB.MIPSOL_POOLOBJBOUND: "Pool Objective Bound",
            GRB.MIPSOL_RELAXATION: "Relaxation",
            GRB.MIPSOL_ROOTLPOBJBND: "Root LP Objective Bound",
            GRB.MIPSOL_MIP_SEARCH: "MIP Search",

            9: "RINS Heuristic",
            10: "Diving Heuristic",
            11: "Feasibility Pump",
            12: "Objective Enumeration",
            13: "Zero Objective Heuristic",
            14: "Local Branching Heuristic",
            15: "Sub-MIP Heuristic",
            16: "MIP Start",
            17: "Local Search",
            18: "Solution Improvement",
            19: "User Heuristic",
            20: "Proximity Heuristic",
            # Default value
            0: "Unknown Method"
        }
        
        # Get method name, return code value if unknown
        return method_map.get(method_code, f"Method Code {method_code}")
    
    def get_heuristic_stats(self) -> Dict[str, Any]:
        """
        Get Heuristic usage statistics.
        
        Returns:
            Dictionary containing Heuristic statistics
        """
        stats = {
            "success_count": self.heuristic_success_count,
            "solutions_found": self.solutions_found,
            "best_objective": self.best_objective,
            "last_solution_method": self.last_solution_method
        }
        return stats



