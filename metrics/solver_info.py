"""
Gurobi solver behavior metric parser.

This module extracts key solver behavior metrics from Gurobi log files, used to understand and analyze
the impact of different MILP instance sets on solver behavior.
"""

import os
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Any, Optional, Tuple
from omegaconf import DictConfig, OmegaConf

class SolverInfoMetric:
    """
    Gurobi solver behavior metric parser.
    
    Extracts key solver behavior metrics from Gurobi log files, used to understand and analyze
    the impact of different MILP instance sets on solver behavior.
    """
    
    def __init__(self, cfg: Optional[DictConfig] = None):
        """
        Initialize the Gurobi solver behavior metric parser.
        
        Args:
            cfg: Hydra configuration object, if None, use default configuration
        """
        self.name = "Gurobi Solver Behavior Metrics"
        self.params = {}
        
        # Get project root directory
        self.script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # If no configuration is provided, use default configuration
        if cfg is None:
            # Try to load configuration file using Hydra
            try:
                cfg = OmegaConf.load(os.path.join(self.script_dir, "configs/solver_info.yaml"))
            except:
                # If loading fails, create default configuration
                cfg = {
                    "solve": {
                        "time_limit": 100,
                        "threads": 1,
                        "mip_gap": 1e-4,
                        "seed": 42
                    },
                    "gurobi": {
                        "log_to_console": 0,
                        "output_flag": 1,
                        "presolve": -1,
                        "mip_focus": 0,
                        "heuristics": 0.5,
                        "cuts": -1
                    },
                    "paths": {
                        "instances_dir": os.path.join(self.script_dir, "data/raw/independent_set"),
                        "output_dir": os.path.join(self.script_dir, "outputs/solver_info")
                    },
                    "output": {
                        "save_results": True,
                        "result_filename": "solver_info.csv",
                        "save_plots": True,
                        "plot_filename": "solver_info_plot.png"
                    },
                    "logging": {
                        "level": "INFO",
                        "format": "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
                    }
                }
                cfg = OmegaConf.create(cfg)
        
        # Save configuration parameters
        self.params.update({
            "time_limit": cfg.solve.time_limit,
            "threads": cfg.solve.threads,
            "mip_gap": cfg.solve.mip_gap,
            "seed": cfg.solve.seed,
            "log_to_console": cfg.gurobi.log_to_console,
            "output_flag": cfg.gurobi.output_flag,
            "presolve": cfg.gurobi.presolve,
            "mip_focus": cfg.gurobi.mip_focus,
            "heuristics": cfg.gurobi.heuristics,
            "cuts": cfg.gurobi.cuts,
            "instances_dir": cfg.paths.instances_dir,
            "output_dir": cfg.paths.output_dir,
            "save_results": cfg.output.save_results,
            "result_filename": cfg.output.result_filename,
            "save_plots": cfg.output.save_plots,
            "plot_filename": cfg.output.plot_filename
        })
        
        # Set logging
        log_level = getattr(logging, cfg.logging.level)
        logging.basicConfig(
            level=log_level,
            format=cfg.logging.format
        )
        
        # Create output directory
        if self.params["save_results"] or self.params["save_plots"]:
            os.makedirs(self.params["output_dir"], exist_ok=True)
    
    def extract_cut_statistics(self, log_content: str) -> Dict[str, int]:
        """
        Extract cut statistics from Gurobi log content.
        
        Args:
            log_content: Gurobi log content
            
        Returns:
            Dictionary containing the number of each type of cut
        """
        cut_stats = {}
        
        # Find the cut statistics section
        cut_section_match = re.search(r"Cutting planes:(.*?)(?:Explored \d+ nodes|$)", log_content, re.DOTALL)
        if not cut_section_match:
            return cut_stats
        
        cut_section = cut_section_match.group(1)
        
        # Match the number of each type of cut
        cut_patterns = [
            (r"Gomory: (\d+)", "Gomory"),
            (r"Cover: (\d+)", "Cover"),
            (r"Implied bound: (\d+)", "ImpliedBound"),
            (r"MIR: (\d+)", "MIR"),
            (r"Flow cover: (\d+)", "FlowCover"),
            (r"Zero half: (\d+)", "ZeroHalf"),
            (r"RLT: (\d+)", "RLT"),
            (r"Relax-and-lift: (\d+)", "RelaxAndLift"),
            (r"Clique: (\d+)", "Clique"),
            (r"User: (\d+)", "User"),
            (r"StrongCG: (\d+)", "StrongCG"),
            (r"Network: (\d+)", "Network"),
            (r"Inf proof: (\d+)", "InfProof"),
            (r"Mod-K: (\d+)", "ModK")
        ]
        
        for pattern, name in cut_patterns:
            match = re.search(pattern, cut_section)
            if match:
                cut_stats[name] = int(match.group(1))
        
        return cut_stats
    
    def extract_heuristic_success(self, log_content: str) -> Dict[str, int]:
        """
        Extract heuristic success counts from Gurobi log content.
        
        Args:
            log_content: Gurobi log content
            
        Returns:
            Dictionary containing the number of each type of heuristic
        """
        heuristic_stats = {}
        
        # Define heuristic method keywords and names - more accurate matching
        heuristic_patterns = [
            # More accurate heuristic pre-defined types
            (r".*?RINS.*?", "RINS"),                                  # RINS Heuristic
            (r".*?[Ff]easibility [Pp]ump.*?", "FeasibilityPump"),   # Feasibility Pump
            (r".*?[Ll]ocal [Bb]ranching.*?", "LocalBranching"),      # Local Branching
            (r".*?MIP [Ss]tart.*?", "MIPStart"),                     # MIP Start
            (r".*?[Dd]iving.*?", "Diving"),                           # Diving
            (r".*?[Ss]ub.*?MIP.*?", "SubMIP"),                       # Sub-MIP
            (r".*?[Zz]ero.*?", "ZeroObjective"),                       # Zero Objective
            (r".*?[Ee]numeration.*?", "ObjectiveEnumeration"),        # Objective Enumeration
            (r".*?[Pp]roximity.*?", "Proximity"),                      # Proximity
            (r".*?[Cc]rossover.*?", "Crossover"),                      # Crossover
            (r".*?[Ff]ix.*?[Rr]elax.*?", "FixAndRelax"),              # Fix and Relax
            
            # Different types of solution discovery
            (r"Found heuristic solution", "FoundHeuristic"),           # Found Heuristic
            (r"Heur\. solution", "HeuristicSolution"),                 # Heuristic Solution
            (r"Improved solution", "ImprovedSolution"),                 # Improved Solution
            (r"Found incumbent", "FoundIncumbent"),                    # Found Incumbent
            (r"Solution improved", "PostProcessing"),                   # PostProcessing
            
            # Generic heuristic (last match)
            (r".*[Hh]euristic.*", "OtherHeuristic")                     # Other Heuristic
        ]
        
        # Initialize all heuristic method counts to 0
        for _, name in heuristic_patterns:
            heuristic_stats[name] = 0
            
        # Parse heuristic information from log file
        # Extract all lines containing heuristic or solution
        heuristic_lines = []
        for line in log_content.split('\n'):
            if re.search(r'[Hh]euristic|[Ss]olution|incumbent|RINS|Diving|Pump', line):
                heuristic_lines.append(line)
        
        # Match specific heuristic patterns in these lines
        # Use two-stage matching, first try to match heuristic name patterns precisely
        for line in heuristic_lines:
            matched = False
            # First try to match specific heuristic names
            for pattern, name in heuristic_patterns[:11]:  # First 11 are specific heuristic types
                if re.search(pattern, line, re.IGNORECASE):
                    heuristic_stats[name] += 1
                    matched = True
                    break
            
            # If no specific type match, try matching solution discovery
            if not matched:
                for pattern, name in heuristic_patterns[11:16]:  # Middle few are solution discovery
                    if re.search(pattern, line, re.IGNORECASE):
                        heuristic_stats[name] += 1
                        matched = True
                        break
            
            # If still no match, use generic heuristic type
            if not matched and re.search(heuristic_patterns[-1][0], line, re.IGNORECASE):
                heuristic_stats["OtherHeuristic"] += 1
        
        # If there is heuristic log but no heuristic method is identified
        if heuristic_lines and sum(heuristic_stats.values()) == 0:
            heuristic_stats["GenericHeuristic"] = len(heuristic_lines)
        
        return heuristic_stats
    
    def extract_root_gap(self, log_content: str) -> Optional[float]:
        """
        Extract root node gap information from Gurobi log content.
        
        Args:
            log_content: Gurobi log content
            
        Returns:
            Root node gap value, or None if extraction fails
        """
        # Extract root relaxation value
        root_relaxation_match = re.search(r"Root relaxation: objective ([-\d.eE+]+)", log_content)
        if not root_relaxation_match:
            return None
        
        root_relaxation = float(root_relaxation_match.group(1))
        
        # Extract final or best objective value
        obj_match = re.search(r"(?:Optimal objective|Best objective) ([-\d.eE+]+)", log_content)
        if not obj_match:
            # Try to find objective value from other locations
            obj_match = re.search(r"Best objective ([-\d.eE+]+).* gap", log_content)
            if not obj_match:
                return None
        
        obj_value = float(obj_match.group(1))
        
        # Calculate gap
        epsilon = 1e-10  # Prevent division by zero
        gap = abs(obj_value - root_relaxation) / (abs(obj_value) + epsilon)
        
        return gap * 100.0  # Convert to percentage
    
    def extract_branch_info(self, log_content: str) -> Dict[str, Any]:
        """
        Since the user requested to remove the branch node count metric, this method returns an empty dictionary for compatibility.
        
        Args:
            log_content: Gurobi log content
            
        Returns:
            An empty dictionary, no longer extracting branch node count
        """
        # Return an empty dictionary, no longer extracting branch node count
        return {}
    
    def analyze_log_file(self, log_file_path: str) -> Dict[str, Any]:
        """
        Analyze Gurobi log file and extract all key metrics.
        
        Args:
            log_file_path: Gurobi log file path
            
        Returns:
            Dictionary containing all extracted metrics
        """
        try:
            with open(log_file_path, 'r') as f:
                log_content = f.read()
            
            # Extract instance name (from log file name)
            instance_name = os.path.basename(log_file_path).replace('.log', '')
            
            # Extract all metrics
            cut_stats = self.extract_cut_statistics(log_content)
            heuristic_stats = self.extract_heuristic_success(log_content)
            root_gap = self.extract_root_gap(log_content)
            
            # Extract solution status and time
            status_match = re.search(r"Optimal solution found.*?Time: ([\d.]+)s", log_content, re.DOTALL)
            if status_match:
                status = "Optimal"
                solve_time = float(status_match.group(1))
            else:
                time_match = re.search(r"Time limit reached.*?Time: ([\d.]+)s", log_content, re.DOTALL)
                if time_match:
                    status = "TimeLimit"
                    solve_time = float(time_match.group(1))
                else:
                    status = "Unknown"
                    solve_time = None
            
            # Combine all metrics
            result = {
                "instance": instance_name,
                "status": status,
                "solve_time": solve_time,
                "root_gap": root_gap,
                **{f"cut_{k}": v for k, v in cut_stats.items()},
                **{f"heur_{k}": v for k, v in heuristic_stats.items()}
            }
            
            return result
        
        except Exception as e:
            logging.error(f"分析日志文件 {log_file_path} 时出错: {str(e)}")
            return {"instance": os.path.basename(log_file_path).replace('.log', ''), "error": str(e)}
    
    def plot_results(self, results_df: pd.DataFrame, output_path: str):
        """
        Plot result visualization charts.
        
        Args:
            results_df: DataFrame containing analysis results
            output_path: Chart output path
        """
        # Check if results are empty
        if results_df.empty:
            logging.warning("Results are empty, cannot generate charts")
            return
        
        # Create a 2x2 subplot layout
        fig, axs = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Gurobi Solver Behavior Metrics Analysis', fontsize=16)
        
        # 1. 割平面统计图
        cut_columns = [col for col in results_df.columns if col.startswith('cut_')]
        if cut_columns:
            cut_data = results_df[cut_columns].sum().sort_values(ascending=False)
            axs[0, 0].bar(cut_data.index, cut_data.values)
            axs[0, 0].set_title('Cut Usage Statistics')
            axs[0, 0].set_xlabel('Cut Type')
            axs[0, 0].set_ylabel('Application Count')
            axs[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Heuristic success count
        heur_columns = [col for col in results_df.columns if col.startswith('heur_')]
        if heur_columns:
            heur_data = results_df[heur_columns].sum().sort_values(ascending=False)
            axs[0, 1].bar(heur_data.index, heur_data.values)
            axs[0, 1].set_title('Heuristic Success Count')
            axs[0, 1].set_xlabel('Heuristic Method')
            axs[0, 1].set_ylabel('Success Count')
            axs[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Root gap distribution
        if 'root_gap' in results_df.columns:
            valid_gaps = results_df['root_gap'].dropna()
            if not valid_gaps.empty:
                axs[1, 0].hist(valid_gaps, bins=20)
                axs[1, 0].set_title('Root Gap Distribution')
                axs[1, 0].set_xlabel('Gap Value (%)')
                axs[1, 0].set_ylabel('Instance Count')
        
        # 4. Heuristic method distribution (replacing branch node count)
        heur_columns = [col for col in results_df.columns if col.startswith('heur_') and col != 'heuristic_data_source']
        
        if heur_columns:
            # Extract heuristic data
            heur_data = results_df[heur_columns].sum().sort_values(ascending=False)
            top_heurs = heur_data.head(10)
            axs[1, 1].barh(top_heurs.index, top_heurs.values)
            
            # Check data source
            if 'heuristic_data_source' in results_df.columns:
                # Calculate percentage of different data sources
                source_counts = results_df['heuristic_data_source'].value_counts()
                callback_percent = source_counts.get('callback', 0) / len(results_df) * 100
                log_percent = source_counts.get('log_parsing', 0) / len(results_df) * 100
                
                if callback_percent > 0 and log_percent > 0:
                    # Mixed data source
                    axs[1, 1].set_title(f'Heuristic Method Distribution (Callback: {callback_percent:.1f}%, Log: {log_percent:.1f}%)')
                elif callback_percent > 0:
                    # Only callback data
                    axs[1, 1].set_title('Heuristic Method Distribution (Callback Data)')
                else:
                    # Only log parsing data
                    axs[1, 1].set_title('Heuristic Method Distribution (Log Parsing Data)')
            else:
                axs[1, 1].set_title('Heuristic Method Distribution')
        
        if heur_columns:
            axs[1, 1].set_xlabel('Frequency of Use')
            axs[1, 1].tick_params(axis='y', labelsize=8)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path)
        logging.info(f"Saved charts to {output_path}")
        plt.close()
    
    def save_results(self, results_list: List[Dict[str, Any]], output_path: str):
        """
        Save analysis results to CSV file.
        
        Args:
            results_list: List of dictionaries containing analysis results
            output_path: CSV output path
        """
        # Convert to DataFrame
        results_df = pd.DataFrame(results_list)
        
        # Save CSV file
        results_df.to_csv(output_path, index=False)
        logging.info(f"Saved results to {output_path}")
        
        return results_df
    
    def get_gurobi_params(self) -> Dict[str, Any]:
        """
        Get Gurobi parameter settings.
        
        Returns:
            Gurobi parameter dictionary
        """
        return {
            "TimeLimit": self.params["time_limit"],
            "Threads": self.params["threads"],
            "MIPGap": self.params["mip_gap"],
            "Seed": self.params["seed"],
            "LogToConsole": self.params["log_to_console"],
            "OutputFlag": self.params["output_flag"],
            "Presolve": self.params["presolve"],
            "MIPFocus": self.params["mip_focus"],
            "Heuristics": self.params["heuristics"],
            "Cuts": self.params["cuts"]
        }
