"""
HiGHS solver behavior metrics extraction module.

This module provides functionality to extract key solver behavior metrics from HiGHS log files, including:
1. Cut types and counts
2. Heuristic method success counts  
3. Root node Gap

These metrics can be used to compare the characteristics of different MILP instance sets and their impact on solver behavior.
"""

import os
import re
import glob
import logging
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from omegaconf import DictConfig

class HiGHSSolverInfoMetric:
    """HiGHS solver behavior metrics extraction and analysis class."""
    
    def __init__(self, cfg: DictConfig):
        """
        Initialize the HiGHS solver metrics extractor.
        
        Args:
            cfg: Hydra configuration object
        """
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        
        # HiGHS cut plane types mapping (based on actual HiGHS log format)
        # HiGHS reports total cuts in the Dynamic Constraints column
        self.highs_cut_types = {
            'total_cuts': 'cut_Total',  # HiGHS mainly reports total cuts
            'gomory': 'cut_Gomory',
            'mir': 'cut_MIR', 
            'clique': 'cut_Clique',
            'knapsack': 'cut_Knapsack',
            'implied_bounds': 'cut_ImpliedBounds',
            'flow_cover': 'cut_FlowCover',
            'lift_and_project': 'cut_LiftAndProject',
            'zero_half': 'cut_ZeroHalf',
            'disjunctive': 'cut_Disjunctive'
        }
        
        # HiGHS heuristic types mapping (based on actual HiGHS log format)
        # HiGHS uses single letter codes: J=Feasibility jump, R=Randomized rounding, etc.
        self.highs_heuristic_types = {
            'J': 'heur_FeasibilityJump',     # Feasibility jump
            'R': 'heur_RandomizedRounding',  # Randomized rounding
            'C': 'heur_CentralRounding',     # Central rounding
            'F': 'heur_FeasibilityPump',     # Feasibility pump
            'H': 'heur_Heuristic',           # General heuristic
            'L': 'heur_SubMIP',              # Sub-MIP
            'I': 'heur_Shifting',            # Shifting
            'Z': 'heur_ZIRound',             # ZI Round
            'S': 'heur_SolveLP',             # Solve LP (root relaxation)
            'total_heuristic_solutions': 'heur_Total'  # Total solutions found
        }
        
    def extract_solver_info(self, log_file: str, instance_name: str) -> Dict[str, Any]:
        """
        Extract solver behavior information from HiGHS log file.
        
        Args:
            log_file: Path to the HiGHS log file
            instance_name: Name of the instance
            
        Returns:
            Dictionary containing extracted solver information
        """
        if not os.path.exists(log_file):
            self.logger.error(f"Log file not found: {log_file}")
            return self._get_empty_metrics(instance_name)
            
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                log_content = f.read()
                
            # Extract basic solve information
            solve_info = self._extract_basic_solve_info(log_content, instance_name)
            
            # Extract cutting plane information
            cut_info = self._extract_cut_info(log_content)
            
            # Extract heuristic information
            heur_info = self._extract_heuristic_info(log_content)
            
            # Extract root node gap
            root_gap = self._extract_root_gap(log_content)
            
            # Combine all information
            result = {**solve_info, **cut_info, **heur_info, 'root_gap': root_gap}
            
            self.logger.info(f"Successfully extracted metrics for {instance_name}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error extracting solver info from {log_file}: {str(e)}")
            return self._get_empty_metrics(instance_name)
    
    def _extract_basic_solve_info(self, log_content: str, instance_name: str) -> Dict[str, Any]:
        """Extract basic solve information from HiGHS log."""
        info = {
            'instance': instance_name,
            'status': 'Unknown',
            'solve_time': 0.0,
            'nodes': 0,
            'objective': None,
            'gap': None
        }
        
        # Extract solve status from "Solving report" section
        status_match = re.search(r'Status\s+Optimal', log_content)
        if status_match:
            info['status'] = 'Optimal'
        elif re.search(r'Status\s+Feasible', log_content):
            info['status'] = 'Feasible'
        elif re.search(r'Status\s+Infeasible', log_content):
            info['status'] = 'Infeasible'
        elif re.search(r'Status\s+Time limit', log_content):
            info['status'] = 'Time limit'
        
        # Extract solve time from "Timing" section
        time_match = re.search(r'Timing\s+([\d.]+)\s+\(total\)', log_content)
        if time_match:
            info['solve_time'] = float(time_match.group(1))
        
        # Extract number of nodes from "Nodes" line
        nodes_match = re.search(r'Nodes\s+(\d+)', log_content)
        if nodes_match:
            info['nodes'] = int(nodes_match.group(1))
        
        # Extract objective value from "Primal bound" or final objective
        obj_match = re.search(r'Primal bound\s+([-+]?[\d.]+)', log_content)
        if not obj_match:
            obj_match = re.search(r'(\d+)\s+\(objective\)', log_content)
        if obj_match:
            info['objective'] = float(obj_match.group(1))
        
        # Extract MIP gap from "Gap" line
        gap_match = re.search(r'Gap\s+([\d.]+)%', log_content)
        if gap_match:
            info['gap'] = float(gap_match.group(1))
        
        return info
    
    def _extract_cut_info(self, log_content: str) -> Dict[str, int]:
        """Extract cutting plane information from HiGHS log."""
        cut_info = {}
        
        # Initialize all cut types to 0
        for cut_type in self.highs_cut_types.values():
            cut_info[cut_type] = 0
        
        # HiGHS reports cuts in the "Dynamic Constraints" column of the progress table
        # Look for lines with cut information in the progress table
        # Format: Src  Proc. InQueue |  Leaves   Expl. | BestBound  BestSol  Gap |   Cuts   InLp Confl. | LpIters  Time
        
        # Extract total cuts from the progress table
        cut_matches = re.findall(r'\s+\d+\s+\d+\s+\d+\s+[\d.]+%.*?\|\s+(\d+)\s+\d+\s+\d+\s+\|', log_content)
        if cut_matches:
            # Sum all cuts reported in the progress table
            total_cuts = sum(int(match) for match in cut_matches)
            cut_info['cut_Total'] = total_cuts
        
        # Look for cuts reported in the LP iterations section
        separation_match = re.search(r'(\d+)\s+\(separation\)', log_content)
        if separation_match:
            separation_cuts = int(separation_match.group(1))
            # If we don't have total cuts from progress table, use separation info
            if cut_info['cut_Total'] == 0:
                cut_info['cut_Total'] = separation_cuts
        
        # HiGHS doesn't typically provide detailed cut type breakdown in standard logs
        # So we mainly report total cuts
        
        return cut_info
    
    def _extract_heuristic_info(self, log_content: str) -> Dict[str, int]:
        """Extract heuristic information from HiGHS log."""
        heur_info = {}
        
        # Initialize all heuristic types to 0
        for heur_type in self.highs_heuristic_types.values():
            heur_info[heur_type] = 0
        
        # HiGHS reports heuristic activity in the progress table with single letter codes
        # Extract heuristic codes from the "Src" column of the progress table
        # Format: Src  Proc. InQueue |  Leaves   Expl. | BestBound  BestSol  Gap |   Cuts   InLp Confl. | LpIters  Time
        
        # Find all heuristic source codes in the progress table
        heur_matches = re.findall(r'^\s*([JRCFHLIZS])\s+\d+', log_content, re.MULTILINE)
        
        # Count each heuristic type
        for heur_code in heur_matches:
            if heur_code in self.highs_heuristic_types:
                standard_name = self.highs_heuristic_types[heur_code]
                heur_info[standard_name] += 1
        
        # Count total heuristic solutions by counting lines where BestSol changes
        # Look for lines in progress table where a new solution is found
        solution_lines = re.findall(r'^\s*[JRCFHLIZS].*?\|.*?\|.*?(\d+)\s+[\d.]+%', log_content, re.MULTILINE)
        
        # Count unique solution values (indicating new solutions found)
        if solution_lines:
            unique_solutions = set(solution_lines)
            heur_info['heur_Total'] = len(unique_solutions)
        
        # Alternative: count from LP iterations section
        heur_iter_match = re.search(r'(\d+)\s+\(heuristics\)', log_content)
        if heur_iter_match and heur_info['heur_Total'] == 0:
            # This indicates LP iterations spent on heuristics, not solution count
            # But it's an indicator of heuristic activity
            heur_iterations = int(heur_iter_match.group(1))
            if heur_iterations > 0:
                heur_info['heur_Total'] = 1  # At least some heuristic activity
        
        return heur_info
    
    def _extract_root_gap(self, log_content: str) -> float:
        """Extract root node gap from HiGHS log."""
        try:
            # HiGHS reports bounds in the progress table
            # Look for the first line of the progress table (root node)
            # Format: Src  Proc. InQueue |  Leaves   Expl. | BestBound  BestSol  Gap |   Cuts   InLp Confl. | LpIters  Time
            
            # Find the first progress line (root node)
            root_line_match = re.search(r'^\s*[JRCFHLIZS]\s+0\s+0\s+0\s+[\d.]+%\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%', 
                                       log_content, re.MULTILINE)
            
            if root_line_match:
                best_bound = float(root_line_match.group(1))
                best_sol = float(root_line_match.group(2))
                gap_percent = float(root_line_match.group(3))
                
                # Return the gap directly from the log
                return gap_percent
            
            # Alternative: look for dual and primal bounds in the solving report
            dual_bound_match = re.search(r'Dual bound\s+([\d.]+)', log_content)
            primal_bound_match = re.search(r'Primal bound\s+([\d.]+)', log_content)
            
            if dual_bound_match and primal_bound_match:
                dual_bound = float(dual_bound_match.group(1))
                primal_bound = float(primal_bound_match.group(1))
                
                # Calculate gap (for maximization problems, dual_bound >= primal_bound)
                if abs(dual_bound) > 1e-10:
                    gap = abs(primal_bound - dual_bound) / abs(dual_bound) * 100
                    return gap
            
            # Look for gap in the final report
            final_gap_match = re.search(r'Gap\s+([\d.]+)%', log_content)
            if final_gap_match:
                return float(final_gap_match.group(1))
            
            return 0.0
            
        except Exception as e:
            self.logger.warning(f"Error extracting root gap: {str(e)}")
            return 0.0
    
    def _get_empty_metrics(self, instance_name: str) -> Dict[str, Any]:
        """Return empty metrics dictionary for failed extractions."""
        metrics = {
            'instance': instance_name,
            'status': 'Failed',
            'solve_time': 0.0,
            'nodes': 0,
            'objective': None,
            'gap': None,
            'root_gap': 0.0
        }
        
        # Add empty cut metrics
        for cut_type in self.highs_cut_types.values():
            metrics[cut_type] = 0
        
        # Add empty heuristic metrics
        for heur_type in self.highs_heuristic_types.values():
            metrics[heur_type] = 0
        
        return metrics
    
    def process_instances(self, instances_dir: str, log_dir: str) -> pd.DataFrame:
        """
        Process multiple instances and extract solver metrics.
        
        Args:
            instances_dir: Directory containing instance files
            log_dir: Directory containing log files
            
        Returns:
            DataFrame containing extracted metrics for all instances
        """
        results = []
        
        # Get all instance files
        instance_files = []
        for ext in ['*.mps', '*.lp', '*.mps.gz', '*.lp.gz']:
            instance_files.extend(glob.glob(os.path.join(instances_dir, ext)))
        
        self.logger.info(f"Found {len(instance_files)} instances to process")
        
        for instance_file in instance_files:
            instance_name = os.path.splitext(os.path.basename(instance_file))[0]
            if instance_name.endswith('.mps'):
                instance_name = instance_name[:-4]
            
            log_file = os.path.join(log_dir, f"{instance_name}.log")
            
            if os.path.exists(log_file):
                metrics = self.extract_solver_info(log_file, instance_name)
                results.append(metrics)
            else:
                self.logger.warning(f"Log file not found for {instance_name}")
                results.append(self._get_empty_metrics(instance_name))
        
        return pd.DataFrame(results)
    
    def save_results(self, df: pd.DataFrame, output_path: str):
        """Save results to CSV file."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            self.logger.info(f"Results saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
    
    def create_summary_plots(self, df: pd.DataFrame, output_dir: str):
        """Create summary visualization plots."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Plot 1: Cut plane usage distribution
            cut_columns = [col for col in df.columns if col.startswith('cut_')]
            if cut_columns:
                plt.figure(figsize=(12, 8))
                cut_data = df[cut_columns].sum()
                cut_data = cut_data[cut_data > 0]  # Only show non-zero cuts
                
                if not cut_data.empty:
                    plt.bar(range(len(cut_data)), cut_data.values)
                    plt.xticks(range(len(cut_data)), cut_data.index, rotation=45)
                    plt.title('HiGHS Cutting Plane Usage Distribution')
                    plt.ylabel('Total Count')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'highs_cutplane_distribution.png'))
                    plt.close()
            
            # Plot 2: Heuristic success distribution
            heur_columns = [col for col in df.columns if col.startswith('heur_')]
            if heur_columns:
                plt.figure(figsize=(12, 8))
                heur_data = df[heur_columns].sum()
                heur_data = heur_data[heur_data > 0]  # Only show successful heuristics
                
                if not heur_data.empty:
                    plt.bar(range(len(heur_data)), heur_data.values)
                    plt.xticks(range(len(heur_data)), heur_data.index, rotation=45)
                    plt.title('HiGHS Heuristic Success Distribution')
                    plt.ylabel('Total Success Count')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'highs_heuristic_distribution.png'))
                    plt.close()
            
            # Plot 3: Root gap distribution
            if 'root_gap' in df.columns:
                plt.figure(figsize=(10, 6))
                valid_gaps = df[df['root_gap'] > 0]['root_gap']
                if not valid_gaps.empty:
                    plt.hist(valid_gaps, bins=30, alpha=0.7)
                    plt.title('HiGHS Root Node Gap Distribution')
                    plt.xlabel('Root Gap (%)')
                    plt.ylabel('Frequency')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'highs_root_gap_distribution.png'))
                    plt.close()
            
            self.logger.info(f"Summary plots saved to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error creating summary plots: {str(e)}")
