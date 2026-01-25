"""
SCIP solver behavior metrics extraction module.

This module provides functionality to extract key solver behavior metrics from SCIP log files, including:
1. Cut types and counts
2. Heuristic method success counts  
3. Root node Gap

These metrics can be used to compare the characteristics of different MILP instance sets and their impact on solver behavior.
"""

import os
import re
import logging
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from omegaconf import DictConfig

class SCIPSolverInfoMetric:
    """SCIP solver behavior metrics extraction and analysis class."""
    
    def __init__(self, cfg: DictConfig):
        """
        Initialize the SCIP solver metrics extractor.
        
        Args:
            cfg: Hydra configuration object
        """
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        
        # SCIP cut plane types mapping
        self.scip_cut_types = {
            'gomory': 'cut_Gomory',
            'strongcg': 'cut_StrongCG', 
            'cmir': 'cut_CMIR',
            'flowcover': 'cut_FlowCover',
            'clique': 'cut_Clique',
            'knapsackcover': 'cut_KnapsackCover',
            'oddcycle': 'cut_OddCycle',
            'zerohalf': 'cut_ZeroHalf',
            'mcf': 'cut_MCF',
            'disjunctive': 'cut_Disjunctive',
            'impliedbounds': 'cut_ImpliedBounds',
            'redcost': 'cut_RedCost'
        }
        
        # SCIP heuristic types mapping (based on actual SCIP log output)
        self.scip_heuristic_types = {
            'alns': 'heur_ALNS',
            'bound': 'heur_Bound',
            'clique': 'heur_Clique',
            'coefdiving': 'heur_CoefDiving',
            'completesol': 'heur_CompleteSol',
            'conflictdiving': 'heur_ConflictDiving',
            'crossover': 'heur_Crossover',
            'dins': 'heur_DINS',
            'distributiondivin': 'heur_DistributionDiving',
            'dualval': 'heur_DualVal',
            'farkasdiving': 'heur_FarkasDiving',
            'feaspump': 'heur_FeasPump',
            'fixandinfer': 'heur_FixAndInfer',
            'fracdiving': 'heur_FracDiving',
            'gins': 'heur_GINS',
            'guideddiving': 'heur_GuidedDiving',
            'indicator': 'heur_Indicator',
            'intdiving': 'heur_IntDiving',
            'intshifting': 'heur_IntShifting',
            'linesearchdiving': 'heur_LinesearchDiving',
            'localbranching': 'heur_LocalBranching',
            'locks': 'heur_Locks',
            'lpface': 'heur_LPFace',
            'mpec': 'heur_MPEC',
            'multistart': 'heur_Multistart',
            'mutation': 'heur_Mutation',
            'nlpdiving': 'heur_NLPDiving',
            'objpscostdiving': 'heur_ObjPscostDiving',
            'octane': 'heur_Octane',
            'oneopt': 'heur_OneOpt',
            'proximity': 'heur_Proximity',
            'pscostdiving': 'heur_PscostDiving',
            'randrounding': 'heur_RandRounding',
            'rens': 'heur_RENS',
            'reoptsols': 'heur_ReoptSols',
            'repair': 'heur_Repair',
            'rins': 'heur_RINS',
            'rootsoldiving': 'heur_RootsolDiving',
            'rounding': 'heur_Rounding',
            'shiftandpropagate': 'heur_ShiftAndPropagate',
            'shifting': 'heur_Shifting',
            'simplerounding': 'heur_SimpleRounding',
            'subnlp': 'heur_SubNLP',
            'trivial': 'heur_Trivial',
            'trysol': 'heur_TrySol',
            'twoopt': 'heur_TwoOpt',
            'undercover': 'heur_Undercover',
            'vbounds': 'heur_VBounds',
            'veclendiving': 'heur_VeclendDiving',
            'zeroobj': 'heur_ZeroObj',
            'zirounding': 'heur_ZiRounding'
        }
    
    def analyze_log_file(self, log_file_path: str) -> Dict[str, Any]:
        """
        Analyze a SCIP log file and extract solver behavior metrics.
        
        Args:
            log_file_path: Path to the SCIP log file
            
        Returns:
            Dictionary containing extracted metrics
        """
        if not os.path.exists(log_file_path):
            self.logger.error(f"Log file not found: {log_file_path}")
            return {}
        
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                log_content = f.read()
            
            metrics = {}
            
            # Extract basic solve information
            metrics.update(self._extract_solve_info(log_content))
            
            # Extract root node gap
            metrics.update(self._extract_root_gap(log_content))
            
            # Extract cutting plane statistics
            metrics.update(self._extract_cutting_planes(log_content))
            
            # Extract heuristic statistics
            metrics.update(self._extract_heuristics(log_content))
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing log file {log_file_path}: {str(e)}")
            return {}
    
    def _extract_solve_info(self, log_content: str) -> Dict[str, Any]:
        """Extract basic solving information from SCIP log."""
        info = {}
        
        # Extract solving status
        if "optimal solution found" in log_content.lower():
            info['status'] = 'Optimal'
        elif "time limit reached" in log_content.lower():
            info['status'] = 'TimeLimit'
        elif "memory limit reached" in log_content.lower():
            info['status'] = 'MemoryLimit'
        else:
            info['status'] = 'Unknown'
        
        # Extract solving time
        time_pattern = r"solving time:\s+(\d+\.\d+)"
        time_match = re.search(time_pattern, log_content, re.IGNORECASE)
        if time_match:
            info['solve_time'] = float(time_match.group(1))
        
        return info
    
    def _extract_root_gap(self, log_content: str) -> Dict[str, Any]:
        """Extract root node gap from SCIP log with improved robustness."""
        gap_info = {}
        
        try:
            # Multiple patterns for root LP relaxation information
            root_lp_patterns = [
                r"First LP value\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
                r"Root LP:\s+objective\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
                r"root LP solution:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
                r"Root relaxation:\s+objective\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
                r"Root LP Estimate\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
            ]
            
            root_obj = None
            for pattern in root_lp_patterns:
                root_match = re.search(pattern, log_content, re.IGNORECASE)
                if root_match:
                    root_obj = float(root_match.group(1))
                    self.logger.debug(f"Found root LP objective: {root_obj} using pattern: {pattern}")
                    break
            
            # Multiple patterns for final/best objective value
            final_obj_patterns = [
                r"Primal Bound\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
                r"Dual Bound\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
                r"Final Dual Bound\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
                r"primal solution:\s+objective\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
                r"objective value:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
                r"best solution:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
                r"Best solution found:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
            ]
            
            final_obj = None
            for pattern in final_obj_patterns:
                final_match = re.search(pattern, log_content, re.IGNORECASE)
                if final_match:
                    final_obj = float(final_match.group(1))
                    self.logger.debug(f"Found final objective: {final_obj} using pattern: {pattern}")
                    break
            
            # Calculate root gap if both values are found
            if root_obj is not None and final_obj is not None:
                # Handle different optimization directions
                # For minimization: gap = (root_obj - final_obj) / |final_obj|
                # For maximization: gap = (final_obj - root_obj) / |final_obj|
                
                # Determine optimization direction from problem sense or objective values
                is_minimization = True  # Default assumption
                
                # Try to detect optimization direction from log
                if "maximize" in log_content.lower() or "maximization" in log_content.lower():
                    is_minimization = False
                elif "minimize" in log_content.lower() or "minimization" in log_content.lower():
                    is_minimization = True
                # If root_obj > final_obj significantly, likely minimization
                elif abs(root_obj - final_obj) > 1e-6 and root_obj > final_obj:
                    is_minimization = True
                elif abs(root_obj - final_obj) > 1e-6 and final_obj > root_obj:
                    is_minimization = False
                
                # Calculate gap based on optimization direction
                epsilon = 1e-10
                if is_minimization:
                    # For minimization: root_obj <= final_obj (root is lower bound)
                    gap = abs(root_obj - final_obj) / (abs(final_obj) + epsilon) * 100
                else:
                    # For maximization: final_obj <= root_obj (root is upper bound)
                    gap = abs(final_obj - root_obj) / (abs(final_obj) + epsilon) * 100
                
                gap_info['root_gap'] = gap
                gap_info['root_objective'] = root_obj
                gap_info['final_objective'] = final_obj
                gap_info['optimization_direction'] = 'minimization' if is_minimization else 'maximization'
                
                self.logger.debug(f"Root LP objective: {root_obj}, Final objective: {final_obj}, "
                                f"Direction: {'min' if is_minimization else 'max'}, Gap: {gap:.2f}%")
            else:
                missing = []
                if root_obj is None:
                    missing.append("root LP objective")
                if final_obj is None:
                    missing.append("final objective")
                self.logger.warning(f"Could not extract root gap: missing {', '.join(missing)}")
                
        except Exception as e:
            self.logger.error(f"Error extracting root gap: {str(e)}")
        
        return gap_info
    
    def _extract_cutting_planes(self, log_content: str) -> Dict[str, Any]:
        """Extract cutting plane statistics from SCIP log."""
        cut_stats = {}
        
        try:
            # Initialize all cut types to 0
            for scip_name, standard_name in self.scip_cut_types.items():
                cut_stats[standard_name] = 0
            
            # Look for separator statistics section
            # SCIP shows separator statistics like:
            # "Separators         :   ExecTime  SetupTime      Calls  RootCalls    Cutoffs    DomReds       Cuts    Applied      Conss"
            # "  gomory           :       0.01       0.00         45         1          0          0        123        123          0"
            
            separator_section = False
            cuts_column_index = None
            lines = log_content.split('\n')
            
            for line in lines:
                line = line.strip()
                
                # Check if we're in the separator statistics section and parse header
                if "Separators" in line and "ExecTime" in line:
                    separator_section = True
                    # Parse header to find the "Cuts" column index dynamically
                    header_parts = line.split()
                    try:
                        cuts_column_index = header_parts.index("Cuts") - 2  # Adjust for the colon split
                        self.logger.debug(f"Found 'Cuts' column at relative index {cuts_column_index}")
                    except ValueError:
                        self.logger.warning("Could not find 'Cuts' column in separator header")
                        cuts_column_index = 4  # Fallback to default index (adjusted)
                    continue
                elif separator_section and line.startswith("Constraints"):
                    # End of separator section
                    break
                
                if separator_section and line and cuts_column_index is not None:
                    # Parse separator statistics line
                    # Format: "  separator_name   :   time1  time2  calls1  calls2  ...  cuts  applied  ..."
                    parts = line.split(':')
                    if len(parts) >= 2:
                        separator_name = parts[0].strip().lower()
                        stats = parts[1].strip().split()
                        
                        if separator_name in self.scip_cut_types and len(stats) > cuts_column_index:
                            try:
                                # Use the dynamically determined cuts column index
                                cuts_generated = int(stats[cuts_column_index])
                                if cuts_generated > 0:
                                    cut_stats[self.scip_cut_types[separator_name]] = cuts_generated
                                    self.logger.debug(f"Found {cuts_generated} cuts for {separator_name}")
                            except (ValueError, IndexError):
                                continue
            
            # Alternative parsing: look for cutting plane summary
            # Some SCIP versions show: "cuts applied: gomory 123, strongcg 45, ..."
            cuts_applied_pattern = r"cuts applied:([^\\n]+)"
            cuts_match = re.search(cuts_applied_pattern, log_content, re.IGNORECASE)
            if cuts_match:
                cuts_text = cuts_match.group(1)
                # Parse individual cut counts
                for scip_name, standard_name in self.scip_cut_types.items():
                    cut_pattern = rf"{scip_name}\s+(\d+)"
                    cut_match = re.search(cut_pattern, cuts_text, re.IGNORECASE)
                    if cut_match:
                        count = int(cut_match.group(1))
                        cut_stats[standard_name] = max(cut_stats.get(standard_name, 0), count)
        
        except Exception as e:
            self.logger.error(f"Error extracting cutting planes: {str(e)}")
        
        # Remove zero entries to keep output clean
        cut_stats = {k: v for k, v in cut_stats.items() if v > 0}
        
        return cut_stats
    
    def _extract_heuristics(self, log_content: str) -> Dict[str, Any]:
        """Extract heuristic statistics from SCIP log."""
        heur_stats = {}
        
        try:
            # Initialize all heuristic types to 0
            for scip_name, standard_name in self.scip_heuristic_types.items():
                heur_stats[standard_name] = 0
            
            # Look for primal heuristics section
            # SCIP shows heuristic statistics like:
            # "Primal Heuristics  :   ExecTime  SetupTime      Calls      Found       Best"
            # "  rins             :       0.12       0.00         15          3          1"
            
            heuristic_section = False
            found_column_index = None
            lines = log_content.split('\n')
            
            for line in lines:
                line = line.strip()
                
                # Check if we're in the primal heuristics section and parse header
                if "Primal Heuristics" in line and "ExecTime" in line:
                    heuristic_section = True
                    # Parse header to find the "Found" column index dynamically
                    header_parts = line.split()
                    try:
                        found_column_index = header_parts.index("Found") - 2  # Adjust for the colon split
                        self.logger.debug(f"Found 'Found' column at relative index {found_column_index}")
                    except ValueError:
                        self.logger.warning("Could not find 'Found' column in heuristics header")
                        found_column_index = 3  # Fallback to default index
                    continue
                elif heuristic_section and (line.startswith("Separators") or line.startswith("Constraints")):
                    # End of heuristics section
                    break
                
                if heuristic_section and line and found_column_index is not None:
                    # Parse heuristic statistics line
                    parts = line.split(':')
                    if len(parts) >= 2:
                        heur_name = parts[0].strip().lower()
                        stats = parts[1].strip().split()
                        
                        if heur_name in self.scip_heuristic_types and len(stats) > found_column_index:
                            try:
                                # Use the dynamically determined found column index
                                solutions_found = int(stats[found_column_index])
                                if solutions_found > 0:
                                    heur_stats[self.scip_heuristic_types[heur_name]] = solutions_found
                                    self.logger.debug(f"Found {solutions_found} solutions from {heur_name}")
                            except (ValueError, IndexError):
                                continue
            
            # Also count total heuristic solutions found
            total_heur_solutions = sum(heur_stats.values())
            if total_heur_solutions > 0:
                heur_stats['heur_TotalFound'] = total_heur_solutions
        
        except Exception as e:
            self.logger.error(f"Error extracting heuristics: {str(e)}")
        
        # Remove zero entries to keep output clean
        heur_stats = {k: v for k, v in heur_stats.items() if v > 0}
        
        return heur_stats
    
    def plot_results(self, results_df: pd.DataFrame, output_path: str):
        """
        Generate visualization plots for the solver behavior analysis results.
        
        Args:
            results_df: DataFrame containing the analysis results
            output_path: Path to save the plot
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('SCIP Solver Behavior Analysis Results', fontsize=16)
            
            # Plot 1: Root Gap distribution
            if 'root_gap' in results_df.columns:
                ax1 = axes[0, 0]
                results_df['root_gap'].dropna().hist(bins=20, ax=ax1, alpha=0.7, color='skyblue')
                ax1.set_title('Root Node Gap Distribution')
                ax1.set_xlabel('Root Gap (%)')
                ax1.set_ylabel('Frequency')
                ax1.grid(True, alpha=0.3)
            
            # Plot 2: Cutting planes usage
            ax2 = axes[0, 1]
            cut_columns = [col for col in results_df.columns if col.startswith('cut_')]
            if cut_columns:
                cut_totals = results_df[cut_columns].sum().sort_values(ascending=False)
                if len(cut_totals) > 0:
                    cut_totals.plot(kind='bar', ax=ax2, color='lightcoral')
                    ax2.set_title('Cutting Planes Usage')
                    ax2.set_xlabel('Cut Types')
                    ax2.set_ylabel('Total Count')
                    ax2.tick_params(axis='x', rotation=45)
                    ax2.grid(True, alpha=0.3)
            
            # Plot 3: Heuristic success
            ax3 = axes[1, 0]
            heur_columns = [col for col in results_df.columns if col.startswith('heur_')]
            if heur_columns:
                heur_totals = results_df[heur_columns].sum().sort_values(ascending=False)
                if len(heur_totals) > 0:
                    heur_totals.head(10).plot(kind='bar', ax=ax3, color='lightgreen')  # Show top 10
                    ax3.set_title('Top 10 Heuristic Success Counts')
                    ax3.set_xlabel('Heuristic Types')
                    ax3.set_ylabel('Success Count')
                    ax3.tick_params(axis='x', rotation=45)
                    ax3.grid(True, alpha=0.3)
            
            # Plot 4: Solve time distribution
            ax4 = axes[1, 1]
            if 'solve_time' in results_df.columns:
                results_df['solve_time'].dropna().hist(bins=20, ax=ax4, alpha=0.7, color='gold')
                ax4.set_title('Solve Time Distribution')
                ax4.set_xlabel('Solve Time (seconds)')
                ax4.set_ylabel('Frequency')
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Results plot saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating plots: {str(e)}")
