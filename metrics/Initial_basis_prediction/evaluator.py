"""
MILP初始基预测GNN模型评估器
"""
import os
import time
import json
import torch
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import scipy.linalg as la
import logging
import warnings
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union, Any

# Filter PyTorch's FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)
# Filter SciPy's LinAlgWarning
warnings.filterwarnings("ignore", category=la.LinAlgWarning)

from .data_utils import MILPInstance, BipartiteGraphFeatureExtractor
from .model import get_basis_from_probs

logger = logging.getLogger(__name__)

class BasisRepair:
    """Initial basis repair tool, used to repair invalid initial basis and generate valid LP initial basis"""
    
    @staticmethod
    def repair_basis(A, B_x, B_s, n_vars, n_constrs, var_probs, constr_probs, 
                     pivot_threshold=1e-12, max_iterations=50):
        """
        Repair invalid initial basis and generate valid LP initial basis
        
        Algorithm flow:
        1. Construct candidate basis matrix B = [A_Bx | -I_Bs]
        2. Try LU factorization on B
        3. Check for small pivot elements, remove corresponding columns
        4. If basis size < m, add remaining candidates in order of probability
        5. Repeat steps 2-4 until success or maximum iterations reached
        
        Args:
            A: constraint matrix
            B_x: initial basis variable indices
            B_s: initial basis slack indices
            n_vars: number of variables
            n_constrs: number of constraints
            var_probs: variable probabilities [n_vars, 3]
            constr_probs: constraint probabilities [n_constrs, 3]
            pivot_threshold: pivot threshold, small pivot elements are considered invalid (default 1e-10)
            max_iterations: maximum number of iterations (default 10)
            
        Returns:
            Tuple: ( repaired basis variable indices, repaired basis slack indices )
        """
        # Record initial state and parameters
        logger.debug(f"Begin to repair initial basis: variable count={n_vars}, constraint count={n_constrs}, "
                  f"initial basis variables={len(B_x)}, initial basis slacks={len(B_s)}, "
                  f"pivot threshold={pivot_threshold}")
        # Only output information at DEBUG level
        logger.debug(f"Repairing initial basis (maximum iterations: {max_iterations})...")
        
        # Convert sparse matrix to dense matrix for easier operations
        A_dense = A.toarray()
        
        # Copy initial basis to avoid modifying original lists
        current_B_x = list(B_x)
        current_B_s = list(B_s)
        
        # Check if initial basis is already valid
        try:
            # Build basis matrix [A_Bx | -I_Bs]
            B_matrix = np.zeros((n_constrs, len(current_B_x) + len(current_B_s)))
            
            # Fill variable part
            for i, var_idx in enumerate(current_B_x):
                B_matrix[:, i] = A_dense[:, var_idx]
            
            # Fill slack part (negative identity matrix)
            for i, constr_idx in enumerate(current_B_s):
                B_matrix[constr_idx, len(current_B_x) + i] = -1
            
            # Try LU factorization
            lu, piv = la.lu_factor(B_matrix)
            
            # Check if pivot elements are large enough
            diag_lu = np.abs(np.diag(lu))
            small_pivots = np.where(diag_lu < pivot_threshold)[0]
            
            if len(small_pivots) == 0:
                logger.info("Initial basis is already valid, no need to repair")
                return current_B_x, current_B_s
                
            logger.debug(f"Initial basis detected {len(small_pivots)} small pivot elements, need to repair")
            
        except Exception as e:
            logger.warning(f"Failed to check initial basis: {str(e)}")
            
        # Prepare candidate list for all variables and slacks, including their basis probabilities
        all_candidates = []
        
        # Create sets of selected variables and slacks for quick lookup
        selected_vars = set(current_B_x)
        selected_constrs = set(current_B_s)
        
        # Add all variables and their basis probabilities
        for i in range(n_vars):
            if i not in selected_vars:  # Only add variables that haven't been selected yet
                all_candidates.append((i, 'var', var_probs[i, 1].item()))
        
        # Add all slack variables and their basis probabilities
        for i in range(n_constrs):
            if i not in selected_constrs:  # Only add slacks that haven't been selected yet
                all_candidates.append((i, 'constr', constr_probs[i, 1].item()))
        
        # Sort by probability in descending order
        all_candidates.sort(key=lambda x: x[2], reverse=True)
        logger.debug(f"Candidate list has {len(all_candidates)} candidate variables")        
        
        # Record success flag for repairing
        repair_success = False
        
        # Start iteration repair loop
        for iteration in range(max_iterations):
            # Only record process information at DEBUG level
            logger.debug(f"Repairing iteration: {iteration+1}/{max_iterations}")
            
            # If current basis is invalid, rebuild basis
            try:
                # Build current basis matrix [A_Bx | -I_Bs]
                B_matrix = np.zeros((n_constrs, len(current_B_x) + len(current_B_s)))
                
                # Fill variable part
                for i, var_idx in enumerate(current_B_x):
                    B_matrix[:, i] = A_dense[:, var_idx]
                
                # Fill slack part (negative identity matrix)
                for i, constr_idx in enumerate(current_B_s):
                    B_matrix[constr_idx, len(current_B_x) + i] = -1
                
                # Record current basis state
                logger.debug(f"Current basis state: |B_x|={len(current_B_x)}, |B_s|={len(current_B_s)}, "
                            f"basis matrix size={B_matrix.shape}")
                
                # If basis exactly matches constraint count, try LU factorization
                if len(current_B_x) + len(current_B_s) == n_constrs:
                    # Try LU factorization
                    lu, piv = la.lu_factor(B_matrix)
                    diag_lu = np.abs(np.diag(lu))
                    small_pivots = np.where(diag_lu < pivot_threshold)[0]
                    
                    if len(small_pivots) == 0:
                        # Successfully found valid basis
                        logger.debug(f"Found valid basis in iteration {iteration+1}")
                        logger.debug(f"Basis repair successful: |B_x|={len(current_B_x)}, |B_s|={len(current_B_s)}")
                        repair_success = True
                        break
                    
                    # List all positions of small pivot elements
                    logger.debug(f"Found {len(small_pivots)} small pivot elements: " + 
                                 ", ".join([f"{p}({diag_lu[p]:.2e})" for p in small_pivots]))
                    
                    # Find the column with the smallest pivot element, and remove it from the basis
                    min_pivot_idx = small_pivots[np.argmin(diag_lu[small_pivots])]
                    min_pivot_value = diag_lu[min_pivot_idx]
                    
                    # All small pivot element positions and values
                    all_small_pivots_info = [(i, diag_lu[i]) for i in small_pivots]
                    logger.debug(f"All small pivot element information: {all_small_pivots_info}")
                    
                    # Determine if the column belongs to a variable or slack
                    if min_pivot_idx < len(current_B_x):
                        # Remove variable
                        removed_var = current_B_x[min_pivot_idx]
                        # Record variable information and probability
                        if removed_var < len(var_probs):
                            var_prob_info = var_probs[removed_var].tolist() if hasattr(var_probs[removed_var], 'tolist') else var_probs[removed_var]
                            logger.debug(f"Remove basis variable {removed_var} (pivot value: {min_pivot_value:.6e}, probability: {var_prob_info})")
                        else:
                            logger.debug(f"Remove basis variable {removed_var} (pivot value: {min_pivot_value:.6e})")
                        current_B_x.pop(min_pivot_idx)
                    else:
                        # Remove slack
                        adj_idx = min_pivot_idx - len(current_B_x)
                        removed_constr = current_B_s[adj_idx]
                        # Record slack information and probability
                        if removed_constr < len(constr_probs):
                            constr_prob_info = constr_probs[removed_constr].tolist() if hasattr(constr_probs[removed_constr], 'tolist') else constr_probs[removed_constr]
                            logger.debug(f"Remove basis slack {removed_constr} (pivot value: {min_pivot_value:.6e}, probability: {constr_prob_info})")
                        else:
                            logger.debug(f"Remove basis slack {removed_constr} (pivot value: {min_pivot_value:.6e})")
                        current_B_s.pop(adj_idx)
                
                # If basis is less than constraint count, add new variables or slacks
                while len(current_B_x) + len(current_B_s) < n_constrs and all_candidates:
                    # Select the variable or slack with the highest probability from the candidate list
                    idx, node_type, prob = all_candidates.pop(0)
                    
                    # Check if it is already in the basis
                    if (node_type == 'var' and idx in current_B_x) or \
                       (node_type == 'constr' and idx in current_B_s):
                        continue
                    
                    # Add to current basis
                    if node_type == 'var':
                        logger.debug(f"Add basis variable {idx} (probability: {prob:.4f})")
                        current_B_x.append(idx)
                    else:
                        logger.debug(f"Add basis slack {idx} (probability: {prob:.4f})")
                        current_B_s.append(idx)
                    
                    # If basis size reaches constraint count, check if it is valid
                    if len(current_B_x) + len(current_B_s) == n_constrs:
                        break
                        
            except Exception as e:
                logger.warning(f"Error occurred in iteration {iteration+1}: {str(e)}")
                # Continue to repair in the next iteration
                continue
                
            # If candidate list is empty and basis still insufficient, it means it cannot be repaired
            if len(current_B_x) + len(current_B_s) < n_constrs and not all_candidates:
                logger.warning("Candidate list exhausted, basis still insufficient")
                break
        
        # Check if the repaired basis is valid
        if repair_success:
            logger.debug(f"Basis repair successful: |B_x|={len(current_B_x)}, |B_s|={len(current_B_s)}")
            return current_B_x, current_B_s
        
        if not repair_success:
            logger.debug(f"Basis repair failed (reached maximum iterations {max_iterations})")
        
        # Rebuild candidate list, combine basis probability and coefficient size factor
        all_candidates = []
        
        # Add all variables, combine probability and coefficient size
        for i in range(n_vars):
            # Calculate the size of the variable in the constraint matrix
            coef_norm = np.linalg.norm(A_dense[:, i]) if i < A_dense.shape[1] else 0
            # Basis probability
            base_prob = var_probs[i, 1].item()
            # Combined score: combine probability and coefficient size
            combined_score = base_prob * (1 + 0.2 * coef_norm)
            logger.debug(f"Variable {i} probability: {base_prob:.4f}, coefficient norm: {coef_norm:.4f}, combined score: {combined_score:.4f}")
            all_candidates.append((i, 'var', combined_score))
        
        # Add all slack variables and their basis probability
        for i in range(n_constrs):
            all_candidates.append((i, 'constr', constr_probs[i, 1].item()))
        
        # Sort by combined score in descending order
        all_candidates.sort(key=lambda x: x[2], reverse=True)
        logger.debug(f"Heuristic method: candidate list has {len(all_candidates)} candidate variables")
        
        # Create new basis variable and slack variable lists
        heuristic_B_x = []
        heuristic_B_s = []
        
        # First try the probability-based method: select the variable and slack with the highest probability
        for i, (idx, node_type, prob) in enumerate(all_candidates):
            if len(heuristic_B_x) + len(heuristic_B_s) >= n_constrs:
                break
                
            if node_type == 'var':
                logger.debug(f"Heuristic method: add basis variable {idx} (probability: {prob:.4f})")
                heuristic_B_x.append(idx)
            else:
                logger.debug(f"Heuristic method: add basis slack {idx} (probability: {prob:.4f})")
                heuristic_B_s.append(idx)
        
        # Check if the heuristic method generates a basis of sufficient size
        if len(heuristic_B_x) + len(heuristic_B_s) < n_constrs:
            logger.warning(f"Heuristic method: basis size insufficient ({len(heuristic_B_x) + len(heuristic_B_s)}/{n_constrs}), will add additional slack variables")
            
            # If basis still insufficient, add remaining slack variables
            already_used_constrs = set(heuristic_B_s)
            for i in range(n_constrs):
                if len(heuristic_B_x) + len(heuristic_B_s) >= n_constrs:
                    break
                    
                if i not in already_used_constrs:
                    logger.debug(f"Heuristic method: add additional slack variable {i}")
                    heuristic_B_s.append(i)
        
        logger.debug(f"Heuristic method: final basis size |B_x|={len(heuristic_B_x)}, |B_s|={len(heuristic_B_s)}, total basis size={len(heuristic_B_x) + len(heuristic_B_s)}/{n_constrs}")
        
        # Verify if the heuristic basis is valid
        try:
            if len(heuristic_B_x) + len(heuristic_B_s) == n_constrs:
                # Build heuristic basis matrix
                H_matrix = np.zeros((n_constrs, len(heuristic_B_x) + len(heuristic_B_s)))
                
                # Fill variable part
                for i, var_idx in enumerate(heuristic_B_x):
                    H_matrix[:, i] = A_dense[:, var_idx]
                
                # Fill slack part
                for i, constr_idx in enumerate(heuristic_B_s):
                    H_matrix[constr_idx, len(heuristic_B_x) + i] = -1
                
                # Try LU factorization to verify if the basis is invertible
                lu, piv = la.lu_factor(H_matrix)
                diag_lu = np.abs(np.diag(lu))
                small_pivots = np.where(diag_lu < pivot_threshold)[0]
                
                if len(small_pivots) == 0:
                    logger.debug("Heuristic method: generated basis matrix is invertible, basis repair successful")
                else:
                    logger.debug(f"Heuristic method: basis still has {len(small_pivots)} small pivots, but will use this basis as the final result")
            else:
                logger.warning(f"Heuristic method: basis size ({len(heuristic_B_x) + len(heuristic_B_s)}) does not equal constraint number ({n_constrs}), but will use this basis as the final result")
        except Exception as e:
            logger.error(f"Heuristic method: basis matrix check failed: {str(e)}")
        
        return heuristic_B_x, heuristic_B_s


class GurobiEvaluator:
    """Gurobi evaluator"""
    
    def __init__(self, model, device=None, config=None):
        """
        Initialize the evaluator
        
        Args:
            model: trained GNN model
            device: calculation device
            config: evaluation configuration
        """
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config or {}
        
        # Move the model to the device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()
        
        # Gurobi parameters
        self.time_limit = self.config.get('time_limit', 600)  # Solving time limit (seconds)
        self.mip_gap = self.config.get('mip_gap', 0.01)       # MIP gap
        self.threads = self.config.get('threads', 1)          # Number of threads
        
        # Results storage directory
        self.results_dir = self.config.get('results_dir', './results')
        os.makedirs(self.results_dir, exist_ok=True)
        
    def prepare_instance(self, instance_path: str) -> Dict[str, Any]:
        """
        Prepare MILP instance data for prediction
        
        Args:
            instance_path: MILP instance file path
            
        Returns:
            Dict: Processed instance data
        """
        try:
            # Extract instance data
            milp_instance = MILPInstance(instance_path)
            instance_name = os.path.basename(instance_path)
            
            # Extract bipartite graph features
            feature_extractor = BipartiteGraphFeatureExtractor(milp_instance)
            graph_data = feature_extractor.get_graph_data()
            
            # Build batch data
            batch_data = {
                'var_features': torch.FloatTensor(graph_data['var_features']).unsqueeze(0),
                'constr_features': torch.FloatTensor(graph_data['constr_features']).unsqueeze(0),
                'edge_index': [torch.LongTensor(graph_data['edge_index'])],
                'edge_attr': [torch.FloatTensor(graph_data['edge_attr'])],
                'n_vars': milp_instance.n,
                'n_constrs': milp_instance.m,
                'instance_name': instance_name,
                'instance_path': instance_path,
                'A': milp_instance.A
            }
            
            # Use the knowledge mask generated by the feature extractor, not recalculated
            if 'var_mask' in graph_data and 'constr_mask' in graph_data:
                logger.debug("Using the knowledge mask generated by the feature extractor")
                batch_data['batch_var_mask'] = torch.FloatTensor(graph_data['var_mask']).unsqueeze(0)
                batch_data['batch_constr_mask'] = torch.FloatTensor(graph_data['constr_mask']).unsqueeze(0)
            else:
                # Compatible with old code, use the manually calculated mask
                logger.warning("The feature extractor does not have a knowledge mask, manually calculate")
                var_mask = np.zeros((milp_instance.n, 3))
                constr_mask = np.zeros((milp_instance.m, 3))
                
                # Variable mask
                for i in range(milp_instance.n):
                    if milp_instance.l_x[i] <= -1e10:
                        var_mask[i, 0] = -1e10  # Cannot be at the lower bound
                    if milp_instance.u_x[i] >= 1e10:
                        var_mask[i, 2] = -1e10  # Cannot be at the upper bound
                
                # Constraint mask
                for i in range(milp_instance.m):
                    if milp_instance.l_s[i] <= -1e10:
                        constr_mask[i, 0] = -1e10  # Cannot be at the lower bound
                    if milp_instance.u_s[i] >= 1e10:
                        constr_mask[i, 2] = -1e10  # Cannot be at the upper bound
                
                batch_data['batch_var_mask'] = torch.FloatTensor(var_mask).unsqueeze(0)
                batch_data['batch_constr_mask'] = torch.FloatTensor(constr_mask).unsqueeze(0)
            
            return batch_data
            
        except Exception as e:
            logger.error(f"Prepare instance {instance_path} failed: {str(e)}")
            raise
    
    def evaluate_instance(self, instance_path: str) -> Dict[str, Any]:
        """
        Evaluate a single MILP instance
        
        Args:
            instance_path: MILP instance file path
            
        Returns:
            Dict: Evaluation results
        """
        instance_name = os.path.basename(instance_path)
        logger.info(f"Evaluate instance: {instance_name}")
        
        try:
            # Prepare instance data
            batch_data = self.prepare_instance(instance_path)
            n_vars = batch_data['n_vars']
            n_constrs = batch_data['n_constrs']
            A = batch_data['A']
            
            # Use the model to predict
            with torch.no_grad():
                outputs = self.model.predict(batch_data)
                var_probs = outputs['var_probs'][0]  # [n_vars, 3]
                constr_probs = outputs['constr_probs'][0]  # [n_constrs, 3]
            
            # Get candidate basis from predicted probabilities
            B_x, B_s = get_basis_from_probs(var_probs, constr_probs, n_vars, n_constrs)
            
            # Repair basis
            B_x, B_s = BasisRepair.repair_basis(A, B_x, B_s, n_vars, n_constrs, var_probs, constr_probs)
            
            # Prepare Gurobi status codes
            var_basis_codes = [-1] * n_vars  # Default all variables at lower bound (NonbasicAtLower)
            constr_basis_codes = [-1] * n_constrs  # Default all constraints at lower bound (NonbasicAtLower)
            
            # Set basis variable status
            for idx in B_x:
                var_basis_codes[idx] = 0  # Basic
            
            # Set basis slack status
            for idx in B_s:
                constr_basis_codes[idx] = 0  # Basic
            
            # Non-basis variable status: lower bound or upper bound
            for i in range(n_vars):
                if i not in B_x:  # Non-basis variable
                    # If the probability of being at the upper bound is greater than the probability of being at the lower bound, place it at the upper bound
                    if var_probs[i, 2] > var_probs[i, 0]:
                        var_basis_codes[i] = -2  # NonbasicAtUpper
            
            # Non-basis slack status: lower bound or upper bound
            for i in range(n_constrs):
                if i not in B_s:  # Non-basis slack
                    # If the probability of being at the upper bound is greater than the probability of being at the lower bound, place it at the upper bound
                    if constr_probs[i, 2] > constr_probs[i, 0]:
                        constr_basis_codes[i] = -2  # NonbasicAtUpper
            
            # Run Gurobi experiment
            # 1. Baseline run (no initial basis)
            baseline_results = self._run_gurobi(instance_path)
            
            # 2. Run with predicted initial basis
            predicted_results = self._run_gurobi(instance_path, var_basis_codes, constr_basis_codes)
            
            # Calculate relative improvements
            relative_improvements = self._calculate_improvements(baseline_results, predicted_results)
            
            # Combine results
            result = {
                'instance_name': instance_name,
                'instance_path': instance_path,
                'n_vars': n_vars,
                'n_constrs': n_constrs,
                'baseline': baseline_results,
                'predicted': predicted_results,
                'improvements': relative_improvements,
                'basis_set_success': predicted_results['basis_set_success']
            }
            
            return result
            
        except Exception as e:
            logger.error(f"评估实例 {instance_name} 失败: {str(e)}")
            return {
                'instance_name': instance_name,
                'instance_path': instance_path,
                'error': str(e),
                'success': False
            }
    
    def _run_gurobi(self, instance_path: str, var_basis_codes: List[int] = None, 
                   constr_basis_codes: List[int] = None) -> Dict[str, Any]:
        """
        Run Gurobi to solve MILP instance
        
        Args:
            instance_path: MILP instance file path
            var_basis_codes: List of variable basis status codes (None means using Gurobi default basis)
            constr_basis_codes: List of constraint basis status codes (None means using Gurobi default basis)
            
        Returns:
            Dict: Solving results
        """
        try:
            # Read model
            env = gp.Env(empty=True)
            
            # Create log file path
            log_dir = os.path.join(self.results_dir, 'gurobi_logs')
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"gurobi_{os.path.basename(instance_path)}.log")
            logger.debug(f"Create Gurobi log file: {log_file}")
            
            # Enable logging to file
            env.setParam('OutputFlag', 1)  # Enable output (maximum value is 1)
            env.setParam('LogFile', log_file)  # Set log file
            env.setParam('LogToConsole', 0)  # Disable output to console
            
            # Presolve settings options
            presolve_options = {
                'no_presolve': {'enabled': False, 'description': 'Disable all presolve'},
                'basic_presolve': {'enabled': True, 'description': 'Basic presolve mode'},
                'presolve_with_basis_mapping': {'enabled': False, 'description': 'Enable presolve and basis mapping adaptation'}
            }
            
            # Select presolve mode
            selected_mode = 'basic_presolve'  # Can be changed to 'no_presolve' or 'presolve_with_basis_mapping'
            
            # Set parameters based on selected mode
            if selected_mode == 'no_presolve':
                env.setParam('Presolve', 0)  # Disable all presolve
                logger.debug(f"Presolve mode: {presolve_options[selected_mode]['description']}")
            elif selected_mode == 'presolve_with_basis_mapping':
                # Enable basis mapping after presolve
                env.setParam('PreCrush', 1)  # Preserve original variables in presolve
                env.setParam('PreMIQPMethod', 1)  # Use a more advanced presolve method to preserve basis
                env.setParam('PreSparsify', 0)  # Avoid basis mapping issues due to sparsification
                logger.debug(f"Presolve mode: {presolve_options[selected_mode]['description']}")
            else:  # basic_presolve
                # Default to only preserving original variables in presolve
                env.setParam('PreCrush', 1)  # Preserve original variables in presolve
                logger.debug(f"Presolve mode: {presolve_options[selected_mode]['description']}")
            
            env.start()
            
            model = gp.read(instance_path, env=env)
            
            # Set solving parameters
            if self.time_limit is not None:
                model.setParam(GRB.Param.TimeLimit, self.time_limit)
            model.setParam(GRB.Param.MIPGap, self.mip_gap)
            model.setParam(GRB.Param.Threads, self.threads)
            
            # Set initial basis
            basis_set_success = False
            if var_basis_codes is not None and constr_basis_codes is not None:
                variables = model.getVars()
                constraints = model.getConstrs()
                
                if len(variables) == len(var_basis_codes) and len(constraints) == len(constr_basis_codes):
                    try:
                        # Use direct API method to set basis status
                        logger.debug(f"Start setting initial basis, variable count: {len(variables)}, constraint count: {len(constraints)}")
                        
                        # Ensure the model is ready to accept basis status
                        model.update()
                        
                        # Record some original basis status information for reference
                        basic_vars_count = var_basis_codes.count(0)
                        basic_constrs_count = constr_basis_codes.count(0)
                        nonbasic_vars_count = len(var_basis_codes) - basic_vars_count
                        nonbasic_constrs_count = len(constr_basis_codes) - basic_constrs_count
                        logger.debug(f"Basis status information - Basic variable count: {basic_vars_count}, Non-basic variable count: {nonbasic_vars_count}, " 
                                     f"Basic constraint count: {basic_constrs_count}, Non-basic constraint count: {nonbasic_constrs_count}")
                        
                        # First, use object attributes to set basis
                        for i, var in enumerate(variables):
                            var.VBasis = var_basis_codes[i]
                            
                        for i, constr in enumerate(constraints):
                            constr.CBasis = constr_basis_codes[i]
                        
                        # Ensure changes are applied
                        model.update()
                        
                        # Second method: use setAttr method to set all variable and constraint basis status
                        try:
                            model.setAttr("VBasis", variables, var_basis_codes)
                            model.setAttr("CBasis", constraints, constr_basis_codes)
                            logger.debug("使用setAttr方法成功设置了初始基")
                        except Exception as e:
                            logger.warning(f"setAttr设置初始基失败: {str(e)}")
                        
                        # Ensure changes are applied
                        model.update()
                        
                        # Verify if the basis is correctly set
                        var_basis_after_setting = [var.VBasis if hasattr(var, 'VBasis') else -99 for var in variables]
                        constr_basis_after_setting = [constr.CBasis if hasattr(constr, 'CBasis') else -99 for constr in constraints]
                        
                        # Calculate the difference in basis setting
                        var_basis_difference = sum(1 for i, basis in enumerate(var_basis_after_setting) if basis != var_basis_codes[i])
                        constr_basis_difference = sum(1 for i, basis in enumerate(constr_basis_after_setting) if basis != constr_basis_codes[i])
                        
                        # Save basis verification results to a separate log file
                        basis_verification_dir = os.path.join(self.results_dir, 'basis_verification')
                        os.makedirs(basis_verification_dir, exist_ok=True)
                        verification_file = os.path.join(basis_verification_dir, f"basis_verification_{os.path.basename(instance_path)}.json")
                        
                        # Convert data to serializable format
                        verification_result = {
                            'instance_path': instance_path,
                            'var_basis_codes_set': [int(x) if isinstance(x, (int, float)) else str(x) for x in var_basis_codes],
                            'constr_basis_codes_set': [int(x) if isinstance(x, (int, float)) else str(x) for x in constr_basis_codes],
                            'var_basis_after_setting': [int(x) if isinstance(x, (int, float)) else str(x) for x in var_basis_after_setting],
                            'constr_basis_after_setting': [int(x) if isinstance(x, (int, float)) else str(x) for x in constr_basis_after_setting],
                            'var_basis_difference': var_basis_difference,
                            'constr_basis_difference': constr_basis_difference,
                            'total_difference': var_basis_difference + constr_basis_difference
                        }
                        
                        with open(verification_file, 'w') as f:
                            json.dump(verification_result, f, indent=2)
                        
                        logger.debug(f"Basis verification results saved to: {verification_file}")
                        
                        basis_set_success = True
                        basic_vars_count = var_basis_codes.count(0) + constr_basis_codes.count(0)
                        logger.debug(f"Successfully set initial basis, variable count: {basic_vars_count}, difference: {var_basis_difference + constr_basis_difference}")
                    except gp.GurobiError as e:
                        logger.warning(f"Failed to set initial basis: {str(e)}")
                else:
                    logger.warning(f"Variable or constraint count mismatch: model({len(variables)}, {len(constraints)}) vs basis({len(var_basis_codes)}, {len(constr_basis_codes)})")
            
            # Solve model
            start_time = time.time()
            model.optimize()
            solve_time = time.time() - start_time
            
            # Verify basis status after optimization, check if Gurobi used the basis we set
            basis_used_by_gurobi = False
            if model.Status == GRB.OPTIMAL and basis_set_success:
                try:
                    # Get basis status after solving
                    var_basis_after_solve = [var.VBasis if hasattr(var, 'VBasis') else -99 for var in variables]
                    constr_basis_after_solve = [constr.CBasis if hasattr(constr, 'CBasis') else -99 for constr in constraints]
                    
                    # Compare if the basis variables we set are still basis variables after solving
                    basic_vars_match = 0
                    total_basic_vars = 0
                    
                    for i, code in enumerate(var_basis_codes):
                        if code == 0:  # Basic variable
                            total_basic_vars += 1
                            if var_basis_after_solve[i] == 0:  # If it is still a basic variable
                                basic_vars_match += 1
                    
                    for i, code in enumerate(constr_basis_codes):
                        if code == 0:  # Basic constraint
                            total_basic_vars += 1
                            if constr_basis_after_solve[i] == 0:  # If it is still a basic constraint
                                basic_vars_match += 1
                    
                    # Calculate matching percentage
                    if total_basic_vars > 0:
                        basis_match_percentage = (basic_vars_match / total_basic_vars) * 100
                        logger.debug(f"Initial basis usage rate: {basis_match_percentage:.2f}% ({basic_vars_match}/{total_basic_vars})")
                        
                        # If the matching rate exceeds a certain threshold, we consider that Gurobi used the initial basis we set
                        if basis_match_percentage >= 50:  # Threshold set to 50%, can be adjusted as needed
                            basis_used_by_gurobi = True
                            logger.debug("Initial basis was successfully used and partially retained in the solving process")
                        else:
                            logger.warning("Initial basis was set but significantly changed during solving")
                    
                    # Save basis status after solving
                    basis_after_solve_file = os.path.join(basis_verification_dir, f"basis_after_solve_{os.path.basename(instance_path)}.json")
                    with open(basis_after_solve_file, 'w') as f:
                        json.dump({
                            'instance_path': instance_path,
                            'var_basis_after_solve': [int(x) if isinstance(x, (int, float)) else str(x) for x in var_basis_after_solve],
                            'constr_basis_after_solve': [int(x) if isinstance(x, (int, float)) else str(x) for x in constr_basis_after_solve],
                            'basis_match_percentage': basis_match_percentage,
                            'basic_vars_match': basic_vars_match,
                            'total_basic_vars': total_basic_vars
                        }, f, indent=2)
                    
                    logger.debug(f"Basis status after solving saved to: {basis_after_solve_file}")
                    
                except Exception as e:
                    logger.warning(f"Failed to get basis status after solving: {str(e)}")
            
            # Collect results
            result = {
                'status': model.Status,
                'status_str': self._get_status_string(model.Status),
                'runtime': model.Runtime,  # Gurobi internal timing
                'solve_time': solve_time,  # Python timing
                'objective': model.ObjVal if model.Status == GRB.OPTIMAL else None,
                'mip_gap': model.MIPGap if model.Status == GRB.OPTIMAL else None,
                'node_count': model.NodeCount,
                'iter_count': model.IterCount,
                'basis_set_success': basis_set_success,
                'basis_used_by_gurobi': basis_used_by_gurobi  # New field, indicating if the initial basis was actually used by Gurobi
            }
            
            # Release resources
            model.dispose()
            env.dispose()
            
            return result
            
        except Exception as e:
            logger.error(f"Run Gurobi to solve {instance_path} failed: {str(e)}")
            return {
                'status': -1,
                'status_str': 'ERROR',
                'runtime': None,
                'solve_time': None,
                'objective': None,
                'mip_gap': None,
                'node_count': None,
                'iter_count': None,
                'error': str(e),
                'basis_set_success': False
            }
    
    def _calculate_improvements(self, baseline: Dict[str, Any], predicted: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate the improvement of predicted basis compared to the baseline
        
        Args:
            baseline: Baseline solving results
            predicted: Solving results using predicted basis
            
        Returns:
            Dict: Improvement metrics
        """
        improvements = {}
        
        # Time improvement
        if baseline['runtime'] is not None and predicted['runtime'] is not None and baseline['runtime'] > 0:
            improvements['runtime_improvement'] = (baseline['runtime'] - predicted['runtime']) / baseline['runtime']
        else:
            improvements['runtime_improvement'] = None
        
        # Node count improvement
        if baseline['node_count'] is not None and predicted['node_count'] is not None and baseline['node_count'] > 0:
            improvements['node_count_improvement'] = (baseline['node_count'] - predicted['node_count']) / baseline['node_count']
        else:
            improvements['node_count_improvement'] = None
        
        # Iteration count improvement
        if baseline['iter_count'] is not None and predicted['iter_count'] is not None and baseline['iter_count'] > 0:
            improvements['iter_count_improvement'] = (baseline['iter_count'] - predicted['iter_count']) / baseline['iter_count']
        else:
            improvements['iter_count_improvement'] = None
            
        return improvements
    
    def _get_status_string(self, status: int) -> str:
        """
        Convert Gurobi status code to string
        
        Args:
            status: Gurobi status code
            
        Returns:
            str: Status description
        """
        status_map = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.NODE_LIMIT: "NODE_LIMIT",
            GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
            GRB.INTERRUPTED: "INTERRUPTED",
            GRB.NUMERIC: "NUMERIC",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
            GRB.LOADED: "LOADED"
        }
        return status_map.get(status, f"UNKNOWN_{status}")
        
    def evaluate_single_model(self, test_dir: str, output_file: str = None) -> Dict[str, Any]:
        """
        Evaluate the performance of a single model on the test set
        
        This method will evaluate each MILP instance in the test set,
        and compare the results with the default basis of Gurobi.
        
        Args:
            test_dir: Test instance directory path
            output_file: Result output file path (optional)
            
        Returns:
            Dict: Dictionary containing evaluation results and statistics
        """
        logger.info(f"Start evaluating test set: {test_dir}")
        
        # Find test instances
        valid_exts = ['.mps', '.lp', '.mps.gz', '.lp.gz']
        test_instances = []
        for root, _, files in os.walk(test_dir):
            for file in files:
                if any(file.endswith(ext) for ext in valid_exts):
                    test_instances.append(os.path.join(root, file))
                    
        test_instances.sort()  # Keep the order consistent
        logger.info(f"Found {len(test_instances)} test instances")
        
        # Evaluate results
        results = {
            'instances': [],
            'summary': {
                'model': {
                    'success_count': 0,
                    'runtime_improvements': [],
                    'node_count_improvements': [],
                    'iter_count_improvements': [],
                    'basis_set_success_count': 0
                }
            }
        }
        
        # Ensure the model is set to evaluation mode
        self.model.eval()
        
        # Evaluate each test instance - use tqdm to create a progress bar
        pbar = tqdm(enumerate(test_instances), total=len(test_instances), desc="Evaluating test instances progress", ncols=100)
        
        for i, instance_path in pbar:
            instance_name = os.path.basename(instance_path)
            # Update progress bar description
            pbar.set_description(f"Instance {i+1}/{len(test_instances)}: {instance_name[:30]}") 
            # Change detailed log to DEBUG level
            logger.debug(f"Evaluating instance {i+1}/{len(test_instances)}: {instance_name}")
            
            instance_results = {'instance_name': instance_name, 'instance_path': instance_path}
            
            try:
                # Prepare instance data
                batch_data = self.prepare_instance(instance_path)
                
                # Run baseline (not using predicted basis)
                baseline_results = self._run_gurobi(instance_path)
                instance_results['baseline'] = baseline_results
                
                # Use model to predict
                with torch.no_grad():
                    outputs = self.model.predict(batch_data)
                    var_probs = outputs['var_probs'][0]  # [n_vars, 3]
                    constr_probs = outputs['constr_probs'][0]  # [n_constrs, 3]
                
                # Get predicted basis
                n_vars = batch_data['n_vars']
                n_constrs = batch_data['n_constrs']
                A = batch_data['A']
                B_x, B_s = get_basis_from_probs(var_probs, constr_probs, n_vars, n_constrs)
                B_x, B_s = BasisRepair.repair_basis(A, B_x, B_s, n_vars, n_constrs, var_probs, constr_probs)
                
                # Prepare Gurobi status codes
                var_basis_codes = [-1] * n_vars  # Default all variables in lower bound
                constr_basis_codes = [-1] * n_constrs  # Default all constraints in lower bound
                
                # Set basis variable status
                for idx in B_x:
                    var_basis_codes[idx] = 0  # Basic variable
                
                # Set basis relaxation status
                for idx in B_s:
                    constr_basis_codes[idx] = 0  # Basic variable
                
                # Non-basis variable status
                for i in range(n_vars):
                    if i not in B_x:  # Non-basis variable
                        if var_probs[i, 2] > var_probs[i, 0]:  # Upper bound probability > lower bound probability
                            var_basis_codes[i] = -2  # In upper bound
                
                # Non-basis relaxation status
                for i in range(n_constrs):
                    if i not in B_s:  # Non-basis relaxation
                        if constr_probs[i, 2] > constr_probs[i, 0]:  # Upper bound probability > lower bound probability
                            constr_basis_codes[i] = -2  # In upper bound
                
                # Run model evaluation
                model_results = self._run_gurobi(instance_path, var_basis_codes, constr_basis_codes)
                instance_results['model_results'] = model_results
                improvements = self._calculate_improvements(baseline_results, model_results)
                instance_results['improvements'] = improvements
                
                # Record if basis application is successful
                instance_results['basis_applied_successfully'] = model_results['basis_set_success']
                
                # Statistics results
                if model_results['basis_set_success']:
                    results['summary']['model']['basis_set_success_count'] += 1
                    
                    if improvements['runtime_improvement'] is not None:
                        results['summary']['model']['runtime_improvements'].append(improvements['runtime_improvement'])
                    
                    if improvements['node_count_improvement'] is not None:
                        results['summary']['model']['node_count_improvements'].append(improvements['node_count_improvement'])
                    
                    if improvements['iter_count_improvement'] is not None:
                        results['summary']['model']['iter_count_improvements'].append(improvements['iter_count_improvement'])
                
                # Determine if the instance is successfully evaluated
                instance_results['success'] = True
                
            except Exception as e:
                logger.error(f"Evaluating instance {instance_name} failed: {str(e)}")
                instance_results['error'] = str(e)
                instance_results['success'] = False
            
            # Add instance results
            results['instances'].append(instance_results)
        
        # Calculate statistical metrics
        summary = results['summary']['model']
        runtime_improvements = summary['runtime_improvements']
        node_count_improvements = summary['node_count_improvements']
        iter_count_improvements = summary['iter_count_improvements']
        
        if runtime_improvements:
            summary['avg_runtime_improvement'] = np.mean(runtime_improvements)
            summary['median_runtime_improvement'] = np.median(runtime_improvements)
        
        if node_count_improvements:
            summary['avg_node_count_improvement'] = np.mean(node_count_improvements)
            summary['median_node_count_improvement'] = np.median(node_count_improvements)
        
        if iter_count_improvements:
            summary['avg_iter_count_improvement'] = np.mean(iter_count_improvements)
            summary['median_iter_count_improvement'] = np.median(iter_count_improvements)
        
        # Save results
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {output_file}")
        
        return results
        
    def evaluate_test_set(self, test_dir: str, model_o_path: str, model_s_path: str, output_file: str = None) -> Dict[str, Any]:
        """
        Evaluate the performance of models trained on the original dataset (O) and the synthetic dataset (S) on the test set
        
        Args:
            test_dir: Test instance directory path
            model_o_path: Path to the model trained on the original dataset
            model_s_path: Path to the model trained on the synthetic dataset
            output_file: Result output file path (optional)
            
        Returns:
            Dict: Evaluation results statistics
        """
        logger.info(f"Start evaluating test set: {test_dir}")
        
        # Find test instances
        valid_exts = ['.mps', '.lp', '.mps.gz', '.lp.gz']
        test_instances = []
        for root, _, files in os.walk(test_dir):
            for file in files:
                if any(file.endswith(ext) for ext in valid_exts):
                    test_instances.append(os.path.join(root, file))
                    
        test_instances.sort()  # Keep the order consistent
        logger.info(f"Found {len(test_instances)} test instances")
        
        # Evaluate results
        results = {
            'instances': [],
            'summary': {
                'o_model': {
                    'success_count': 0,
                    'runtime_improvements': [],
                    'node_count_improvements': [],
                    'iter_count_improvements': [],
                    'basis_set_success_count': 0
                },
                's_model': {
                    'success_count': 0,
                    'runtime_improvements': [],
                    'node_count_improvements': [],
                    'iter_count_improvements': [],
                    'basis_set_success_count': 0
                }
            }
        }
        
        # Load model O
        from .model import InitialBasisGNN
        logger.info(f"Loading model O (trained on original dataset): {model_o_path}")
        model_o = InitialBasisGNN()
        model_o.load_state_dict(torch.load(model_o_path, map_location=self.device, weights_only=True)['model_state_dict'])
        model_o.to(self.device)
        model_o.eval()
        
        # Load model S
        logger.info(f"Loading model S (trained on synthetic dataset): {model_s_path}")
        model_s = InitialBasisGNN()
        model_s.load_state_dict(torch.load(model_s_path, map_location=self.device, weights_only=True)['model_state_dict'])
        model_s.to(self.device)
        model_s.eval()
        
        # Evaluate each test instance - use tqdm to create a progress bar
        pbar = tqdm(enumerate(test_instances), total=len(test_instances), desc="Evaluating test instances progress", ncols=100)
        
        for i, instance_path in pbar:
            instance_name = os.path.basename(instance_path)
            # Update progress bar description
            pbar.set_description(f"Instance {i+1}/{len(test_instances)}: {instance_name[:30]}") 
            # Change detailed log to DEBUG level
            logger.debug(f"Evaluating instance {i+1}/{len(test_instances)}: {instance_name}")
            
            instance_results = {'instance_name': instance_name, 'instance_path': instance_path}
            
            try:
                # Prepare instance data
                batch_data = self.prepare_instance(instance_path)
                
                # Run baseline (not using predicted basis)
                baseline_results = self._run_gurobi(instance_path)
                instance_results['baseline'] = baseline_results
                
                # Use model O
                self.model = model_o
                with torch.no_grad():
                    outputs_o = model_o.predict(batch_data)
                    var_probs_o = outputs_o['var_probs'][0]  # [n_vars, 3]
                    constr_probs_o = outputs_o['constr_probs'][0]  # [n_constrs, 3]
                
                # Get basis of model O
                n_vars = batch_data['n_vars']
                n_constrs = batch_data['n_constrs']
                A = batch_data['A']
                B_x_o, B_s_o = get_basis_from_probs(var_probs_o, constr_probs_o, n_vars, n_constrs)
                B_x_o, B_s_o = BasisRepair.repair_basis(A, B_x_o, B_s_o, n_vars, n_constrs, var_probs_o, constr_probs_o)
                
                # Prepare Gurobi status codes for model O
                var_basis_codes_o = [-1] * n_vars  # Default all variables in lower bound
                constr_basis_codes_o = [-1] * n_constrs  # Default all constraints in lower bound
                
                # Set basis variable status
                for idx in B_x_o:
                    var_basis_codes_o[idx] = 0  # Basic variable
                
                # Set basis relaxation status
                for idx in B_s_o:
                    constr_basis_codes_o[idx] = 0  # Basic variable
                
                # Non-basis variable status
                for i in range(n_vars):
                    if i not in B_x_o:  # Non-basis variable
                        if var_probs_o[i, 2] > var_probs_o[i, 0]:  # Upper bound probability > lower bound probability
                            var_basis_codes_o[i] = -2  # In upper bound
                
                # Non-basis relaxation status
                for i in range(n_constrs):
                    if i not in B_s_o:  # Non-basis relaxation
                        if constr_probs_o[i, 2] > constr_probs_o[i, 0]:  # Upper bound probability > lower bound probability
                            constr_basis_codes_o[i] = -2  # In upper bound
                
                # Run model O evaluation
                o_results = self._run_gurobi(instance_path, var_basis_codes_o, constr_basis_codes_o)
                instance_results['o_model'] = o_results
                o_improvements = self._calculate_improvements(baseline_results, o_results)
                instance_results['o_improvements'] = o_improvements
                
                # Statistics model O results
                if o_results['basis_set_success']:
                    instance_results['o_model']['basis_set_success_count'] = 1
                    
                if o_improvements['runtime_improvement'] is not None:
                    instance_results['o_model']['runtime_improvements'] = [o_improvements['runtime_improvement']]
                    
                if o_improvements['node_count_improvement'] is not None:
                    instance_results['o_model']['node_count_improvements'] = [o_improvements['node_count_improvement']]
                    
                if o_improvements['iter_count_improvement'] is not None:
                    instance_results['o_model']['iter_count_improvements'] = [o_improvements['iter_count_improvement']]
                
                # Use model S
                self.model = model_s
                with torch.no_grad():
                    outputs_s = model_s.predict(batch_data)
                    var_probs_s = outputs_s['var_probs'][0]  # [n_vars, 3]
                    constr_probs_s = outputs_s['constr_probs'][0]  # [n_constrs, 3]
                
                # Get basis of model S
                B_x_s, B_s_s = get_basis_from_probs(var_probs_s, constr_probs_s, n_vars, n_constrs)
                B_x_s, B_s_s = BasisRepair.repair_basis(A, B_x_s, B_s_s, n_vars, n_constrs, var_probs_s, constr_probs_s)
                
                # Prepare Gurobi status codes for model S
                var_basis_codes_s = [-1] * n_vars  # Default all variables in lower bound
                constr_basis_codes_s = [-1] * n_constrs  # Default all constraints in lower bound
                
                # Set basis variable status
                for idx in B_x_s:
                    var_basis_codes_s[idx] = 0  # Basic variable
                
                # Set basis relaxation status
                for idx in B_s_s:
                    constr_basis_codes_s[idx] = 0  # Basic variable
                
                # Non-basis variable status
                for i in range(n_vars):
                    if i not in B_x_s:  # Non-basis variable
                        if var_probs_s[i, 2] > var_probs_s[i, 0]:  # Upper bound probability > lower bound probability
                            var_basis_codes_s[i] = -2  # In upper bound
                
                # Non-basis relaxation status
                for i in range(n_constrs):
                    if i not in B_s_s:  # Non-basis relaxation
                        if constr_probs_s[i, 2] > constr_probs_s[i, 0]:  # Upper bound probability > lower bound probability
                            constr_basis_codes_s[i] = -2  # In upper bound
                
                # Run model S evaluation
                s_results = self._run_gurobi(instance_path, var_basis_codes_s, constr_basis_codes_s)
                instance_results['s_model'] = s_results
                s_improvements = self._calculate_improvements(baseline_results, s_results)
                instance_results['s_improvements'] = s_improvements
                
                # Statistics model S results
                if s_results['basis_set_success']:
                    instance_results['s_model']['basis_set_success_count'] = 1
                    
                if s_improvements['runtime_improvement'] is not None:
                    instance_results['s_model']['runtime_improvements'] = [s_improvements['runtime_improvement']]
                    
                if s_improvements['node_count_improvement'] is not None:
                    instance_results['s_model']['node_count_improvements'] = [s_improvements['node_count_improvement']]
                    
                if s_improvements['iter_count_improvement'] is not None:
                    instance_results['s_model']['iter_count_improvements'] = [s_improvements['iter_count_improvement']]
                
                # Determine if the instance is successfully evaluated
                instance_results['success'] = True
                
            except Exception as e:
                logger.error(f"Evaluating instance {instance_name} failed: {str(e)}")
                instance_results['error'] = str(e)
                instance_results['success'] = False
            
            # Add to results list
            results = {
                'instances': [instance_results],
                'summary': {
                    'o_model': {
                        'success_count': instance_results.get('o_model', {}).get('basis_set_success_count', 0),
                        'runtime_improvements': instance_results.get('o_model', {}).get('runtime_improvements', []),
                        'node_count_improvements': instance_results.get('o_model', {}).get('node_count_improvements', []),
                        'iter_count_improvements': instance_results.get('o_model', {}).get('iter_count_improvements', []),
                        'basis_set_success_count': instance_results.get('o_model', {}).get('basis_set_success_count', 0)
                    },
                    's_model': {
                        'success_count': instance_results.get('s_model', {}).get('basis_set_success_count', 0),
                        'runtime_improvements': instance_results.get('s_model', {}).get('runtime_improvements', []),
                        'node_count_improvements': instance_results.get('s_model', {}).get('node_count_improvements', []),
                        'iter_count_improvements': instance_results.get('s_model', {}).get('iter_count_improvements', []),
                        'basis_set_success_count': instance_results.get('s_model', {}).get('basis_set_success_count', 0)
                    }
                }
            }
            
            # Calculate statistical summary
            for model_key in ['o_model', 's_model']:
                summary = results['summary'][model_key]
                
                # Calculate average improvements
                runtime_imps = summary['runtime_improvements']
                if runtime_imps:
                    summary['avg_runtime_improvement'] = sum(runtime_imps) / len(runtime_imps)
                    summary['median_runtime_improvement'] = sorted(runtime_imps)[len(runtime_imps) // 2]
                else:
                    summary['avg_runtime_improvement'] = None
                    summary['median_runtime_improvement'] = None
                
                node_imps = summary['node_count_improvements']
                if node_imps:
                    summary['avg_node_count_improvement'] = sum(node_imps) / len(node_imps)
                    summary['median_node_count_improvement'] = sorted(node_imps)[len(node_imps) // 2]
                else:
                    summary['avg_node_count_improvement'] = None
                    summary['median_node_count_improvement'] = None
                
                iter_imps = summary['iter_count_improvements']
                if iter_imps:
                    summary['avg_iter_count_improvement'] = sum(iter_imps) / len(iter_imps)
                    summary['median_iter_count_improvement'] = sorted(iter_imps)[len(iter_imps) // 2]
                else:
                    summary['avg_iter_count_improvement'] = None
                    summary['median_iter_count_improvement'] = None
            
            # Save intermediate results every 10 instances
            if output_file and i > 0 and i % 10 == 0:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.debug(f"Intermediate evaluation results saved to {output_file} (instance {i}/{len(test_instances)})")
            
        
        
        pbar.close()
        
        # Save final complete results
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Evaluation completed, results saved to {output_file}")
        
        return results
