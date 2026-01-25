"""
GNN Data Processing Module for converting MILP instances into input format for GNN models.

This module provides functions to load MILP instances, extract features, and convert them into formats suitable for GNN models.
It supports loading instances directly from files without relying on preprocessed data.
"""

import os
import logging
import numpy as np
import pandas as pd
import gurobipy as gp
from typing import Dict, List, Tuple, Any, Optional
import tensorflow as tf
import networkx as nx

# Use absolute import paths to ensure correct imports from any location
try:
    from Benchmark4L2O.utils.common import solve_instance, instance2graph
except ImportError:
    from utils.common import solve_instance, instance2graph


def get_instance_files(instance_dir: str) -> List[str]:
    """
    Get paths to all MILP instance files in the directory.
    
    Args:
        instance_dir: Directory containing MILP instance files
        
    Returns:
        List of paths to MILP instance files
    """
    instance_files = []
    for file in os.listdir(instance_dir):
        if file.endswith('.mps') or file.endswith('.lp'):
            instance_files.append(os.path.join(instance_dir, file))
    return instance_files


def extract_milp_features(instance_path: str) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """
    Extract features from a MILP instance.
    
    Args:
        instance_path: Path to the MILP instance file
        
    Returns:
        solving_results: Solving results
        features: Dictionary of extracted features, including constraint features, variable features, edge features, and edge indices
    """
    try:
        # Use functions from common.py directly from the file path to extract the graph structure
        graph, _ = instance2graph(instance_path)
        
        # Unpack the graph structure
        constraint_features, edge_indices, edge_features, variable_features = graph
        
        # Solve the model to get the objective value
        solving_results = solve_instance(instance_path)
        
        # Create a dictionary of features
        features_dict = {
            'con_features': constraint_features,
            'var_features': variable_features,
            'edge_indices': edge_indices,
            'edge_features': edge_features
        }
        
        return solving_results, features_dict
    
    except Exception as e:
        logging.error(f"Error extracting features from instance {instance_path}: {str(e)}")
        return None, None


def process_instance_batch(instance_paths: List[str], 
                          task_type: str = 'obj') -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Process a batch of MILP instances, extract features, and prepare model inputs.
    
    Args:
        instance_paths: List of paths to MILP instance files
        task_type: Task type, 'fea' (feasibility), 'obj' (objective value), or 'sol' (solution)
        
    Returns:
        model_inputs: List of model input tensors
        labels: Array of labels
    """
    # Collect features and labels
    all_con_features = []
    all_var_features = []
    all_edge_indices = []
    all_edge_features = []
    all_n_cons = []
    all_n_vars = []
    all_labels = []
    
    # Process each instance
    valid_instances = []
    for instance_path in instance_paths:
        # Extract features
        solving_results, features = extract_milp_features(instance_path)
        
        if solving_results is None or features is None:
            logging.warning(f"Skipping instance {instance_path}")
            continue
        
        valid_instances.append(instance_path)
        
        # Collect features
        con_features = features['con_features']
        var_features = features['var_features']
        edge_indices = features['edge_indices']
        edge_features = features['edge_features']
        
        # Add features
        n_cons = con_features.shape[0]
        n_vars = var_features.shape[0]
        
        # Determine labels
        if task_type == 'fea':
            # Feasibility task
            label = 1.0 if solving_results['is_feasible'] else 0.0
            all_labels.append(label)
        elif task_type == 'obj':
            # Objective value task
            if not solving_results['is_feasible']:
                logging.warning(f"Instance {instance_path} is infeasible, skipping")
                continue
            label = solving_results['obj']
            all_labels.append(label)
        elif task_type == 'sol':
            # Solution value task
            if not solving_results['is_feasible']:
                logging.warning(f"Instance {instance_path} is infeasible, skipping")
                continue
            # TODO: 实现解值任务的标签提取
            logging.warning("Solution value task is not implemented fully")
            continue
            
        # Collect features
        all_con_features.append(con_features)
        all_var_features.append(var_features)
        all_n_cons.append(n_cons)
        all_n_vars.append(n_vars)
        
        # Adjust edge indices, considering offsets
        if len(all_edge_indices) > 0:
            # Calculate current offset
            con_offset = sum(all_n_cons[:-1])
            var_offset = sum(all_n_vars[:-1])
            
            # Adjust edge indices
            edge_indices_offset = edge_indices.copy()
            edge_indices_offset[0, :] += con_offset
            edge_indices_offset[1, :] += var_offset
            
            all_edge_indices.append(edge_indices_offset)
            all_edge_features.append(edge_features)
        else:
            all_edge_indices.append(edge_indices)
            all_edge_features.append(edge_features)
    
    if len(valid_instances) == 0:
        logging.error("No valid instances")
        return None, None
    
    # Merge features
    c = np.vstack(all_con_features)
    v = np.vstack(all_var_features)
    ei = np.hstack(all_edge_indices)
    ev = np.vstack(all_edge_features)
    n_cs = np.array(all_n_cons)
    n_vs = np.array(all_n_vars)
    labels = np.array(all_labels)
    
    # Create model inputs
    model_inputs = [c, ei, ev, v, n_cs, n_vs]
    
    # Normalize objective values
    if task_type == 'obj':
        # Simple normalization, can be adjusted as needed
        mean = np.mean(labels)
        std = np.std(labels)
        if std > 0:
            labels = (labels - mean) / std
    
    return model_inputs, labels


def batch_instances(instance_paths: List[str], 
                   batch_size: int,
                   task_type: str = 'obj') -> List[Tuple[List[np.ndarray], np.ndarray]]:
    """
    Batch instances for processing.
    
    Args:
        instance_paths: List of paths to MILP instance files
        batch_size: Batch size
        task_type: Task type, 'fea' (feasibility), 'obj' (objective value), or 'sol' (solution)
        
    Returns:
        List of batches, each containing model inputs and labels
    """
    batches = []
    
    # Process instances in batches
    for i in range(0, len(instance_paths), batch_size):
        batch_paths = instance_paths[i:i+batch_size]
        model_inputs, labels = process_instance_batch(batch_paths, task_type)
        
        if model_inputs is not None and labels is not None:
            batches.append((model_inputs, labels))
    
    return batches


def save_processed_data(processed_data: List[Tuple[List[np.ndarray], np.ndarray]], 
                        output_dir: str,
                        task_type: str = 'obj'):
    """
    Save processed data.
    
    Args:
        processed_data: Processed data, each element containing model inputs and labels
        output_dir: Output directory
        task_type: Task type, 'fea' (feasibility), 'obj' (objective value), or 'sol' (solution)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Merge all batches
    all_c = []
    all_v = []
    all_ei = []
    all_ev = []
    all_n_cs = []
    all_n_vs = []
    all_labels = []
    
    con_offsets = []
    var_offsets = []
    
    con_offset = 0
    var_offset = 0
    
    for model_inputs, labels in processed_data:
        c, ei, ev, v, n_cs, n_vs = model_inputs
        
        # Record offsets
        con_offsets.append(con_offset)
        var_offsets.append(var_offset)
        
        # Update offsets
        con_offset += c.shape[0]
        var_offset += v.shape[0]
        
        # Collect data
        all_c.append(c)
        all_v.append(v)
        all_ei.append(ei)
        all_ev.append(ev)
        all_n_cs.append(n_cs)
        all_n_vs.append(n_vs)
        all_labels.append(labels)
    
    # Merge data
    c = np.vstack(all_c)
    v = np.vstack(all_v)
    ei = np.hstack(all_ei)
    ev = np.vstack(all_ev)
    n_cs = np.concatenate(all_n_cs)
    n_vs = np.concatenate(all_n_vs)
    labels = np.concatenate(all_labels)
    
    # Save data
    np.save(os.path.join(output_dir, f'con_features_{task_type}.npy'), c)
    np.save(os.path.join(output_dir, f'var_features_{task_type}.npy'), v)
    np.save(os.path.join(output_dir, f'edge_indices_{task_type}.npy'), ei)
    np.save(os.path.join(output_dir, f'edge_features_{task_type}.npy'), ev)
    np.save(os.path.join(output_dir, f'n_cons_{task_type}.npy'), n_cs)
    np.save(os.path.join(output_dir, f'n_vars_{task_type}.npy'), n_vs)
    np.save(os.path.join(output_dir, f'labels_{task_type}.npy'), labels)
    np.save(os.path.join(output_dir, f'con_offsets_{task_type}.npy'), np.array(con_offsets))
    np.save(os.path.join(output_dir, f'var_offsets_{task_type}.npy'), np.array(var_offsets))
    
    # Save metadata
    metadata = {
        'n_instances': len(n_cs),
        'n_constraints': int(np.sum(n_cs)),
        'n_variables': int(np.sum(n_vs)),
        'n_edges': ei.shape[1],
        'con_feature_dim': c.shape[1],
        'var_feature_dim': v.shape[1],
        'edge_feature_dim': ev.shape[1],
        'task_type': task_type
    }
    
    pd.DataFrame([metadata]).to_csv(os.path.join(output_dir, f'metadata_{task_type}.csv'), index=False)
    
    logging.info(f"Processed data saved to {output_dir}")
    logging.info(f"Processed {metadata['n_instances']} instances")
    logging.info(f"Constraints: {metadata['n_constraints']}")
    logging.info(f"Variables: {metadata['n_variables']}")
    logging.info(f"Edges: {metadata['n_edges']}")


def load_processed_data(data_dir: str, task_type: str = 'obj') -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load processed data.
    
    Args:
        data_dir: Data directory
        task_type: Task type, 'fea' (feasibility), 'obj' (objective value), or 'sol' (solution)
        
    Returns:
        model_inputs: List of model input tensors
        labels: Array of labels
    """
    # Load data
    c = np.load(os.path.join(data_dir, f'con_features_{task_type}.npy'))
    v = np.load(os.path.join(data_dir, f'var_features_{task_type}.npy'))
    ei = np.load(os.path.join(data_dir, f'edge_indices_{task_type}.npy'))
    ev = np.load(os.path.join(data_dir, f'edge_features_{task_type}.npy'))
    n_cs = np.load(os.path.join(data_dir, f'n_cons_{task_type}.npy'))
    n_vs = np.load(os.path.join(data_dir, f'n_vars_{task_type}.npy'))
    labels = np.load(os.path.join(data_dir, f'labels_{task_type}.npy'))
    
    # Create model inputs
    model_inputs = [c, ei, ev, v, n_cs, n_vs]
    
    return model_inputs, labels
