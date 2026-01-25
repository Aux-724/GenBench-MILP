"""
Common utility functions for the benchmark framework.
"""

import os
import random
import numpy as np
import torch
import community.community_louvain as community
import ecole
import gurobipy as gp
import networkx as nx
from scipy.sparse import coo_matrix
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

# Define variable feature names
VAR_FEATURES = ["objective", "is_type_binary", "is_type_integer", "is_type_implicit_integer", "is_type_continuous", "has_lower_bound", "has_upper_bound", "lower_bound", "upper_bound"]

VAR_TYPE_FEATURES = ["B", "I", "M", "C"]

def instance2graph(path: str, compute_features: bool = False):
    """
    Extract the bipartite graph from the instance file.
    
    Args:
        path: Path to the MILP instance file.
        compute_features: Whether to compute the features of the instance.
        
    Returns:
        A tuple containing the graph representation and optionally the features.
    """
    model = ecole.scip.Model.from_file(path)
    obs = ecole.observation.MilpBipartite().extract(model, True)
    constraint_features = obs.constraint_features
    edge_indices = np.array(obs.edge_features.indices, dtype=int)
    edge_features = obs.edge_features.values.reshape((-1,1))
    variable_features = obs.variable_features
    graph = [constraint_features, edge_indices, edge_features, variable_features]
    
    if not compute_features:
        return graph, None
    else:
        n_conss = len(constraint_features)
        n_vars = len(variable_features)
        n_cont_vars = np.sum(variable_features, axis=0)[1]
        
        lhs = coo_matrix((edge_features.reshape(-1), edge_indices), shape=(n_conss, n_vars)).toarray()
        rhs = constraint_features.flatten()
        obj = variable_features[:, 0].flatten()

        nonzeros = (lhs != 0)
        n_nonzeros = np.sum(nonzeros)
        lhs_coefs = lhs[np.where(nonzeros)]
        var_degree, cons_degree = nonzeros.sum(axis=0), nonzeros.sum(axis=1)

        nx_edge_indices = edge_indices.copy()
        nx_edge_indices[1] += nx_edge_indices[0].max() + 1
        pyg_graph = Data(
            x_s = constraint_features,
            x_t = variable_features,
            edge_index = torch.LongTensor(edge_indices),
            node_attribute = "bipartite"
        )
        pyg_graph.num_nodes = len(constraint_features) + len(variable_features)
        nx_graph = to_networkx(pyg_graph, to_undirected=True)

        features = {
            "instance": path,
 
            "n_conss": n_conss,
            "n_vars": n_vars,
            "n_cont_vars": n_cont_vars,
            "ratio_cont_vars": float(n_cont_vars / n_vars),

            "n_nonzeros": n_nonzeros,
            "coef_dens": float(len(edge_features) / (n_vars * n_conss)),

            "var_degree_mean": float(var_degree.mean()),
            "var_degree_std": float(var_degree.std()),
            "var_degree_min": float(var_degree.min()),
            "var_degree_max": float(var_degree.max()),

            "cons_degree_mean": float(cons_degree.mean()),
            "cons_degree_std": float(cons_degree.std()),
            "cons_degree_min": int(cons_degree.min()),
            "cons_degree_max": int(cons_degree.max()),

            "lhs_mean": float(lhs_coefs.mean()),
            "lhs_std": float(lhs_coefs.std()),
            "lhs_min": float(lhs_coefs.min()),
            "lhs_max": float(lhs_coefs.max()),

            "rhs_mean": float(rhs.mean()),
            "rhs_std": float(rhs.std()),
            "rhs_min": float(rhs.min()),
            "rhs_max": float(rhs.max()),

            "obj_mean": float(obj.mean()),
            "obj_std": float(obj.std()),
            "obj_min": float(obj.min()),
            "obj_max": float(obj.max()),

            "clustering": float(nx.average_clustering(nx_graph)),
            "modularity": float(community.modularity(community.best_partition(nx_graph), nx_graph)),
        }
        return graph, features

def solve_instance(path: str, time_limit: int = 300, threads: int = 1):
    """
    Solve the instance using Gurobi.
    
    Args:
        path: Path to the MILP instance file.
        time_limit: Time limit in seconds (None for no limit).
        threads: Number of threads to use.
        
    Returns:
        Dictionary containing solve results including feasibility.
    """
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    if time_limit is not None:
        env.setParam("TimeLimit", time_limit)
    env.setParam("Threads", threads)
    env.start()
    model = gp.read(path, env=env)
    model.optimize()

    # 检查是否可行
    is_feasible = False
    obj_val = float('inf')
    
    # 根据状态码判断可行性
    # 2: OPTIMAL - 找到最优解
    # 3: INFEASIBLE - 问题无解
    # 5: UNBOUNDED - 问题无界
    # 9: TIME_LIMIT - 达到时间限制
    status_map = {
        gp.GRB.OPTIMAL: "OPTIMAL",
        gp.GRB.INFEASIBLE: "INFEASIBLE",
        gp.GRB.UNBOUNDED: "UNBOUNDED",
        gp.GRB.INF_OR_UNBD: "INF_OR_UNBD",
        gp.GRB.TIME_LIMIT: "TIME_LIMIT",
        gp.GRB.SUBOPTIMAL: "SUBOPTIMAL",
        gp.GRB.LOADED: "LOADED",
        gp.GRB.INTERRUPTED: "INTERRUPTED",
        gp.GRB.NUMERIC: "NUMERIC",
        gp.GRB.NODE_LIMIT: "NODE_LIMIT",
        gp.GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        gp.GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        gp.GRB.CUTOFF: "CUTOFF",
    }
    
    if model.status == gp.GRB.OPTIMAL:
        is_feasible = True
        obj_val = model.objVal
    elif model.status == gp.GRB.INFEASIBLE:
        is_feasible = False
    elif model.status == gp.GRB.UNBOUNDED:
        is_feasible = False
    elif model.status == gp.GRB.TIME_LIMIT:
        # 如果超时但找到了可行解
        if model.SolCount > 0:
            is_feasible = True
            obj_val = model.objVal
    
    results = {
        "status": model.status,
        "status_name": status_map.get(model.status, f"UNKNOWN({model.status})"),
        "is_feasible": is_feasible,
        "obj": obj_val if is_feasible else float('inf'),
        "num_nodes": model.NodeCount,
        "num_sols": model.SolCount,
        "solving_time": model.Runtime,
    }
    return results

def set_seed(seed):
    """
    Set the random seed for reproducibility.
    
    Args:
        seed: The random seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Set CuDNN to be deterministic. Notice that this may slow down the training.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

def set_cpu_num(cpu_num):
    """
    Set the number of CPU cores to use.
    
    Args:
        cpu_num: Number of CPU cores to use.
    """
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)
