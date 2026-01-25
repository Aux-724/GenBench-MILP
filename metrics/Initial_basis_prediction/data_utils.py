"""
MILP data processing and feature extraction tools
"""
import os
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import scipy.sparse as sp
import logging
import pickle
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm import tqdm

logger = logging.getLogger(__name__)

class MILPInstance:
    """MILP instance data wrapper class"""
    
    def __init__(self, model_path: str):
        """
        Load MILP instance from file and extract necessary data
        
        Args:
            model_path: MILP instance file path (.mps, .lp, etc.)
        """
        self.model_path = model_path
        self.instance_name = os.path.basename(model_path)
        self.model = None
        self.A = None  # constraint matrix (m x n)
        self.b = None  # right-hand side vector (m x 1)
        self.c = None  # objective coefficients (n x 1)
        self.l_x = None  # variable lower bounds (n x 1)
        self.u_x = None  # variable upper bounds (n x 1)
        self.l_s = None  # slack lower bounds (m x 1)
        self.u_s = None  # slack upper bounds (m x 1)
        self.var_names = []  # variable names
        self.constr_names = []  # constraint names
        self.m = 0  # number of constraints
        self.n = 0  # number of variables
        
        # Extract data
        self._extract_data()
        
    def _extract_data(self):
        """Extract MILP model data"""
        try:
            # Load model
            env = gp.Env(empty=True)
            env.setParam('OutputFlag', 0)
            env.start()
            self.model = gp.read(self.model_path, env=env)
            
            # Get basic information
            variables = self.model.getVars()
            constraints = self.model.getConstrs()
            self.n = len(variables)
            self.m = len(constraints)
            self.var_names = [var.VarName for var in variables]
            self.constr_names = [constr.ConstrName for constr in constraints]
            
            # Extract objective coefficients
            self.c = np.zeros(self.n)
            obj = self.model.getObjective()
            for i, var in enumerate(variables):
                try:
                    # Use getCoeff method to get variable coefficients
                    self.c[i] = obj.getCoeff(var)
                except Exception:
                    # If error, keep coefficient as 0
                    pass
            
            # Extract variable bounds
            self.l_x = np.array([var.LB for var in variables])
            self.u_x = np.array([var.UB for var in variables])
            
            # Replace infinite values with a large number
            big_m = 1e10
            self.l_x[self.l_x <= -GRB.INFINITY] = -big_m
            self.u_x[self.u_x >= GRB.INFINITY] = big_m
            
            # Build constraint matrix and right-hand side vector
            rows, cols, values = [], [], []
            self.b = np.zeros(self.m)
            self.l_s = np.zeros(self.m)
            self.u_s = np.zeros(self.m)
            
            for i, constr in enumerate(constraints):
                self.b[i] = constr.RHS
                
                # Handle constraint sense
                sense = constr.Sense
                if sense == GRB.LESS_EQUAL:
                    self.l_s[i] = -big_m  # Slack lower bound
                    self.u_s[i] = 0.0     # Slack upper bound
                elif sense == GRB.GREATER_EQUAL:
                    self.l_s[i] = 0.0     # Slack lower bound
                    self.u_s[i] = big_m   # Slack upper bound
                elif sense == GRB.EQUAL:
                    self.l_s[i] = 0.0     # Slack lower bound
                    self.u_s[i] = 0.0     # Slack upper bound
                
                # Build constraint matrix
                try:
                    # Iterate over all variables in the model
                    for j, var in enumerate(variables):
                        try:
                            # Try to get the coefficient of the current variable in the constraint
                            coeff = self.model.getCoeff(constr, var)
                            if coeff != 0:  # Only add non-zero coefficients
                                rows.append(i)
                                cols.append(j)
                                values.append(coeff)
                        except Exception:
                            # If getting coefficient fails, skip this variable
                            pass
                except Exception as e:
                    logger.warning(f"Failed to extract coefficients for constraint {constr.ConstrName}: {str(e)}")
            
            # Create sparse constraint matrix
            self.A = sp.csr_matrix((values, (rows, cols)), shape=(self.m, self.n))
            
            logger.debug(f"Successfully extracted instance {self.instance_name}: {self.m} constraints, {self.n} variables")
            
        except Exception as e:
            logger.error(f"Failed to extract instance {self.instance_name}: {str(e)}")
            raise
            
    def get_optimal_basis(self) -> Tuple[List[int], List[int], List[int], List[int]]:
        """
        Get the optimal basis of the LP relaxation and generate label data
        
        Returns:
            Tuple[List[int], List[int], List[int], List[int]]: 
                (basis variable indices, basis slack indices, variable status labels, slack status labels)
        """
        try:
            # Create LP relaxation model
            relaxed = self.model.copy()
            for var in relaxed.getVars():
                var.setAttr(GRB.Attr.VType, GRB.CONTINUOUS)
            
            # Solve LP relaxation
            relaxed.optimize()
            
            if relaxed.Status != GRB.OPTIMAL:
                logger.warning(f"Instance {self.instance_name}: LP relaxation did not reach optimal solution, status: {relaxed.Status}")
                return [], [], [], []
            
            variables = relaxed.getVars()
            constraints = relaxed.getConstrs()
            
            # Get basis variables and non-basis variables
            B_x, B_s = [], []  # basis variable and basis slack indices
            var_status, constr_status = [], []  # variable and constraint status labels
            
            for i, var in enumerate(variables):
                status = var.VBasis
                if status == 0:  # basis variable
                    B_x.append(i)
                    var_status.append(1)  # Category 2: basis variable (index 1)
                elif status == -1:  # non-basis variable at lower bound
                    var_status.append(0)  # Category 1: at lower bound (index 0)
                elif status == -2:  # non-basis variable at upper bound
                    var_status.append(2)  # Category 3: at upper bound (index 2)
                else:
                    var_status.append(1)  # Default to basis variable
            
            for i, constr in enumerate(constraints):
                status = constr.CBasis
                if status == 0:  # basis slack
                    B_s.append(i)
                    constr_status.append(1)  # Category 2: basis slack
                elif status == -1:  # non-basis slack at lower bound
                    constr_status.append(0)  # Category 1: at lower bound
                elif status == -2:  # non-basis slack at upper bound
                    constr_status.append(2)  # Category 3: at upper bound
                else:
                    constr_status.append(1)  # Default to basis variable
            
            return B_x, B_s, var_status, constr_status
            
        except Exception as e:
            logger.error(f"Failed to get optimal basis for instance {self.instance_name}: {str(e)}")
            return [], [], [], []

class BipartiteGraphFeatureExtractor:
    """MILP instance bipartite graph feature extractor"""
    
    def __init__(self, milp_instance: MILPInstance):
        """
        Initialize feature extractor
        
        Args:
            milp_instance: MILP instance object
        """
        self.milp = milp_instance
        self.var_features = None  # variable node features
        self.constr_features = None  # constraint node features
        self.edge_index = None  # edge index
        self.edge_attr = None  # edge weight
        
    def extract_features(self):
        """Extract bipartite graph features"""
        self._extract_var_features()
        self._extract_constr_features()
        self._build_edge_data()
        
    def _extract_var_features(self):
        """Extract variable node features (8 dimensions)"""
        n = self.milp.n
        m = self.milp.m
        A = self.milp.A
        c = self.milp.c
        l_x = self.milp.l_x
        u_x = self.milp.u_x
        l_s = self.milp.l_s
        u_s = self.milp.u_s
        
        var_features = np.zeros((n, 8))
        
        # Feature 1: objective coefficients
        var_features[:, 0] = c
        
        # Feature 2: variable connectivity (column non-zero elements/constraint count)
        var_degrees = np.array(A.getnnz(axis=0)) / m
        var_features[:, 1] = var_degrees
        
        # Feature 3-4: similarity with constraint lower and upper bounds
        for i in range(n):
            col = A.getcol(i).toarray().flatten()
            if np.linalg.norm(col) > 0 and np.linalg.norm(l_s) > 0:
                var_features[i, 2] = np.dot(col, l_s) / (np.linalg.norm(col) * np.linalg.norm(l_s))
            if np.linalg.norm(col) > 0 and np.linalg.norm(u_s) > 0:
                var_features[i, 3] = np.dot(col, u_s) / (np.linalg.norm(col) * np.linalg.norm(u_s))
        
        # Feature 5-8: variable bounds
        big_m = 1e10
        for i in range(n):
            # Lower bound value and flag
            if l_x[i] > -big_m:
                var_features[i, 4] = l_x[i]  # Lower bound value
                var_features[i, 5] = 0       # Finite lower bound
            else:
                var_features[i, 4] = 0       # Lower bound value (when infinite, set to 0)
                var_features[i, 5] = -1      # Infinite lower bound
                
            # Upper bound value and flag
            if u_x[i] < big_m:
                var_features[i, 6] = u_x[i]  # Upper bound value
                var_features[i, 7] = 0       # Finite upper bound
            else:
                var_features[i, 6] = 0       # Upper bound value (when infinite, set to 0)
                var_features[i, 7] = 1       # Infinite upper bound
        
        # Standardize features
        for j in range(8):
            if j not in [5, 7]:  # Skip flag features
                column = var_features[:, j]
                if np.max(column) - np.min(column) > 0:
                    var_features[:, j] = (column - np.mean(column)) / (np.std(column) + 1e-8)
        
        self.var_features = var_features
        
    def _extract_constr_features(self):
        """Extract constraint node features (8 dimensions)"""
        n = self.milp.n
        m = self.milp.m
        A = self.milp.A
        c = self.milp.c
        l_x = self.milp.l_x
        u_x = self.milp.u_x
        l_s = self.milp.l_s
        u_s = self.milp.u_s
        
        constr_features = np.zeros((m, 8))
        
        # Feature 1: similarity with objective coefficients
        for i in range(m):
            row = A.getrow(i).toarray().flatten()
            if np.linalg.norm(row) > 0 and np.linalg.norm(c) > 0:
                constr_features[i, 0] = np.dot(row, c) / (np.linalg.norm(row) * np.linalg.norm(c))
        
        # Feature 2: constraint connectivity (number of non-zero elements/number of variables)
        constr_degrees = np.array(A.getnnz(axis=1)) / n
        constr_features[:, 1] = constr_degrees
        
        # Feature 3-4: similarity with variable lower and upper bounds
        for i in range(m):
            row = A.getrow(i).toarray().flatten()
            if np.linalg.norm(row) > 0 and np.linalg.norm(l_x) > 0:
                constr_features[i, 2] = np.dot(row, l_x) / (np.linalg.norm(row) * np.linalg.norm(l_x))
            if np.linalg.norm(row) > 0 and np.linalg.norm(u_x) > 0:
                constr_features[i, 3] = np.dot(row, u_x) / (np.linalg.norm(row) * np.linalg.norm(u_x))
        
        # Feature 5-8: slack bounds
        big_m = 1e10
        for i in range(m):
            # Lower bound value and flag
            if l_s[i] > -big_m:
                constr_features[i, 4] = l_s[i]  # Lower bound value
                constr_features[i, 5] = 0       # Finite lower bound
            else:
                constr_features[i, 4] = 0       # Lower bound value (when infinite, set to 0)
                constr_features[i, 5] = -1      # Infinite lower bound
                
            # Upper bound value and flag
            if u_s[i] < big_m:
                constr_features[i, 6] = u_s[i]  # Upper bound value
                constr_features[i, 7] = 0       # Finite upper bound
            else:
                constr_features[i, 6] = 0       # Upper bound value (when infinite, set to 0)
                constr_features[i, 7] = 1       # Infinite upper bound
        
        # Standardize features
        for j in range(8):
            if j not in [5, 7]:  # Skip flag features
                column = constr_features[:, j]
                if np.max(column) - np.min(column) > 0:
                    constr_features[:, j] = (column - np.mean(column)) / (np.std(column) + 1e-8)
        
        self.constr_features = constr_features
        
    def _build_edge_data(self):
        """Build edge data"""
        A = self.milp.A
        
        # Get non-zero element coordinates and values
        coo = A.tocoo()
        row_idx, col_idx = coo.row, coo.col
        values = coo.data
        
        # Build edge index (2 x E)    
        self.edge_index = np.vstack([col_idx, row_idx])  # From variable nodes to constraint nodes
        
        # Build edge weights
        self.edge_attr = values
        
    def _create_knowledge_masks(self):
        """
        Create knowledge masks to mask out impossible categories before Softmax
        
        Apply the following logic to variables and constraints:
        - If lower bound is -infinity, mask the first element (lower bound) as -infinity
        - If upper bound is +infinity, mask the third element (upper bound) as -infinity
        - Otherwise, mask elements as 0, do not affect logits
        
        Returns:
            (np.ndarray, np.ndarray): Variable mask and constraint mask
        """
        n = self.milp.n
        m = self.milp.m
        l_x = self.milp.l_x
        u_x = self.milp.u_x
        l_s = self.milp.l_s
        u_s = self.milp.u_s
        
        # Create mask: 3 categories - [lower bound, basis variable, upper bound]
        var_mask = np.zeros((n, 3))
        constr_mask = np.zeros((m, 3))
        
        # Large negative number represents -infinity
        NEG_INF = -1e10
        
        # Variable mask
        for i in range(n):
            # If lower bound is -infinity, this variable cannot be in the lower bound
            if l_x[i] == float('-inf') or l_x[i] <= -GRB.INFINITY:
                var_mask[i, 0] = NEG_INF
            
            # If upper bound is +infinity, this variable cannot be in the upper bound
            if u_x[i] == float('inf') or u_x[i] >= GRB.INFINITY:
                var_mask[i, 2] = NEG_INF
        
        # Constraint mask
        for j in range(m):
            # If slack lower bound is -infinity, this constraint cannot be in the lower bound
            if l_s[j] == float('-inf') or l_s[j] <= -GRB.INFINITY:
                constr_mask[j, 0] = NEG_INF
            
            # If slack upper bound is +infinity, this constraint cannot be in the upper bound
            if u_s[j] == float('inf') or u_s[j] >= GRB.INFINITY:
                constr_mask[j, 2] = NEG_INF
        
        return var_mask, constr_mask
    
    def get_graph_data(self) -> Dict[str, np.ndarray]:
        """
        Get bipartite graph data
        
        Returns:
            Dict: dictionary containing graph data
        """
        if self.var_features is None or self.constr_features is None:
            self.extract_features()
        
        # Create knowledge masks
        var_mask, constr_mask = self._create_knowledge_masks()
            
        return {
            'var_features': self.var_features,
            'constr_features': self.constr_features,
            'edge_index': self.edge_index,
            'edge_attr': self.edge_attr,
            'var_mask': var_mask,
            'constr_mask': constr_mask
        }

class MILPDatasetProcessor:
    """MILP dataset processor"""
    
    def __init__(self, instance_dir: str, cache_dir: Optional[str] = None):
        """
        Initialize dataset processor
        
        Args:
            instance_dir: MILP instance file directory
            cache_dir: feature cache directory, None for no caching
        """
        self.instance_dir = instance_dir
        self.cache_dir = cache_dir
        self.instances = []
        self.instance_paths = []
        
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        
        self._find_instances()
    
    def _find_instances(self):
        """Find MILP instance files"""
        valid_exts = ['.mps', '.lp', '.mps.gz', '.lp.gz']
        for root, _, files in os.walk(self.instance_dir):
            for file in files:
                if any(file.endswith(ext) for ext in valid_exts):
                    self.instance_paths.append(os.path.join(root, file))
        
        logger.info(f"Found {len(self.instance_paths)} MILP instances in {self.instance_dir}")
    
    def process_dataset(self, max_instances: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Process dataset, extract features
        
        Args:
            max_instances: maximum number of instances to process, None to process all
            
        Returns:
            List[Dict]: list of processed data
        """
        processed_data = []
        paths = self.instance_paths[:max_instances] if max_instances else self.instance_paths
        
        # Use tqdm to create progress bar, disable verbose logging
        logger.info(f"Processing dataset, {len(paths)} instances")
        pbar = tqdm(paths, desc="Processing MILP instances", unit="instance")
        
        for path in pbar:
            instance_name = os.path.basename(path)
            cache_path = None
            
            # Update progress bar description
            pbar.set_description(f"Processing MILP instance {instance_name}")
            
            if self.cache_dir:
                cache_path = os.path.join(self.cache_dir, f"{instance_name}.pkl")
            
            # Check cache
            if cache_path and os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                processed_data.append(data)
                continue
            
            try:
                # Extract instance data
                milp_instance = MILPInstance(path)
                
                # Extract optimal basis
                B_x, B_s, var_labels, constr_labels = milp_instance.get_optimal_basis()
                
                if not B_x or not B_s:
                    continue
                
                # Extract bipartite graph features
                feature_extractor = BipartiteGraphFeatureExtractor(milp_instance)
                graph_data = feature_extractor.get_graph_data()
                
                # Assemble data
                data = {
                    'instance_name': instance_name,
                    'instance_path': path,
                    'var_features': graph_data['var_features'],
                    'constr_features': graph_data['constr_features'],
                    'edge_index': graph_data['edge_index'],
                    'edge_attr': graph_data['edge_attr'],
                    'var_mask': graph_data['var_mask'],     # Add variable knowledge mask
                    'constr_mask': graph_data['constr_mask'], # Add constraint knowledge mask
                    'var_labels': np.array(var_labels),
                    'constr_labels': np.array(constr_labels),
                    'B_x': B_x,
                    'B_s': B_s,
                    'n_vars': milp_instance.n,
                    'n_constrs': milp_instance.m
                }
                
                processed_data.append(data)
                
                # Save processed data to cache
                if cache_path:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(data, f)
                
            except Exception as e:
                continue
                
        logger.info(f"Successfully processed {len(processed_data)}/{len(paths)} instances")
        return processed_data

def split_dataset(data_list: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.2, 
                  random_seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """
    Split dataset into training and validation sets
    
    Args:
        data_list: list of data
        train_ratio: ratio of training set
        val_ratio: ratio of validation set
        random_seed: random seed
        
    Returns:
        Tuple[List, List]: (training set, validation set)
    """
    assert train_ratio + val_ratio <= 1.0, "Training set and validation set ratio sum cannot exceed 1.0"
    
    data_size = len(data_list)
    indices = list(range(data_size))
    
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    
    train_size = int(train_ratio * data_size)
    val_size = int(val_ratio * data_size)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    
    train_data = [data_list[i] for i in train_indices]
    val_data = [data_list[i] for i in val_indices]
    
    logger.info(f"Dataset split completed: {len(train_data)} training instances, {len(val_data)} validation instances")
    
    return train_data, val_data
