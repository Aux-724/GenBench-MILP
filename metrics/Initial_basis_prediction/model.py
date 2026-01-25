"""
基于GNN的MILP初始基预测模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union

logger = logging.getLogger(__name__)

class MLPLayer(nn.Module):
    """Multi-layer Perceptron layer"""
    
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        """
        Initialize MLP layer
        
        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            dropout: Dropout rate
        """
        super(MLPLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """Forward propagation"""
        x = self.fc(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x

class BipartiteMessagePassing(nn.Module):
    """Bipartite message passing layer"""
    
    def __init__(self, var_dim: int, constr_dim: int, hidden_dim: int, dropout: float = 0.1):
        """
        Initialize bipartite message passing layer
        
        Args:
            var_dim: Variable node feature dimension
            constr_dim: Constraint node feature dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super(BipartiteMessagePassing, self).__init__()
        
        # Variable node update
        self.var_update = nn.Sequential(
            MLPLayer(var_dim + hidden_dim, hidden_dim, dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Constraint node update
        self.constr_update = nn.Sequential(
            MLPLayer(constr_dim + hidden_dim, hidden_dim, dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, var_feats, constr_feats, edge_index, edge_attr):
        """
        Forward propagation
        
        Args:
            var_feats: Variable node features [n_vars, var_dim]
            constr_feats: Constraint node features [n_constrs, constr_dim]
            edge_index: Edge index [2, n_edges]
            edge_attr: Edge attributes [n_edges]
            
        Returns:
            Tuple: (Updated variable node features, Updated constraint node features)
        """
        # Get source and target nodes from edge index
        src_idx, dst_idx = edge_index
        
        # Variable to constraint message passing
        var_to_constr_msg = var_feats[src_idx] * edge_attr.unsqueeze(1)
        
        # Constraint node aggregation from variable messages
        constr_agg = torch.zeros_like(constr_feats)
        constr_agg.index_add_(0, dst_idx, var_to_constr_msg)
        
        # Constraint to variable message passing
        constr_to_var_msg = constr_feats[dst_idx] * edge_attr.unsqueeze(1)
        
        # Variable node aggregation from constraint messages
        var_agg = torch.zeros_like(var_feats)
        var_agg.index_add_(0, src_idx, constr_to_var_msg)
        
        # Update variable node embedding
        var_concat = torch.cat([var_feats, var_agg], dim=1)
        var_updated = self.var_update(var_concat)
        var_updated = var_feats + var_updated  # Residual connection
        
        # Update constraint node embedding
        constr_concat = torch.cat([constr_feats, constr_agg], dim=1)
        constr_updated = self.constr_update(constr_concat)
        constr_updated = constr_feats + constr_updated  # Residual connection
        
        return var_updated, constr_updated

class InitialBasisGNN(nn.Module):
    """Initial basis prediction GNN model for MILP"""
    
    def __init__(self, var_feat_dim: int = 8, constr_feat_dim: int = 8, 
                 hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.1):
        """
        Initialize GNN model
        
        Args:
            var_feat_dim: Variable node feature dimension
            constr_feat_dim: Constraint node feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of GNN layers
            dropout: Dropout rate
        """
        super(InitialBasisGNN, self).__init__()
        
        # Initial feature projection
        self.var_proj = MLPLayer(var_feat_dim, hidden_dim, dropout)
        self.constr_proj = MLPLayer(constr_feat_dim, hidden_dim, dropout)
        
        # Message passing layers
        self.mp_layers = nn.ModuleList([
            BipartiteMessagePassing(hidden_dim, hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.var_out = nn.Linear(hidden_dim, 3)  # 3 classes: lower bound, basis variable, upper bound
        self.constr_out = nn.Linear(hidden_dim, 3)  # 3 classes: lower bound, basis variable, upper bound
        
        # Use Kaiming uniform initialization
        self._init_weights()
        
    def forward(self, batch_data):
        """
        Foward propagation
        
        Args:
            batch_data: Batch data dictionary, containing:
                - var_features: Variable features [batch_size, n_vars, var_feat_dim]
                - constr_features: Constraint features [batch_size, n_constrs, constr_feat_dim]
                - edge_index: Edge index [batch_size, 2, n_edges]
                - edge_attr: Edge attributes [batch_size, n_edges]
                - batch_var_mask: Variable mask [batch_size, n_vars, 3]
                - batch_constr_mask: Constraint mask [batch_size, n_constrs, 3]
                
        Returns:
            Dict: Dictionary containing prediction results
        """
        device = next(self.parameters()).device
        
        # Get batch data
        var_features = batch_data['var_features'].to(device)
        constr_features = batch_data['constr_features'].to(device)
        edge_index = batch_data['edge_index'] 
        edge_attr = batch_data['edge_attr']    
        batch_var_mask = batch_data.get('batch_var_mask')
        batch_constr_mask = batch_data.get('batch_constr_mask')
        
        batch_size = var_features.size(0)
        var_probs, constr_probs = [], []
        var_logits_list, constr_logits_list = [], []
        
        # Process each batch instance separately
        for i in range(batch_size):
            # Get current instance data
            var_feats = var_features[i]
            constr_feats = constr_features[i]
            edges = edge_index[i]
            attrs = edge_attr[i]
            
            # Feature projection
            var_embed = self.var_proj(var_feats)
            constr_embed = self.constr_proj(constr_feats)
            
            # Move edge index and edge attributes to the same device
            edges = edges.to(device)
            attrs = attrs.to(device)
            
            # Message passing
            for mp_layer in self.mp_layers:
                var_embed, constr_embed = mp_layer(var_embed, constr_embed, edges, attrs)
            
            # Output layer
            var_logits = self.var_out(var_embed)
            constr_logits = self.constr_out(constr_embed)
            
            # Apply knowledge masking
            if batch_var_mask is not None:
                var_mask = batch_var_mask[i].to(device)
                var_logits = var_logits + var_mask
            
            if batch_constr_mask is not None:
                constr_mask = batch_constr_mask[i].to(device)
                constr_logits = constr_logits + constr_mask
            
            # Calculate probabilities
            var_prob = F.softmax(var_logits, dim=1)
            constr_prob = F.softmax(constr_logits, dim=1)
            
            # Add probabilities and logits to lists
            var_probs.append(var_prob)
            constr_probs.append(constr_prob)
            var_logits_list.append(var_logits)
            constr_logits_list.append(constr_logits)
        # Stack all instance predictions
        var_probs = torch.stack(var_probs)
        constr_probs = torch.stack(constr_probs)
        var_logits = torch.stack(var_logits_list)
        constr_logits = torch.stack(constr_logits_list)
        
        return {
            'var_probs': var_probs,
            'constr_probs': constr_probs,
            'var_logits': var_logits,
            'constr_logits': constr_logits
        }
    
    def _init_weights(self):
        """Initialize model weights to improve numerical stability"""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                # Use Kaiming uniform initialization
                nn.init.kaiming_uniform_(param, a=0, mode='fan_in', nonlinearity='relu')
            elif 'bias' in name:
                # Initialize bias to a small value
                nn.init.constant_(param, 0.01)
    
    def predict(self, batch_data):
        """
        Model prediction
        
        Args:
            batch_data: Batch data dictionary
            
        Returns:
            Dict: Dictionary containing prediction results
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(batch_data)
        return output
        
class WeightedCELoss(nn.Module):
    """Weighted cross-entropy loss"""
    
    def __init__(self, reduction: str = 'mean', label_smoothing: float = 0.1):
        """
        Initialize weighted cross-entropy loss
        
        Args:
            reduction: Reduction method ('none', 'mean', 'sum')
            label_smoothing: Label smoothing parameter
        """
        super(WeightedCELoss, self).__init__()
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(self, logits, targets, weights=None):
        """
        Calculate weighted cross-entropy loss
        
        Args:
            logits: Predicted logits [batch_size, num_classes]
            targets: Target classes [batch_size]
            weights: Class weights [batch_size]
            
        Returns:
            Loss value
        """
        # Ensure input values are valid
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            # If input contains NaN or Inf, truncate values
            logits = torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0)
        
        # Ensure targets are on the same device
        device = logits.device
        targets = targets.to(device)
        
        # Use label smoothing
        loss = F.cross_entropy(logits, targets, reduction='none', label_smoothing=self.label_smoothing)
        
        if weights is not None:
            # Ensure weights are on the same device
            weights = weights.to(device)
            loss = loss * weights
            
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"Unsupported reduction method: {self.reduction}")

def compute_class_weights(labels, num_classes=3):
    """
    Compute class weights (inverse of class frequency)
    
    Args:
        labels: Labels [batch_size]
        num_classes: Number of classes
        
    Returns:
        Class weights
    """
    class_counts = torch.zeros(num_classes, device=labels.device)
    for c in range(num_classes):
        class_counts[c] = (labels == c).sum().float()
    
    # Avoid division by zero
    class_counts[class_counts == 0] = 1
    
    # Calculate weights, add smoothing factor to prevent extreme values
    smoothing_factor = 0.1
    weights = 1.0 / (class_counts + smoothing_factor)
    
    # Clip weights to prevent large values
    max_weight = 10.0
    weights = torch.clamp(weights, max=max_weight)
    
    # Normalize weights
    weights = weights / weights.sum() * num_classes
    
    return weights

def get_basis_from_probs(var_probs, constr_probs, n_vars, n_constrs):
    """
    Get candidate basis from predicted probabilities
    
    According to the specified requirements:
    1. Select the indices with the highest probabilities based on p_{x,i}[2] and p_{s,j}[2]
    2. Form candidate basis (\mathcal{B}_x, \mathcal{B}_s)
    
    Args:
        var_probs: Variable probabilities [n_vars, 3]
        constr_probs: Constraint probabilities [n_constrs, 3]
        n_vars: Number of variables
        n_constrs: Number of constraints
        
    Returns:
        Tuple: (基变量索引列表, 基松弛索引列表)
    """
    # Get the probability of each variable/constraint as a basis
    # The index 1 represents the basis variable/basis slack
    var_basis_probs = var_probs[:, 1].cpu().numpy()  
    constr_basis_probs = constr_probs[:, 1].cpu().numpy()
    
    # Merge all variable and constraint information
    # Create an object array containing type and index for clearer operations
    candidates = []
    for i in range(n_vars):
        candidates.append({'type': 'var', 'index': i, 'prob': var_basis_probs[i]})
    
    for j in range(n_constrs):
        candidates.append({'type': 'constr', 'index': j, 'prob': constr_basis_probs[j]})
    
    # Sort by probability in descending order
    candidates.sort(key=lambda x: x['prob'], reverse=True)
    
    # Select the top m probabilities as the initial basis
    selected_candidates = candidates[:n_constrs]
    
    # Separate the basis variables and constraints
    B_x = [c['index'] for c in selected_candidates if c['type'] == 'var']
    B_s = [c['index'] for c in selected_candidates if c['type'] == 'constr']
    
    return B_x, B_s
