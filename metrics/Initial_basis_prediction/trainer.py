"""
MILP Initial Basis Prediction GNN Model Trainer
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union, Any

from .model import WeightedCELoss, compute_class_weights

logger = logging.getLogger(__name__)

class GNNTrainer:
    """GNN Model Trainer"""
    
    def __init__(self, model, config, device=None):
        """
        Initialize the trainer
        
        Args:
            model: GNN model
            config: Training configuration
            device: Training device
        """
        self.model = model
        self.config = config
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move the model to the device
        self.model.to(self.device)
        
        # Set the optimizer, use the learning rate in the configuration
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],  
            weight_decay=config['weight_decay']
        )
        
        # Set the learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=20,
            min_lr=1e-7, verbose=True
        )
        
        # Set the gradient clipping parameter
        self.grad_clip_value = 1.0
        
        # Set the loss function
        self.criterion = WeightedCELoss()
        
        # Training status tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.early_stopping = config.get('early_stopping', 50)
        
        # Model save path
        self.save_dir = config.get('save_dir', './models')
        os.makedirs(self.save_dir, exist_ok=True)
        
    def prepare_batch(self, batch_data):
        """
        Prepare batch data
        
        Args:
            batch_data: Batch data list
            
        Returns:
            Dict: Processed batch data
        """
        batch_size = len(batch_data)
        
        # Get the maximum number of variables and constraints
        max_vars = max(data['n_vars'] for data in batch_data)
        max_constrs = max(data['n_constrs'] for data in batch_data)
        
        # Initialize tensors
        batch_var_features = torch.zeros((batch_size, max_vars, 8))
        batch_constr_features = torch.zeros((batch_size, max_constrs, 8))
        batch_edge_index = []
        batch_edge_attr = []
        batch_var_labels = torch.zeros(batch_size, max_vars, dtype=torch.long)
        batch_constr_labels = torch.zeros(batch_size, max_constrs, dtype=torch.long)
        batch_var_mask = torch.zeros((batch_size, max_vars, 3))
        batch_constr_mask = torch.zeros((batch_size, max_constrs, 3))
        batch_var_counts = []
        batch_constr_counts = []
        
        # Process each instance
        for i, data in enumerate(batch_data):
            n_vars = data['n_vars']
            n_constrs = data['n_constrs']
            
            # Fill features
            batch_var_features[i, :n_vars] = torch.FloatTensor(data['var_features'])
            batch_constr_features[i, :n_constrs] = torch.FloatTensor(data['constr_features'])
            
            # Process edge indices and attributes
            edge_index = torch.LongTensor(data['edge_index'])
            edge_attr = torch.FloatTensor(data['edge_attr'])
            batch_edge_index.append(edge_index)
            batch_edge_attr.append(edge_attr)
            
            # Fill labels
            batch_var_labels[i, :n_vars] = torch.LongTensor(data['var_labels'])
            batch_constr_labels[i, :n_constrs] = torch.LongTensor(data['constr_labels'])
            
            # Use the mask generated in data_utils.py
            if 'var_mask' in data and 'constr_mask' in data:
                batch_var_mask[i, :n_vars] = torch.FloatTensor(data['var_mask'])
                batch_constr_mask[i, :n_constrs] = torch.FloatTensor(data['constr_mask'])
            else:
                # Compatible with old data, create a full zero mask if there is no mask (does not affect logits)
                batch_var_mask[i, :n_vars] = torch.zeros((n_vars, 3))
                batch_constr_mask[i, :n_constrs] = torch.zeros((n_constrs, 3))
            
            # Record counts
            batch_var_counts.append(n_vars)
            batch_constr_counts.append(n_constrs)
        
        return {
            'var_features': batch_var_features,
            'constr_features': batch_constr_features,
            'edge_index': batch_edge_index,
            'edge_attr': batch_edge_attr,
            'var_labels': batch_var_labels,
            'constr_labels': batch_constr_labels,
            'batch_var_mask': batch_var_mask,
            'batch_constr_mask': batch_constr_mask,
            'batch_var_counts': batch_var_counts,
            'batch_constr_counts': batch_constr_counts
        }
    
    
    def train_epoch(self, train_loader):
        """
        Train one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            float: Average training loss
        """
        self.model.train()
        total_loss = 0.0
        var_correct, var_total = 0, 0
        constr_correct, constr_total = 0, 0
        
        for batch_data in train_loader:
            self.optimizer.zero_grad()
            
            # Prepare batch data
            batch = self.prepare_batch(batch_data)
            var_features = batch['var_features'].to(self.device)
            constr_features = batch['constr_features'].to(self.device)
            var_labels = batch['var_labels'].to(self.device)
            constr_labels = batch['constr_labels'].to(self.device)
            var_mask = batch['batch_var_mask'].to(self.device)
            constr_mask = batch['batch_constr_mask'].to(self.device)
            var_counts = batch['batch_var_counts']
            constr_counts = batch['batch_constr_counts']
            
            # Forward propagation
            outputs = self.model(batch)
            var_probs = outputs['var_probs']
            constr_probs = outputs['constr_probs']
            var_logits = outputs['var_logits']
            constr_logits = outputs['constr_logits']
            
            # Calculate loss
            batch_loss = 0.0
            batch_size = len(batch_data)
            
            for i in range(batch_size):
                n_vars = var_counts[i]
                n_constrs = constr_counts[i]
                
                # Variable loss
                var_logits_i = var_logits[i, :n_vars]  
                var_labels_i = var_labels[i, :n_vars]
                var_weights = compute_class_weights(var_labels_i)
                var_sample_weights = torch.tensor([var_weights[lab] for lab in var_labels_i]).to(self.device)
                var_loss = self.criterion(var_logits_i, var_labels_i, var_sample_weights)
                
                # Constraint loss
                constr_logits_i = constr_logits[i, :n_constrs]  
                constr_labels_i = constr_labels[i, :n_constrs]
                constr_weights = compute_class_weights(constr_labels_i)
                constr_sample_weights = torch.tensor([constr_weights[lab] for lab in constr_labels_i]).to(self.device)
                constr_loss = self.criterion(constr_logits_i, constr_labels_i, constr_sample_weights)
                
                # Total loss
                instance_loss = var_loss + constr_loss
                batch_loss += instance_loss
                
                # Statistics accuracy
                var_pred = var_probs[i, :n_vars].argmax(dim=1)
                var_correct += (var_pred == var_labels_i).sum().item()
                var_total += n_vars
                
                constr_pred = constr_probs[i, :n_constrs].argmax(dim=1)
                constr_correct += (constr_pred == constr_labels_i).sum().item()
                constr_total += n_constrs
            
            # Average loss
            batch_loss /= batch_size
            
            # Backward propagation and optimization
            batch_loss.backward()
            
            # Gradient clipping to reduce gradient explosion risk
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
            
            self.optimizer.step()
            
            total_loss += batch_loss.item()
        
        # Calculate average loss and accuracy
        avg_loss = total_loss / len(train_loader)
        var_acc = var_correct / var_total if var_total > 0 else 0
        constr_acc = constr_correct / constr_total if constr_total > 0 else 0
        
        return avg_loss, var_acc, constr_acc
    
    def validate(self, val_loader):
        """
        Validate the model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            float: Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        var_correct, var_total = 0, 0
        constr_correct, constr_total = 0, 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                # Prepare batch data
                batch = self.prepare_batch(batch_data)
                var_features = batch['var_features'].to(self.device)
                constr_features = batch['constr_features'].to(self.device)
                var_labels = batch['var_labels'].to(self.device)
                constr_labels = batch['constr_labels'].to(self.device)
                var_mask = batch['batch_var_mask'].to(self.device)
                constr_mask = batch['batch_constr_mask'].to(self.device)
                var_counts = batch['batch_var_counts']
                constr_counts = batch['batch_constr_counts']
                
                # Forward propagation
                outputs = self.model(batch)
                var_probs = outputs['var_probs']
                constr_probs = outputs['constr_probs']
                var_logits = outputs['var_logits']
                constr_logits = outputs['constr_logits']
                
                # Calculate loss
                batch_loss = 0.0
                batch_size = len(batch_data)
                
                for i in range(batch_size):
                    n_vars = var_counts[i]
                    n_constrs = constr_counts[i]
                    
                    # Variable loss
                    var_logits_i = var_logits[i, :n_vars]  
                    var_labels_i = var_labels[i, :n_vars]
                    var_weights = compute_class_weights(var_labels_i)
                    var_sample_weights = torch.tensor([var_weights[lab] for lab in var_labels_i]).to(self.device)
                    var_loss = self.criterion(var_logits_i, var_labels_i, var_sample_weights)
                    
                    # Constraint loss
                    constr_logits_i = constr_logits[i, :n_constrs]  
                    constr_labels_i = constr_labels[i, :n_constrs]
                    constr_weights = compute_class_weights(constr_labels_i)
                    constr_sample_weights = torch.tensor([constr_weights[lab] for lab in constr_labels_i]).to(self.device)
                    constr_loss = self.criterion(constr_logits_i, constr_labels_i, constr_sample_weights)
                    
                    # Total loss
                    instance_loss = var_loss + constr_loss
                    batch_loss += instance_loss
                    
                    # Statistics accuracy
                    var_pred = var_probs[i, :n_vars].argmax(dim=1)
                    var_correct += (var_pred == var_labels_i).sum().item()
                    var_total += n_vars
                    
                    constr_pred = constr_probs[i, :n_constrs].argmax(dim=1)
                    constr_correct += (constr_pred == constr_labels_i).sum().item()
                    constr_total += n_constrs
                
                # Average loss
                batch_loss /= batch_size
                total_loss += batch_loss.item()
        
        # Calculate average loss and accuracy
        avg_loss = total_loss / len(val_loader)
        var_acc = var_correct / var_total if var_total > 0 else 0
        constr_acc = constr_correct / constr_total if constr_total > 0 else 0
        
        return avg_loss, var_acc, constr_acc
    
    def train(self, train_loader, val_loader, epochs=None):
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Training epochs, if None, use the value in the configuration
            
        Returns:
            Dict: Training results
        """
        epochs = epochs or self.config['epochs']
        logger.info(f"Begin training, {epochs} epochs in total")
        
        train_losses, val_losses = [], []
        train_var_accs, train_constr_accs = [], []
        val_var_accs, val_constr_accs = [], []
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # Train
            train_loss, train_var_acc, train_constr_acc = self.train_epoch(train_loader)
            train_losses.append(train_loss)
            train_var_accs.append(train_var_acc)
            train_constr_accs.append(train_constr_acc)
            
            # Validate
            val_loss, val_var_acc, val_constr_acc = self.validate(val_loader)
            val_losses.append(val_loss)
            val_var_accs.append(val_var_acc)
            val_constr_accs.append(val_constr_acc)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Record time
            epoch_time = time.time() - start_time
            
            # Print progress
            logger.info(f"Epoch {epoch}/{epochs} | "
                        f"Training loss: {train_loss:.4f} | "
                        f"Validation loss: {val_loss:.4f} | "
                        f"Variable accuracy: {val_var_acc:.4f} | "
                        f"Constraint accuracy: {val_constr_acc:.4f} | "
                        f"Time: {epoch_time:.2f}s")
            
            # Check if it is the best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # Save the best model
                save_path = os.path.join(self.save_dir, 'best_model.pth')
                self._save_model(save_path)
                logger.info(f"Save the best model to {save_path}")
            else:
                self.patience_counter += 1
                
            # Save checkpoint
            if epoch % 50 == 0:
                save_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
                self._save_model(save_path)
            
            # Early stopping
            if self.patience_counter >= self.early_stopping:
                logger.info(f"Early stopping: {self.patience_counter} epochs validation loss not improved")
                break
        
        # Load the best model
        best_model_path = os.path.join(self.save_dir, 'best_model.pth')
        self._load_model(best_model_path)
        
        logger.info(f"Training completed! The best model is at epoch {self.best_epoch}, validation loss: {self.best_val_loss:.4f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_var_accs': train_var_accs,
            'train_constr_accs': train_constr_accs,
            'val_var_accs': val_var_accs,
            'val_constr_accs': val_constr_accs,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss
        }
    
    def _save_model(self, path):
        """
        Save the model
        
        Args:
            path: Save path
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'config': self.config
        }, path)
    
    def _load_model(self, path):
        """
        Load the model
        
        Args:
            path: Model path
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_epoch = checkpoint['best_epoch']
